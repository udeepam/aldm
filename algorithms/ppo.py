"""
Taken from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence
from utils.math import js_divergence


class PPO():
    def __init__(self,
                 actor_critic,
                 ppo_epoch,
                 num_mini_batch,
                 clip_param,
                 value_loss_coef,
                 entropy_coef,
                 max_grad_norm,
                 lr,
                 eps,
                 vib_coef,
                 sni_coef,
                 use_dist_matching,
                 dist_matching_loss,
                 dist_matching_coef,
                 num_train_envs,
                 num_val_envs):

        self.actor_critic = actor_critic

        # ppo parameters
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm

        # bottleneck parameters
        self.vib_coef = vib_coef
        # sni parameters
        self.sni_coef = sni_coef
        # distribution matching parameters
        self.use_dist_matching = use_dist_matching
        self.dist_matching_loss = dist_matching_loss
        self.dist_matching_coef = dist_matching_coef
        self.num_train_envs = num_train_envs
        self.num_val_envs = num_val_envs

        # optimiser
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        """
        Update model using PPO.
        """
        advantages = rollouts.returns[:, :self.num_train_envs][:-1] - rollouts.value_preds[:, :self.num_train_envs][:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # initialise epoch values
        total_loss_epoch = 0
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        vib_kl_epoch = 0
        dist_matching_loss_epoch = 0

        # iterate through experience stored in rollouts
        for _ in range(self.ppo_epoch):

            # get generator which batches experience stored in rollouts
            data_generator = rollouts.feed_forward_generator(advantages,
                                                             self.num_train_envs,
                                                             self.num_val_envs,
                                                             self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ, val_obs_batch, val_actions_batch = sample

                # --- PPO ---
                values, action_dist, action_dist_bar, z, mu, std = self.actor_critic.get_action_dist(obs_batch)

                # calculate entropy
                dist_entropy = action_dist.entropy().mean()

                ratio = torch.exp(action_dist.log_probs(actions_batch) - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                # get value loss
                value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                # --- BOTTLENECK ---

                if self.actor_critic.use_bottleneck:
                    # calculate kl divergence between p(z|s) and q(z) averaged over s
                    vib_kl = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1, dim=1).mean()
                else:
                    vib_kl = torch.tensor(0.)

                # --- DISTRIBUTION MATCHING ---

                dist_matching_loss = torch.tensor(0.)
                if self.num_val_envs>0:

                    # forward pass of validation observations
                    val_z, _, _ = self.actor_critic.encode(val_obs_batch)

                    # get KL between train and val for each action
                    for i in range(self.actor_critic.num_actions):
                        train_z_action = z[(actions_batch==i).squeeze(), :]
                        val_z_action   = val_z[(val_actions_batch==i).squeeze(), :]
                        if len(train_z_action)<2 or len(val_z_action)<2:
                            # must have more than 1 data point for fitting a multivariate Gaussian
                            pass
                        else:
                            # fit single multivariate Gaussian using the mean of the z's and the std of the z's from train and val
                            eps = 1e-3
                            train_dist = MultivariateNormal(train_z_action.mean(dim=0), scale_tril=torch.diag(train_z_action.std(dim=0)+eps))
                            val_dist  = MultivariateNormal(val_z_action.mean(dim=0), scale_tril=torch.diag(val_z_action.std(dim=0)+eps))
                            # calculate distribution matching loss either using KL divergence or Jensen-Shannon divergence
                            if self.dist_matching_loss == "kl":
                                div_loss = kl_divergence(train_dist, val_dist).mean() * (len(train_z_action)+len(val_z_action)) / (len(z)+len(val_z))
                            elif self.dist_matching_loss == "jsd":
                                div_loss = js_divergence(train_dist, val_dist).mean() * (len(train_z_action)+len(val_z_action)) / (len(z)+len(val_z))
                            if torch.isnan(div_loss):
                                pass
                            else:
                                dist_matching_loss = dist_matching_loss + div_loss
                dist_matching_loss = dist_matching_loss if self.use_dist_matching else dist_matching_loss.detach()

                # --- SNI ---

                if self.actor_critic.sni_type == 'vib':
                    dist_entropy = self.sni_coef*action_dist_bar.entropy().mean() + (1-self.sni_coef)*dist_entropy

                    ratio = torch.exp(action_dist_bar.log_probs(actions_batch) - old_action_log_probs_batch)
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                    action_loss = self.sni_coef*-torch.min(surr1, surr2).mean() + (1-self.sni_coef)*action_loss

                # zero accumulated gradients
                self.optimizer.zero_grad()
                # calculate loss
                loss = action_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef + vib_kl * self.vib_coef + dist_matching_loss * self.dist_matching_coef
                # backpropogate: calculate gradients
                loss.backward()
                # clippling
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                # update parameters of model
                self.optimizer.step()

                # update epoch values
                total_loss_epoch += loss.item()
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                vib_kl_epoch += vib_kl.item()
                dist_matching_loss_epoch += dist_matching_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        # calculate losses for epoch
        total_loss_epoch /= num_updates
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        vib_kl_epoch /= num_updates
        dist_matching_loss_epoch /= num_updates

        return total_loss_epoch, value_loss_epoch, action_loss_epoch, dist_entropy_epoch, vib_kl_epoch, dist_matching_loss_epoch
