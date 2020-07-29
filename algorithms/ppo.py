"""
Taken from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import torch
import torch.nn as nn
import torch.optim as optim

from pycave.bayes import GMM  # https://pycave.borchero.com/guides/quickstart.html#example
from torch.distributions import MultivariateNormal
from utils.math import js_divergence


class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 kld_coeff,
                 sni_coeff,
                 dist_matching_coeff,
                 num_components,
                 lr=None,
                 eps=None,
                 max_grad_norm=None):

        self.actor_critic = actor_critic

        # ppo parameters
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        # bottleneck parameters
        self.kld_coeff = kld_coeff
        # sni parameters
        self.sni_coeff = sni_coeff
        # distribution matching parameters
        self.dist_matching_coeff = dist_matching_coeff
        self.num_components= num_components

        # optimiser
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        """
        Update model using PPO.
        """
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # initialise epoch values
        total_loss_epoch = 0
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        kld_epoch = 0
        jsd_epoch = 0

        # iterate through experience stored in rollouts
        for _ in range(self.ppo_epoch):

            # get generator which batches experience stored in rollouts
            data_generator = rollouts.feed_forward_generator(advantages,
                                                             self.actor_critic.num_train_envs,
                                                             self.actor_critic.use_distribution_matching,
                                                             self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ, val_obs_batch = sample

                # reshape to do in a single forward pass for all steps
                values, action_dist, action_dist_bar, z, mu, std = self.actor_critic.get_action_dist(obs_batch)

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

                # calculate SNI loss
                if self.actor_critic.sni_type == 'dvib':
                    dist_entropy = self.sni_coeff*action_dist_bar.entropy().mean() + (1-self.sni_coeff)*dist_entropy

                    ratio = torch.exp(action_dist_bar.log_probs(actions_batch) - old_action_log_probs_batch)
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                    action_loss = self.sni_coeff*-torch.min(surr1, surr2).mean() + (1-self.sni_coeff)*action_loss

                # calculate kl divergence between p(z|s) and q(z) averaged over x
                if self.actor_critic.use_bottleneck:
                    kld = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1, dim=1).mean()
                else:
                    kld = torch.tensor(0.)

                # calculate Jensen-Shannon divergence
                if self.actor_critic.use_distribution_matching:
                    # forward pass using experience from validation
                    val_z, val_mu, val_std = self.actor_critic.encode(val_obs_batch)
                    # TODO: fix grad_fn for pycave
                    if self.num_components==1:
                        # fit single multivariate Gaussian using averaged means and stds from train and val
                        train_dist= MultivariateNormal(mu.mean(dim=0), torch.diag(std.mean(dim=0).pow(2)))
                        val_dist= MultivariateNormal(val_mu.mean(dim=0), torch.diag(val_std.mean(dim=0).pow(2)))
                        jsd = js_divergence(train_dist, val_dist)
                    elif self.num_components>1:
                        # fit multivariate mixture of Gaussians to z and val_z
                        train_gmm = GMM(num_components=self.num_components, num_features=mu.shape[1], covariance='diag')
                        train_gmm.fit(z)
                        val_gmm = GMM(num_components=self.num_components, num_features=mu.shape[1], covariance='diag')
                        val_gmm.fit(val_z)
                        jsd = js_divergence(train_gmm, val_gmm)
                else:
                    jsd = torch.tensor(0.)

                # update actor-critic using PPO
                # zero accumulated gradients
                self.optimizer.zero_grad()
                # calculate loss
                loss = action_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef + kld * self.kld_coeff + jsd * self.dist_matching_coeff
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
                kld_epoch += kld.item()
                jsd_epoch += jsd.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        # calculate losses for epoch
        total_loss_epoch /= num_updates
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        kld_epoch /= num_updates
        jsd_epoch /= num_updates

        return total_loss_epoch, value_loss_epoch, action_loss_epoch, dist_entropy_epoch, kld_epoch, jsd_epoch
