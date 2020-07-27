"""
Based on: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
"""
import os
import time
import datetime
from collections import deque

import torch

import wandb

from utils.make_envs import make_vec_envs
from utils import helpers as utl
from utils import math as utl_math
from utils import evaluation as utl_eval
from utils import test as utl_test
from utils import representation as utl_rep

from models.policy import ACModel
from algorithms.ppo import PPO
from algorithms.storage import RolloutStorage

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Learner:
    """
    Learner
    """
    def __init__(self, args):
        """
        Takes environment specific arguments to train policy.

        Arguments:
        ----------
        args: `argparse.Namespace`
            The model and environment specific arguments for the experiment.
        """
        self.args = args

        # Weights & Biases logger
        if self.args.log:

            if self.args.run_name is None:
                # make run name as {env_name}_{TIME}
                now = datetime.datetime.now().strftime('_%d-%m_%H:%M:%S')
                self.args.run_name = self.args.env_name+'_'+self.args.algo+now

            # initialise wandb
            wandb.init(name=self.args.run_name,
                       project=self.args.proj_name,
                       group=self.args.group_name,
                       config=self.args,
                       monitor_gym=True)

            # save wandb dir path
            self.args.run_dir = wandb.run.dir

            # make directory for saving models
            self.save_dir = os.path.join(wandb.run.dir, 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            # make directory for logging train envs
            self.train_log_dir = os.path.join(wandb.run.dir, 'train')
            if not os.path.exists(self.train_log_dir):
                os.makedirs(self.train_log_dir)

            if self.args.eval_interval:
                # make directory for logging eval envs
                self.eval_log_dir = os.path.join(wandb.run.dir, 'eval')
                if not os.path.exists(self.eval_log_dir):
                    os.makedirs(self.eval_log_dir)

        else:
            self.train_log_dir = None
            self.eval_log_dir  = None

        # set random seed of random, torch and numpy
        utl.set_global_seed(self.args.seed, self.args.deterministic_execution)

        # initialise environments for training
        self.envs = make_vec_envs(env_name=self.args.env_name,
                                  start_level=self.args.train_start_level,
                                  num_levels=self.args.train_num_levels,
                                  distribution_mode=self.args.distribution_mode,
                                  paint_vel_info=self.args.paint_vel_info,
                                  num_processes=self.args.num_processes,
                                  log_dir=self.train_log_dir,
                                  device=device,
                                  num_frame_stack=self.args.num_frame_stack,
                                  use_distribution_matching=self.args.use_distribution_matching,
                                  percentage_levels_train=self.args.percentage_levels_train,
                                  num_val_envs=self.args.num_val_envs)

        # initialise policy network
        self.initialise_policy()

        # count number of frames and updates
        self.frames   = 0
        self.iter_idx = 0


    def initialise_policy(self):
        # initialise policy network
        actor_critic = ACModel(obs_shape=self.envs.observation_space.shape,
                               action_space=self.envs.action_space,
                               hidden_size=self.args.hidden_size,
                               base=self.args.env_name,
                               use_bottleneck=self.args.use_bottleneck,
                               sni_type=self.args.sni_type,
                               use_distribution_matching=self.args.use_distribution_matching,
                               num_train_envs=self.args.num_train_envs,
                               recurrent=self.args.recurrent_policy).to(device)

        # initialise policy trainer
        if self.args.algo == 'ppo':
            self.policy = PPO(actor_critic=actor_critic,
                              clip_param=self.args.policy_clip_param,
                              ppo_epoch=self.args.policy_ppo_epoch,
                              num_mini_batch=self.args.policy_num_mini_batch,
                              value_loss_coef=self.args.policy_value_loss_coef,
                              entropy_coef=self.args.policy_entropy_coef,
                              kld_coeff=self.args.dvib_beta,
                              sni_coeff=self.args.sni_coeff,
                              dist_matching_coeff=self.args.dist_matching_coeff,
                              num_components=self.args.num_components,
                              lr=self.args.policy_lr,
                              eps=self.args.policy_eps,
                              max_grad_norm=self.args.policy_max_grad_norm)
        else:
            raise NotImplementedError

        # initialise rollout storage for the policy
        self.rollouts = RolloutStorage(num_steps=self.args.policy_num_steps,
                                       num_processes=self.args.num_processes,
                                       obs_shape=self.envs.observation_space.shape,
                                       action_space=self.envs.action_space,
                                       recurrent_hidden_state_size=self.policy.actor_critic.recurrent_hidden_state_size)

    def train(self):
        """
        Given some stream of environments, train the policy.
        """
        # Weights & Biases logger
        if self.args.log:
            wandb.config.update(self.args)
            wandb.watch(self.policy.actor_critic, log="all")

        start_time = time.time()

        # reset environments
        obs = self.envs.reset()  # obs.shape = (n_envs,C,H,W)

        # insert initial observation to rollout storage
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(device)

        # initialise buffer for calculating mean episodic returns
        episode_info_buf = deque(maxlen=10)

        # calculate number of updates
        # number of frames ÷ number of policy steps before update ÷ number of cpu processes used for train envs
        self.args.num_batch = self.args.num_train_envs * self.args.policy_num_steps
        self.args.num_updates = int(self.args.num_frames) // self.args.num_batch

        print("Number of updates: ", self.args.num_updates)
        for self.iter_idx in range(self.args.num_updates):
            print("Iter: ", self.iter_idx)

            # put actor-critic into train mode
            self.policy.actor_critic.train()

            # rollout policy to collect num_batch of experience and store in storage
            for step in range(self.args.policy_num_steps):

                # sample actions from policy
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.policy.actor_critic.act(self.rollouts.obs[step],
                                                                                                           self.rollouts.recurrent_hidden_states[step],
                                                                                                           self.rollouts.masks[step],
                                                                                                           train=True)

                # observe rewards and next obs
                obs, reward, done, infos = self.envs.step(action)

                # log episode info if episode finished, need logging on in VecMonitor
                for info in infos:
                    if 'episode' in info.keys():
                        episode_info_buf.append(info['episode'])

                # create mask for episode ends
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)
                # bad_mask is true if episode ended because time limit was reached - ignore for procgen
                bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos]).to(device)

                # add experience to policy buffer
                self.rollouts.insert(obs,
                                     reward,
                                     action,
                                     value,
                                     action_log_prob,
                                     masks,
                                     recurrent_hidden_states,
                                     bad_masks)

                self.frames += self.args.num_train_envs

            # --- UPDATE ---
            # here the policy is updated for good average performance across tasks.

            # bootstrap next value prediction
            with torch.no_grad():
                next_value = self.policy.actor_critic.get_value(self.rollouts.obs[-1],
                                                                self.rollouts.recurrent_hidden_states[-1],
                                                                self.rollouts.masks[-1]).detach()

            # compute returns for current rollouts
            self.rollouts.compute_returns(next_value,
                                          self.args.policy_gamma,
                                          self.args.policy_gae_lambda)

            # update actor-critic using policy gradient algo
            total_loss, value_loss, action_loss, dist_entropy, kld, jsd = self.policy.update(self.rollouts)

            # clean up after update
            self.rollouts.after_update()

            # --- EVALUATE POLICY ----

            if self.args.eval_interval is not None and self.iter_idx % self.args.eval_interval == 0:
                eval_episode_info_buf = utl_eval.evaluate(args=self.args,
                                                          actor_critic=self.policy.actor_critic,
                                                          eval_log_dir=self.eval_log_dir,
                                                          device=device)

            # --- LOGGING ---

            if self.args.log and self.iter_idx % self.args.log_interval == 0:
                end_time = time.time()
                # Calculates if value function is a good predicator of the returns (ev > 1)
                # or if it's just worse than predicting nothing (ev =< 0)
                ev = utl_math.explained_variance(utl.sf01(self.rollouts.value_preds),
                                                 utl.sf01(self.rollouts.returns))
                wandb.log({'misc/updates': self.iter_idx,
                           'misc/timesteps': self.frames,
                           'misc/time_elapsed': end_time - start_time,
                           'misc/fps': int(self.frames/ (end_time - start_time)),
                           'misc/explained_variance': float(ev),
                           'train/mean_episodic_return': utl_math.safe_mean([episode_info['r'] for episode_info in episode_info_buf]),
                           'train/mean_episodic_length': utl_math.safe_mean([episode_info['l'] for episode_info in episode_info_buf]),
                           'losses/total_loss': total_loss,
                           'losses/val_loss': value_loss,
                           'losses/action_loss': action_loss,
                           'losses/dist_entropy': dist_entropy,
                           'losses/kld': kld,
                           'losses/jsd': jsd}, step=self.iter_idx)

                if self.args.eval_interval is not None:
                    wandb.log({'eval/mean_episodic_return': utl_math.safe_mean([episode_info['r'] for episode_info in eval_episode_info_buf]),
                               'eval/mean_episodic_length': utl_math.safe_mean([episode_info['l'] for episode_info in eval_episode_info_buf])}, step=self.iter_idx)

            # --- SAVE MODELS ---

            # save for every interval-th episode or for the last epoch
            if self.args.log and (self.iter_idx % self.args.save_interval == 0 or self.iter_idx == self.args.num_updates - 1):
                torch.save(self.policy.actor_critic.state_dict(), os.path.join(self.save_dir, "policy{0}.h5".format(self.iter_idx)))


    def test(self):
        # Weights & Biases logger
        if self.args.log:
            # make directory for logging test envs
            test_log_dir = os.path.join(wandb.run.dir, 'test')
            if not os.path.exists(test_log_dir):
                os.makedirs(test_log_dir)
        else:
            test_log_dir = None

        mean_episodic_return = utl_test.test(args=self.args,
                                             actor_critic=self.policy.actor_critic,
                                             test_log_dir=test_log_dir,
                                             device=device)

        if self.args.log:
            wandb.run.summary["train_mean_episodic_return"] = mean_episodic_return[0]
            wandb.run.summary["test_mean_episodic_return"]  = mean_episodic_return[1]


    def analyse_representation(self):
        # TODO: methods for analysing the latent representation
        # Weights & Biases logger
        if self.args.log:
            # make directory for logging test envs
            rep_log_dir = os.path.join(wandb.run.dir, 'representations')
            if not os.path.exists(rep_log_dir):
                os.makedirs(rep_log_dir)
        else:
            rep_log_dir = None

        utl_rep.analyse_representation(args=self.args,
                                       actor_critic=self.policy.actor_critic,
                                       rollouts=self.rollouts)
