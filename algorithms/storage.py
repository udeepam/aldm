"""
Taken from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self,
                 num_steps,
                 num_processes,
                 obs_shape,
                 action_space):

        # initialise list for storing batch of experience to train on
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)

        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_processes, action_shape)

        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()

        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)

        # for counting
        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        """
        Send lists to device.
        """
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.actions = self.actions.to(device)
        self.value_preds = self.value_preds.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.masks = self.masks.to(device)
        self.returns = self.returns.to(device)

    def insert(self, obs, rewards, actions, value_preds, action_log_probs, masks):
        """
        Adding experience from timestep to buffer.
        """
        self.obs[self.step + 1].copy_(obs)
        self.rewards[self.step].copy_(rewards)
        self.actions[self.step].copy_(actions)
        self.value_preds[self.step].copy_(value_preds)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.masks[self.step + 1].copy_(masks)
        # increment step counter
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        """
        Update first element of some of the lists as when episode
        ends new episode immediately starts.
        No need to reinitialise the lists.
        """
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self,
                        next_value,
                        gamma,
                        gae_lambda):
        """
        Compute the returns for accumulated rollouts.
        """
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.num_steps)):
            delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator(self,
                               advantages,
                               num_train_envs,
                               use_distribution_matching,
                               num_mini_batch=None,
                               mini_batch_size=None):
        """
        Batches experience stored in lists.
        """
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_train_envs * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes for training ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_train_envs, num_steps, num_train_envs * num_steps, num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        # get lists of experience for training envs and reshape
        obs = self.obs[:, :num_train_envs][:-1].reshape(-1, *self.obs[:, :num_train_envs].size()[2:])
        actions = self.actions[:, :num_train_envs].reshape(-1, self.actions[:, :num_train_envs].size(-1))
        value_preds = self.value_preds[:, :num_train_envs][:-1].reshape(-1, 1)
        returns = self.returns[:, :num_train_envs][:-1].reshape(-1, 1)
        old_action_log_probs = self.action_log_probs[:, :num_train_envs].reshape(-1, 1)
        # create train indices sampler
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size,
                               drop_last=True)

        if use_distribution_matching:
            val_batch_size = (num_processes - num_train_envs) * num_steps
            # get lists of experience for validation envs and reshape
            val_obs = self.obs[:, num_train_envs:][:-1].reshape(-1, *self.obs[:, num_train_envs:].size()[2:])
            # create validaiton indices sampler
            val_sampler = BatchSampler(SubsetRandomSampler(range(val_batch_size)),
                                       int(val_batch_size / (batch_size // mini_batch_size)),
                                       drop_last=True)
        else:
            val_sampler = torch.zeros(int(batch_size/mini_batch_size))

        for indices, val_indices in zip(sampler, val_sampler):
            # reshape experience from training envs
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            old_action_log_probs_batch = old_action_log_probs[indices]

            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            if use_distribution_matching:
                # reshape experience from validation envs
                val_obs_batch = val_obs[val_indices]
            else:
                val_obs_batch = None

            yield obs_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ, val_obs_batch
