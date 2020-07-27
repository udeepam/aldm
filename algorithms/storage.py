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
                 action_space,
                 recurrent_hidden_state_size):

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
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)

        # masks that indicate whether it's a true terminal state or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

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
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.returns = self.returns.to(device)

    def insert(self, obs, rewards, actions, value_preds, action_log_probs,
               masks, recurrent_hidden_states, bad_masks):
        """
        Adding experience from timestep to buffer.
        """
        self.obs[self.step + 1].copy_(obs)
        self.rewards[self.step].copy_(rewards)
        self.actions[self.step].copy_(actions)
        self.value_preds[self.step].copy_(value_preds)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.masks[self.step + 1].copy_(masks)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.bad_masks[self.step + 1].copy_(bad_masks)
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
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

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

        # get lists of experience for training envs
        obs = self.obs[:, :num_train_envs]
        recurrent_hidden_states = self.recurrent_hidden_states[:, :num_train_envs]
        actions = self.actions[:, :num_train_envs]
        value_preds = self.value_preds[:, :num_train_envs]
        returns = self.returns[:, :num_train_envs]
        masks = self.masks[:, :num_train_envs]
        action_log_probs = self.action_log_probs[:, :num_train_envs]

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes for training ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_train_envs, num_steps, num_train_envs * num_steps, num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size,
                               drop_last=True)

        if use_distribution_matching:
            val_batch_size = (num_processes - num_train_envs) * num_steps
            # get lists of experience for validation envs
            val_obs = self.obs[:, num_train_envs:]
            val_recurrent_hidden_states = self.recurrent_hidden_states[:, num_train_envs:]
            val_masks = self.masks[:, num_train_envs:]
            val_sampler = BatchSampler(SubsetRandomSampler(range(val_batch_size)),
                                       int(val_batch_size / (batch_size // mini_batch_size)),
                                       drop_last=True)
        else:
            val_sampler = sampler

        for indices, val_indices in zip(sampler, val_sampler):
            # reshape experience from training envs
            obs_batch = obs[:-1].reshape(-1, *obs.size()[2:])[indices]
            recurrent_hidden_states_batch = recurrent_hidden_states[:-1].reshape(-1, recurrent_hidden_states.size(-1))[indices]
            actions_batch = actions.reshape(-1, actions.size(-1))[indices]
            value_preds_batch = value_preds[:-1].reshape(-1, 1)[indices]
            return_batch = returns[:-1].reshape(-1, 1)[indices]
            masks_batch = masks[:-1].reshape(-1, 1)[indices]
            old_action_log_probs_batch = action_log_probs.reshape(-1, 1)[indices]

            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            if use_distribution_matching:
                # reshape experience from validation envs
                val_obs_batch = val_obs[:-1].reshape(-1, *val_obs.size()[2:])[val_indices]
                val_recurrent_hidden_states_batch = val_recurrent_hidden_states[:-1].reshape(-1, val_recurrent_hidden_states.size(-1))[val_indices]
                val_masks_batch = val_masks[:-1].reshape(-1, 1)[val_indices]
            else:
                val_obs_batch = val_recurrent_hidden_states_batch = val_masks_batch = None

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, \
                val_obs_batch, val_recurrent_hidden_states_batch, val_masks_batch

    def recurrent_generator(self, advantages, num_train_envs, use_distribution_matching, num_mini_batch):
        """
        TODO: need to adapt for --percentage_train currently only work if percentahe_train=1.0
        """
        if use_distribution_matching:
            raise NotImplementedError

        num_processes = self.rewards.size(1)

        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))

        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)

        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = list()
            recurrent_hidden_states_batch = list()
            actions_batch = list()
            value_preds_batch = list()
            return_batch = list()
            masks_batch = list()
            old_action_log_probs_batch = list()
            adv_targ = list()

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
