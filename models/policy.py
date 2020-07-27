"""
Based on: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import torch
import torch.nn as nn

from models.impala_cnn import ResNetBase
from models.bottleneck import Bottleneck

from utils import helpers as utl
from utils import distributions as utl_dist


init_ = lambda m: utl.init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
init_actor_ = lambda m: utl.init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
init_relu_ = lambda m: utl.init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))


class ACModel(nn.Module):
    def __init__(self,
                 obs_shape,
                 action_space,
                 hidden_size=256,
                 base=None,
                 use_bottleneck=False,
                 sni_type=None,
                 use_distribution_matching=False,
                 num_train_envs=0,
                 recurrent=False):
        """
        Actor-critic network.
        """
        super(ACModel, self).__init__()

        # decide which components are enabled
        self.use_bottleneck = use_bottleneck
        self.is_recurrent = recurrent
        self.sni_type = sni_type
        self.use_distribution_matching = use_distribution_matching

        # parameters
        self.recurrent_hidden_state_size = hidden_size
        self.num_train_envs = num_train_envs

        # define feature extractor
        if base.startswith('procgen'):
            self.feature_extractor = ResNetBase(num_inputs=obs_shape[0],
                                                num_actions=action_space.n)
        else:
            raise NotImplementedError

        # define intermediate layer: bottleneck or linear layer
        if use_bottleneck:
            self.reg_layer = Bottleneck(input_size=2048,
                                        output_size=hidden_size)
        else:
            self.reg_layer = nn.Sequential(init_relu_(nn.Linear(2048, hidden_size)),
                                           nn.ReLU(inplace=True))

        # define rnn
        if recurrent:
            self.rnn = RNN(input_size=hidden_size,
                           hidden_size=hidden_size)

        # define critic model
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        # define actor model
        self.actor_linear = init_actor_(nn.Linear(hidden_size, action_space.n))

        # intialise output distribution of the actor network
        if action_space.__class__.__name__ == "Discrete":
            self.dist = utl_dist.FixedCategorical
        else:
            raise NotImplementedError

        # put model into train mode
        self.train()

    def encode(self, inputs, rnn_hxs, masks):
        # forward pass through feature extractor
        x = self.feature_extractor(inputs)

        if self.use_bottleneck:
            # bottlneck: x are actually z
            z, mu, std = self.reg_layer(x)
            x = z
        else:
            x  = self.reg_layer(x)
            z = mu = std = None

        if self.is_recurrent:
            # forward pass through RNN
            x, rnn_hxs = self.rnn(x, rnn_hxs, masks)

        return x, rnn_hxs, z, mu, std

    def act(self, inputs, rnn_hxs, masks, train=False):
        """
        Receive input from environment and return value, action, action_log_probs

        If sni_type is not None and use_distribution_matching=True, we will only
        correct the actor_features and not values, for validation
        i.e. they should use x not mu, as the values for validation are not used.
        """
        x, rnn_hxs, _, mu, _ = self.encode(inputs, rnn_hxs, masks)

        if train and self.sni_type is not None:
            # for any SNI type, the rollout values and actor features are determinisitic
            value = self.critic_linear(mu)
            actor_features = self.actor_linear(mu)
            if self.use_distribution_matching:
                # agent in validation envs should not use deterministic actor features
                actor_features = torch.cat([actor_features[:self.num_train_envs, :],
                                            self.actor_linear(x[self.num_train_envs:, :])],
                                           dim=0)
        else:
            value = self.critic_linear(x)
            actor_features = self.actor_linear(x)

        # create action distribution
        dist = self.dist(logits=actor_features)
        # sample actions
        action = dist.sample()
        # get action log probabilities from distribution
        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        """
        Receive input from environment and return value.

        If sni_type is not None and use_distribution_matching=True, we do not need to
        correct the values for validation, i.e. they should use x not mu, as the values
        for validation are not used.
        """
        x, _, _, mu, _ = self.encode(inputs, rnn_hxs, masks)
        if self.sni_type is not None:
            # for any SNI type, the rollout values are determinisitic
            value = self.critic_linear(mu)
        else:
            value = self.critic_linear(x)
        return value

    def get_action_dist(self, inputs, rnn_hxs, masks):
        """
        Forward pass for policy gradient algorithm update.
        """
        x, _, z, mu, std = self.encode(inputs, rnn_hxs, masks)

        actor_features = self.actor_linear(x)
        dist = self.dist(logits=actor_features)

        if self.sni_type == 'dvib':
            # need both policies for training, but still only one value function:
            value = self.critic_linear(mu)
            actor_features = self.actor_linear(mu)
            dist_bar = self.dist(logits=actor_features)
        else:
            value = self.critic_linear(x)
            dist_bar = None

        return value, dist, dist_bar, z, mu, std


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Base network class.

        Arguments:
        ----------
        input_size : `int`
            The number of expected features in the input for GRU.
        hidden_size : `int`
            The number of features in the hidden state of GRU (default: 256).
        """
        super(RNN, self).__init__()

        self._hidden_size = hidden_size

        # define GRU
        self.gru = nn.GRU(input_size, hidden_size)

        # initialise parameters
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flattened to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = list()
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs
