"""
Based on: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import torch.nn as nn

from models.impala_cnn import ImpalaCNN
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
                 use_bottleneck=False,
                 sni_type=None):
        """
        Actor-critic network.
        """
        super(ACModel, self).__init__()

        self.num_actions = action_space.n

        # decide which components are enabled
        self.use_bottleneck = use_bottleneck
        self.sni_type = sni_type

        # define feature extractor
        self.feature_extractor = ImpalaCNN(num_inputs=obs_shape[0])

        # define intermediate layer: bottleneck or linear layer
        if use_bottleneck:
            self.reg_layer = Bottleneck(input_size=2048,
                                        output_size=hidden_size)
        else:
            self.reg_layer = nn.Sequential(init_relu_(nn.Linear(2048, hidden_size)),
                                           nn.ReLU(inplace=True))

        # define critic model
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        # define actor model
        self.actor_linear = init_actor_(nn.Linear(hidden_size, self.num_actions))

        # intialise output distribution of the actor network
        if action_space.__class__.__name__ == "Discrete":
            self.dist = utl_dist.FixedCategorical
        else:
            raise NotImplementedError

        # put model into train mode
        self.train()

    def encode(self, inputs):
        # forward pass through feature extractor
        x = self.feature_extractor(inputs)

        # forward pass through intermediate layer
        if self.use_bottleneck:
            z, mu, std = self.reg_layer(x)
        else:
            z  = self.reg_layer(x)
            mu = std = None

        return z, mu, std

    def act(self, inputs):
        """
        Receive input from environment and return value, action, action_log_probs
        """
        z, mu, _ = self.encode(inputs)

        if self.sni_type is not None:
            # for any SNI type, the rollout values and actor features are determinisitic
            value = self.critic_linear(mu)
            actor_features = self.actor_linear(mu)
        else:
            value = self.critic_linear(z)
            actor_features = self.actor_linear(z)

        # create action distribution
        dist = self.dist(logits=actor_features)
        # sample actions
        action = dist.sample()
        # get action log probabilities from distribution
        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, z

    def get_value(self, inputs):
        """
        Receive input from environment and return value.
        """
        z, mu, _ = self.encode(inputs)
        if self.sni_type is not None:
            # for any SNI type, the rollout values are determinisitic
            value = self.critic_linear(mu)
        else:
            value = self.critic_linear(z)
        return value

    def get_action_dist(self, inputs):
        """
        Forward pass for policy gradient algorithm update.
        """
        z, mu, std = self.encode(inputs)

        actor_features = self.actor_linear(z)
        dist = self.dist(logits=actor_features)

        if self.sni_type == 'vib':
            # need both policies for training, but still only one value function:
            value = self.critic_linear(mu)
            actor_features = self.actor_linear(mu)
            dist_bar = self.dist(logits=actor_features)
        else:
            value = self.critic_linear(z)
            dist_bar = None

        return value, dist, dist_bar, z, mu, std

    def get_analysis(self, inputs):
        """
        Forward pass for policy gradient algorithm update.
        """
        z, mu, std = self.encode(inputs)

        if self.sni_type is not None:
            # for any SNI type, the rollout values and actor features are determinisitic
            actor_features = self.actor_linear(mu)
        else:
            actor_features = self.actor_linear(z)

        # create action distribution
        dist = self.dist(logits=actor_features)
        # sample actions
        action = dist.sample()

        return action, z, mu, std
