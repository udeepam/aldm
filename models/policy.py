"""
Based on: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

Impala CNN based on: https://github.com/facebookresearch/torchbeast
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import helpers as utl
from utils import distributions as utl_dist


class Policy(nn.Module):
    def __init__(self, 
                 obs_shape, 
                 action_space, 
                 base=None, 
                 base_kwargs=None):
        """
        Actor-critic network.
        """
        super(Policy, self).__init__()
        
        if base_kwargs is None:
            base_kwargs = dict()

        # initialise actor-critic networks
        if base in ['coinrun', 'maze']:
            self.base = ImpalaCNN(obs_shape[0], action_space.n, **base_kwargs)
        else:
            raise NotImplementedError

        # intialise output distributions of the actor network
        if action_space.__class__.__name__ == "Discrete":
            self.dist = utl_dist.FixedCategorical
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        """
        Check if actor-critic is uses GRU.
        """
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """
        Size of rnn_hx.
        """
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        """
        Receive input from environment and return value, action, action_log_probs
        """
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(logits=actor_features)    

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        """
        Receive input from environment and return value
        """
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(logits=actor_features)                             

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        """
        Base network class.

        Arguments:
        ----------
        recurrent : `Boolean`
            Whether to use a recurrent network.
        recurrent_input_size : `int`
            The number of expected features in the input for GRU.
        hidden_size : `int`
            The number of features in the hidden state of GRU (default: 256).        
        """
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            # define GRU
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            # initialise parameters
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
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
            outputs = []
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


class ImpalaCNN(NNBase):
    def __init__(self, obs_shape, num_actions, recurrent=False, hidden_size=256):
        """
        Impala CNN.

        For helpful image of network: 
        https://medium.com/aureliantactics/custom-models-with-baselines-impala-cnn-cnns-with-features-and-contra-3-hard-mode-811dbdf2dff9 

        Arguments:
        ----------
        obs_shape : `int`
            Number of channels in the input image.
        num_actions : `int`
            Size of actions space (Discrete) which is the output of actor network.
        recurrent : `Boolean`
            Whether to use a recurrent network.
        hidden_size : `int`
            The number of features in the hidden state of GRU (default: 256).
        """
        super(ImpalaCNN, self).__init__(recurrent, hidden_size, hidden_size)
        self.obs_shape = obs_shape
        
        self.feat_convs = list()
        self.resnet1 = list()
        self.resnet2 = list()      

        init_conv2d_ = lambda m: utl.init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu')) 
        init_critic_ = lambda m: utl.init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        init_actor_  = lambda m: utl.init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01) 

        input_channels = self.obs_shape
        for num_ch in [16, 32, 32]:
            feats_convs = list()
            feats_convs.append(init_conv2d_(nn.Conv2d(in_channels=input_channels,
                                                      out_channels=num_ch,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1)))

            feats_convs.append(nn.MaxPool2d(kernel_size=3, 
                                            stride=2, 
                                            padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = list()
                resnet_block.append(nn.ReLU())
                resnet_block.append(init_conv2d_(nn.Conv2d(in_channels=input_channels,
                                                           out_channels=num_ch,
                                                           kernel_size=3,
                                                           stride=1,
                                                           padding=1)))
                resnet_block.append(nn.ReLU())
                resnet_block.append(init_conv2d_(nn.Conv2d(in_channels=input_channels,
                                                           out_channels=num_ch,
                                                           kernel_size=3,
                                                           stride=1,
                                                           padding=1)))
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        # define Impala CNN
        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2) 
        self.fc = init_critic_(nn.Linear(2048, hidden_size))

        # define actor model
        self.actor = init_actor_(nn.Linear(hidden_size, num_actions))
        # define critic model
        self.critic = init_critic_(nn.Linear(hidden_size, 1))

        # put model into train mode
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        # normalise observations
        x = inputs.float() / 255.0  # x.shape = (n_envs,C,H,W)

        # forward pass through Impala CNN
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input                    
        x = F.relu(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc(x))

        if self.is_recurrent:
            # forward pass through GRU
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)   

        return self.critic(x), self.actor(x), rnn_hxs
