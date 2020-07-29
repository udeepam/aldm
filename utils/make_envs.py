"""
Based on: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
"""
import gym
from gym.spaces.box import Box
import torch
import numpy as np

from procgen import ProcgenEnv

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env import VecEnvWrapper, VecMonitor, VecNormalize, VecExtractDictObs


def make_env(env_name,
             start_level,
             num_levels,
             distribution_mode,
             paint_vel_info):
    """
    Make a single environment.
    """
    def _thunk():

        if env_name.startswith('procgen'):
            # generate OpenAI Procgen environment.
            env = gym.make(env_name,
                           start_level=start_level,
                           num_levels=num_levels,
                           distribution_mode=distribution_mode,
                           paint_vel_info=paint_vel_info)
        else:
            raise NotImplementedError

        return env

    return _thunk


def make_vec_envs(env_name,
                  start_level,
                  num_levels,
                  distribution_mode,
                  paint_vel_info,
                  num_processes,
                  log_dir,
                  device,
                  num_frame_stack,
                  use_distribution_matching=False,
                  percentage_levels_train=None,
                  num_val_envs=None):
    """
    Make vector of environments.

    Parameters:
    -----------
    env_name : `str`
        Name of environment to train on.
    start_level : `int`
        The point in the list of levels available to the environment at which to index into.
    num_levels : `int`
        The number of unique levels that can be generated. Set to 0 to use unlimited levels.
    distribution_mode : `str`
        What variant of the levels to use {easy, hard, extreme, memory, exploration}.
    paint_vel_info : `Boolean`
        Paint player velocity info in the top left corner. Only supported by certain games.
    num_processes : `int`
        How many training CPU processes to use (default: 64).
        This will give the number of environments to make.
    log_dir : `str`
        Directory to save agents logs.
    device : `torch.device`
        CPU or GPU.
    num_frame_stack : `int`
        Number of frames to stack for VecFrameStack wrapper (default: 0).
    use_distribution_matching : `Boolean`
        Whether current model is using distribution matching.
    percentage_levels_train : `float` or `NoneType`
        Proportion of the train levels to use for train and the rest is used for validation.
    num_val_envs : `int` or `NoneType`
        Number of environments from num_processes to use for validation and the rest are used for train.

    Returns:
    --------
    env :
        Vector of environments.
    """
    # generate list of environments
    if use_distribution_matching:
        train_start_level = start_level
        train_num_levels = int(num_levels * percentage_levels_train)
        val_start_level = start_level + train_num_levels
        val_num_levels = num_levels - train_num_levels

        # make the (num_processes-num_val_envs) train_envs and  num_val_envs val_envs
        envs = [make_env(env_name=env_name,
                         start_level=train_start_level if i<=num_processes-num_val_envs-1 else val_start_level,
                         num_levels=train_num_levels if i<=num_processes-num_val_envs-1 else val_num_levels,
                         distribution_mode=distribution_mode,
                         paint_vel_info=paint_vel_info) for i in range(num_processes)]

        # vectorise environments
        if len(envs) > 1:
            # create a multiprocess vectorised wrapper for multiple environments,
            # distributing each environment to its own process
            envs = SubprocVecEnv(envs)
        else:
            # create a simple vectorised wrapper for multiple environments,
            # calling each environment in sequence on the current Python process.
            envs = DummyVecEnv(envs)

    else:
        if env_name == "procgen:procgen-coinrun-v0":
            env_name = "coinrun"
        envs = ProcgenEnv(num_envs=num_processes,
                          env_name=env_name,
                          start_level=start_level,
                          num_levels=num_levels,
                          distribution_mode=distribution_mode,
                          paint_vel_info=paint_vel_info)

        # extract image from dict
        envs = VecExtractDictObs(envs, "rgb")

    # re-order channels, (H,W,C) => (C,H,W).
    # required for PyTorch convolution layers.
    envs = VecTransposeImage(envs)

    # records:
    #  1. episode reward,
    #  2. episode length
    #  3. episode time taken
    envs = VecMonitor(venv=envs,
                      filename=log_dir,
                      keep_buf=100)

    # normalise the rewards
    envs = VecNormalize(envs, ob=False)

    # wrapper to convert observation arrays to torch.Tensors
    # normalise observations / 255.
    envs = VecPyTorch(envs, device)

    # Frame stacking wrapper for vectorized environment
    if num_frame_stack !=0:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)

    return envs


class VecTransposeImage(VecEnvWrapper):
    """
    Based on: https://github.com/DLR-RM/stable-baselines3
    Re-order channels, from (H,W,C) to (C,H,W).
    It is required for PyTorch convolution layers.
    """

    def __init__(self, venv):
        height, width, channels = venv.observation_space.shape
        observation_space = Box(low=0,
                                high=255,
                                shape=(channels, height, width),
                                dtype=venv.observation_space.dtype)
        super(VecTransposeImage, self).__init__(venv, observation_space=observation_space)

    @staticmethod
    def transpose_image(image):
        """
        Transpose an image or batch of images (re-order channels).
        :param image: (np.ndarray)
        :return: (np.ndarray)
        """
        if len(image.shape) == 3:
            return image.transpose(2, 0, 1)
        return image.transpose(0, 3, 1, 2)

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        return self.transpose_image(observations), rewards, dones, infos

    def reset(self):
        """
        Reset all environments
        """
        return self.transpose_image(self.venv.reset())

    def close(self):
        self.venv.close()


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """
        Taken from: https://github.com/harry-uglow/Curriculum-Reinforcement-Learning

        Converts array of observations to Tensors. This makes them
        usable as input to a PyTorch policy network.
        """
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        """
        Convert numpy.array observations into torch.tensor for policy network.
        """
        obs = self.venv.reset()
        # convert obs to torch tensor
        obs = torch.from_numpy(obs).float().to(self.device) / 255.
        return obs

    def step_async(self, actions):
        """
        Convert torch.tensor actions into numpy.array for envs.
        """
        if isinstance(actions, torch.LongTensor) or len(actions.shape) > 1:
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        """
        Convert numpy.array observations into torch.tensor for policy network.
        Convert numpy.array rewards into torch.tensor for policy network.
        """
        obs, reward, done, info = self.venv.step_wait()
        # convert obs to torch tensor
        obs = torch.from_numpy(obs).float().to(self.device) / 255.
        # convert reward to torch tensor
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecPyTorchFrameStack(VecEnvWrapper):
    """
    Derived from: https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
    """
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()


# from procgen import ProcgenEnv
# from baselines.common.vec_env import VecEnvWrapper, VecExtractDictObs, VecMonitor, VecNormalize
# def make_vec_envs(env_name,
#                   start_level,
#                   num_levels,
#                   distribution_mode,
#                   paint_vel_info,
#                   num_processes,
#                   log_dir,
#                   device,
#                   num_frame_stack):
#     """
#     Make vector of environments.

#     Parameters:
#     -----------
#     env_name : `str`
#         Name of environment to train on.
#     start_level : `int`
#         The point in the list of levels available to the environment at which to index into.
#     num_levels : `int`
#         The number of unique levels that can be generated. Set to 0 to use unlimited levels.
#     distribution_mode : `str`
#         What variant of the levels to use {easy, hard, extreme, memory, exploration}.
#     paint_vel_info : `Boolean`
#         Paint player velocity info in the top left corner. Only supported by certain games.
#     num_processes : `int`
#         How many training CPU processes to use (default: 16).
#         This will give the number of environments to make.
#     log_dir : `str` or `NoneType`
#         Directory to save agents logs.
#     device : `torch.device`
#         CPU or GPU.
#     num_frame_stack : `int`
#         Number of frames to stack for VecFrameStack wrapper (default: 0).

#     Returns:
#     --------
#     env :
#         Vector of environments.
#     """
#     if env_name in ['coinrun', 'maze']:
#         # generate OpenAI Procgen environments.
#         envs = ProcgenEnv(num_envs=num_processes,
#                           env_name=env_name,
#                           start_level=start_level,
#                           num_levels=num_levels,
#                           distribution_mode=distribution_mode,
#                           paint_vel_info=paint_vel_info)

#         # extract image from dict
#         envs = VecExtractDictObs(envs, "rgb")

#         # re-order channels, (H,W,C) => (C,H,W).
#         # required for PyTorch convolution layers.
#         envs = VecTransposeImage(envs)
#     else:
#         raise NotImplementedError

#     # records:
#     #  1. episode reward,
#     #  2. episode length
#     #  3. episode time taken
#     envs = VecMonitor(venv=envs,
#                       filename=log_dir,
#                       keep_buf=100)

#     # normalise the rewards
#     # we don't normalise the obs as the network does this /255.
#     envs = VecNormalize(envs, ob=False)

#     # wrapper to convert observation arrays to torch.tensors
#     envs = VecPyTorch(envs, device)

#     # Frame stacking wrapper for vectorized environment
#     if num_frame_stack !=0:
#         envs = VecPyTorchFrameStack(envs, num_frame_stack, device)

#     return envs
