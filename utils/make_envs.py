"""
Based on: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
"""
import gym
from gym.spaces.box import Box
import torch
import numpy as np

from procgen import ProcgenEnv
from baselines.common.vec_env import VecEnvWrapper, VecExtractDictObs, VecMonitor, VecNormalize


def make_vec_envs(env_name, 
                  start_level,
                  num_levels,
                  distribution_mode,
                  seed,
                  num_processes, 
                  gamma, 
                  log_dir,
                  device, 
                  num_frame_stack):
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
    seed : `int`
        Random seed.
    num_processes : `int`
        How many training CPU processes to use (default: 16).
        This will give the number of environments to make.
    gamma : `float` or `NoneType`
        Discount factor for rewards.
        `None` when used for evaluation.
    log_dir : `str` or `NoneType`
        Directory to save agents logs.       
    device : `torch.device`
        CPU or GPU.
    num_frame_stack : `int`
        Number of frames to stack for VecFrameStack wrapper (default: 0).        

    Returns:
    --------
    env : 
        Vector of environments.
    """  
    if env_name in ['coinrun', 'maze']:
        # generate OpenAI Procgen environments.
        # note that we need to seed the envs, set_global_env does not do this
        envs = ProcgenEnv(num_envs=num_processes,
                          env_name=env_name, 
                          start_level=start_level, 
                          num_levels=num_levels, 
                          distribution_mode=distribution_mode,
                          rand_seed=seed)     

        # extract image from dict
        envs = VecExtractDictObs(envs, "rgb")  

        # re-order channels, (H,W,C) => (C,H,W). 
        # required for PyTorch convolution layers.
        envs = VecTransposeImage(envs)
    else:
        raise NotImplementedError

    if log_dir:
        # records:
        #  1. episode reward, 
        #  2. episode length
        #  3. episode time taken
        envs = VecMonitor(venv=envs, 
                          filename=log_dir,
                          keep_buf=100)

    # normalise the rewards during training but not during testing
    # we don't normalise the obs as the network does this /255.
    if gamma is not None:
        # training: normalise rewards but not the obs.
        envs = VecNormalize(envs, ob=False, gamma=gamma)            
    else:
        # eval: normalise neither reward nor obs.
        envs = VecNormalize(envs, ob=False, ret=False)  

    # wrapper to convert observation arrays to torch.tensors
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
            return np.transpose(image, (2, 0, 1))
        return np.transpose(image, (0, 3, 1, 2))

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
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        """
        Convert torch.tensor actions into numpy.array for envs.
        """
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        """
        Convert numpy.array observations into torch.tensor for policy network.
        Convert numpy.array rewards into torch.tensor for policy network.
        """      
        obs, reward, done, info = self.venv.step_wait()
        # convert obs to torch tensor
        obs = torch.from_numpy(obs).float().to(self.device)
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


# from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
# from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
# from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
# from baselines.common.vec_env import VecNormalize, VecMonitor

# def make_env(env_id,
#              rank,
#              log_dir,             
#              mode,
#              args):
#     """
#     Make a single environment.

#     Parameters:
#     -----------
#     env_id : `str`
#         Name of environment to train on.      
#     rank : `int`
#         ID of environment. 
#     log_dir : `str`
#         Directory to save agents logs.                     
#     mode : `int`
#         {train:1, eval:0}       
#     args : `argparse.Namespace`
#         The model and environment specific arguments for the experiment.        

#     Returns:
#     --------
#     env : 
#         The envrionement.
#     """  

#     def _thunk():

#         if env_id.startswith('procgen'):
#             # generate OpenAI Procgen environment.
#             # note that we need to seed the env, set_global_env does not do this
#             env = gym.make(env_id, 
#                            start_level=args.start_level, 
#                            num_levels=mode*args.num_levels, 
#                            distribution_mode=args.distribution_mode,
#                            rand_seed=args.seed+rank) 
#         else:       
#             raise NotImplementedError       

#         obs_shape = env.observation_space.shape
#         if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
#             # re-order channels, from (H,W,C) to (C,H,W). 
#             # it is required for PyTorch convolution layers.
#             env = TransposeImage(env)              

#         return env

#     return _thunk

    # # generate list of environments
    # envs = [make_env(env_id=env_name, 
    #                  rank=i,
    #                  log_dir=log_dir,
    #                  mode=mode,
    #                  args=args) 
    #         for i in range(num_processes)]
    # # vectorise environments
    # if len(envs) > 1:
    #     # create a multiprocess vectorised wrapper for multiple environments, 
    #     # distributing each environment to its own process
    #     envs = SubprocVecEnv(envs)
    # else:
    #     # create a simple vectorised wrapper for multiple environments,
    #     # calling each environment in sequence on the current Python process.
    #     envs = DummyVecEnv(envs)        

# class TransposeImage(gym.ObservationWrapper):
#     """
#     Taken from: https://github.com/dannysdeng/dqn-pytorch
#     """
#     def __init__(self, env=None):
#         super(TransposeImage, self).__init__(env)
#         obs_shape = self.observation_space.shape
#         self.observation_space = Box(self.observation_space.low[0, 0, 0],
#                                      self.observation_space.high[0, 0, 0],
#                                      [obs_shape[2], obs_shape[1], obs_shape[0]],
#                                      dtype=self.observation_space.dtype)

#     def observation(self, observation):
#         # observation is of type Tensor
#         return observation.transpose(2, 0, 1)           
