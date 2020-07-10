"""
Based on: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
"""
import os
from collections import deque

import torch

import wandb

from utils.make_envs import make_vec_envs


class Evaluator:
    def __init__(self, args, recurrent_hidden_state_size, device):
        self.args = args
        self.device = device  

        # Weights & Biases logger 
        if self.args.log:  
            # make directory for logging eval envs
            self.eval_log_dir = os.path.join(wandb.run.dir, 'eval')   
            if not os.path.exists(self.eval_log_dir):
                os.makedirs(self.eval_log_dir)
        else:
            self.eval_log_dir = None            

        # initialise environments for evaluation
        self.eval_envs = make_vec_envs(env_name=self.args.env_name, 
                                       start_level=0, 
                                       num_levels=0, 
                                       distribution_mode=self.args.distribution_mode, 
                                       paint_vel_info=self.args.paint_vel_info,
                                       seed=self.args.seed,                                                                        
                                       num_processes=self.args.num_processes, 
                                       log_dir=self.eval_log_dir,
                                       device=device, 
                                       recurrent=self.args.recurrent_policy,
                                       num_frame_stack=self.args.num_frame_stack)     

        # reset environments
        self.obs = self.eval_envs.reset()  # obs.shape = (n_env,C,H,W)
        self.obs = self.obs.to(self.device)

        # initialisations 
        self.recurrent_hidden_states = torch.zeros(self.args.num_processes, 
                                                   recurrent_hidden_state_size, 
                                                   device=self.device)
        self.masks = torch.zeros(self.args.num_processes, 1, device=self.device)  

        # initialise buffer for calculating means
        self.eval_episode_info_buf = deque(maxlen=100)            

    def evaluate(self, actor_critic):
        # put actor-critic into evaluation mode 
        actor_critic.eval()

        # rollout policy to collect num_batch of experience and store in storage 
        for step in range(self.args.policy_num_steps):
            # sample actions from policy
            with torch.no_grad():
                _, action, _, self.recurrent_hidden_states = actor_critic.act(self.obs, 
                                                                              self.recurrent_hidden_states,
                                                                              self.masks,
                                                                              deterministic=True)                                                                                     

            # observe rewards and next obs
            self.obs, _, done, infos = self.eval_envs.step(action)       

            # log episode info if finished
            # need logging on for VecMonitor
            for info in infos:
                if 'episode' in info.keys():
                    self.eval_episode_info_buf.append(info['episode'])   
                    
            # create mask for episode ends
            self.masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(self.device)               

        return self.eval_episode_info_buf      


# def evaluate(args,
#              step, 
#              actor_critic, 
#              eval_log_dir,
#              device):

#     # put actor-critic into evaluation mode 
#     actor_critic.eval()

#     # initialise environments for evaluation
#     eval_envs = make_vec_envs(env_name=args.env_name, 
#                               start_level=0, 
#                               num_levels=0, 
#                               distribution_mode=args.distribution_mode, 
#                               paint_vel_info=args.paint_vel_info,
#                               seed=args.seed + step,                                                                        
#                               num_processes=args.num_processes, 
#                               log_dir=eval_log_dir,
#                               device=device, 
#                               recurrent=args.recurrent_policy,
#                               num_frame_stack=args.num_frame_stack)     

#     # initialise buffer for calculating means
#     eval_episode_info_buf = list()                       

#     # reset environments
#     obs = eval_envs.reset()  # obs.shape = (n_env,C,H,W)
#     obs = obs.to(device)

#     # initialisations 
#     recurrent_hidden_states = torch.zeros(args.num_processes, 
#                                           actor_critic.recurrent_hidden_state_size, 
#                                           device=device)
#     masks = torch.zeros(args.num_processes, 1, device=device)  

#     # collect returns from 10 full episodes
#     while len(eval_episode_info_buf) < 10:
#         # sample actions from policy
#         with torch.no_grad():
#             _, action, _, recurrent_hidden_states = actor_critic.act(obs, 
#                                                                      recurrent_hidden_states,
#                                                                      masks,
#                                                                      deterministic=True)                                                                                     
                                                                                                
#         # observe rewards and next obs
#         obs, _, done, infos = eval_envs.step(action)

#         # create mask for episode ends
#         masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)        

#         # log episode info if finished
#         # need logging on for VecMonitor
#         for info in infos:
#             if 'episode' in info.keys():
#                 eval_episode_info_buf.append(info['episode'])          

#     eval_envs.close()

#     return eval_episode_info_buf                     