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
                                       seed=self.args.seed,
                                       start_level=self.args.eval_start_level, 
                                       num_levels=self.args.eval_num_levels, 
                                       distribution_mode=self.args.distribution_mode,                                  
                                       num_processes=self.args.num_processes, 
                                       gamma=self.args.policy_gamma, 
                                       log_dir=self.eval_log_dir,
                                       device=device, 
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
        self.eval_epinfobuf = deque(maxlen=100)


    def evaluate(self,
                 actor_critic,
                 summary_stats):

        # lists to hold episode info 
        eval_epinfos = list()
        # rollout policy to collect num_batch of experience and store in storage 
        for step in range(self.args.policy_num_steps):
            # sample actions from policy
            with torch.no_grad():
                _, action, _, self.recurrent_hidden_states = actor_critic.act(self.obs.contiguous(), 
                                                                              self.recurrent_hidden_states,
                                                                              self.masks,
                                                                              deterministic=True)                                                                                     
                                                                                                    
            # observe rewards and next obs
            self.obs, _, done, infos = self.eval_envs.step(action)

            # log episode info if finished
            # need logging on for VecMonitor
            for info in infos:
                maybe_epinfo = info.get('episode')
                if maybe_epinfo:
                    eval_epinfos.append(maybe_epinfo)
                    summary_stats['eval_total_eps'] += 1
                    summary_stats['eval_total_eprews']  += maybe_epinfo['r']
                    summary_stats['eval_total_eplens']  += maybe_epinfo['l']
                    summary_stats['eval_total_eptimes'] += maybe_epinfo['t']              

            # create mask for episode ends
            self.masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(self.device)

        # log episode info for current batch    
        self.eval_epinfobuf.extend(eval_epinfos)

        return self.eval_epinfobuf, summary_stats