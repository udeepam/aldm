"""
Based on: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
"""
import os
import time
import datetime
from collections import deque

import numpy as np
import torch

import wandb

from utils.make_envs import make_vec_envs
from utils import helpers as utl
from utils import math as utl_math
from utils import evaluation as utl_eval

from models.policy import Policy
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
                # make run name as env_name_current_time
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

            # if self.args.eval_interval:
            #     # make directory for logging eval envs
            #     self.eval_log_dir = os.path.join(wandb.run.dir, 'eval')   
            #     if not os.path.exists(self.eval_log_dir):
            #         os.makedirs(self.eval_log_dir)

        else:
            self.train_log_dir = None                                         

        # set random seed of random, torch and numpy
        utl.set_global_seed(self.args.seed, self.args.deterministic_execution)

        # initialise environments for training
        self.envs = make_vec_envs(env_name=self.args.env_name, 
                                  start_level=self.args.train_start_level, 
                                  num_levels=self.args.train_num_levels, 
                                  distribution_mode=self.args.distribution_mode,   
                                  paint_vel_info=self.args.paint_vel_info,
                                  seed=self.args.seed,                                                                 
                                  num_processes=self.args.num_processes, 
                                  log_dir=self.train_log_dir,
                                  device=device, 
                                  recurrent=self.args.recurrent_policy,
                                  num_frame_stack=self.args.num_frame_stack)                       

        # initialise policy network
        self.initialise_policy()

        # initialise evaluation loop
        if self.args.eval_interval is not None:
            self.evaluator = utl_eval.Evaluator(self.args, 
                                                self.policy.actor_critic.recurrent_hidden_state_size,
                                                device)        

        # count number of frames and updates
        self.frames   = 0
        self.iter_idx = 0        
        
    
    def initialise_policy(self):
        # initialise policy network
        actor_critic = Policy(self.envs.observation_space.shape,
                              self.envs.action_space,
                              base=self.args.env_name,
                              base_kwargs={'recurrent': self.args.recurrent_policy, 
                                           'hidden_size': self.args.hidden_size}).to(device)
        
        # initialise policy trainer
        if self.args.algo == 'ppo':
            self.policy = PPO(actor_critic=actor_critic,
                              clip_param=self.args.policy_clip_param,
                              ppo_epoch=self.args.policy_ppo_epoch,
                              num_mini_batch=self.args.policy_num_mini_batch,
                              value_loss_coef=self.args.policy_value_loss_coef,
                              entropy_coef=self.args.policy_entropy_coef,
                              lr=self.args.policy_lr,
                              eps=self.args.policy_eps,
                              max_grad_norm=self.args.policy_max_grad_norm)
        else:
            raise NotImplementedError
            
        # initialise rollout storage for the policy
        self.rollouts = RolloutStorage(self.args.policy_num_steps, 
                                       self.args.num_processes,
                                       self.envs.observation_space.shape, 
                                       self.envs.action_space,
                                       self.policy.actor_critic.recurrent_hidden_state_size)        
        
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
        episode_info_buf = deque(maxlen=100)

        # calculate number of updates
        # number of frames ÷ number of policy steps before update ÷ number of cpu processes
        self.args.num_batch = self.args.num_processes  * self.args.policy_num_steps
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
                                                                                                           self.rollouts.masks[step])                                                                                     
                                                                                                        
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
                
                self.frames += self.args.num_processes   

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
            
            # update actor-critic
            value_loss, action_loss, dist_entropy = self.policy.update(self.rollouts)

            # clean up after update
            self.rollouts.after_update()

            # --- EVALUATE POLICY ----

            if self.args.eval_interval is not None and self.iter_idx % self.args.eval_interval == 0:
                eval_episode_info_buf = self.evaluator.evaluate(actor_critic=self.policy.actor_critic)   
                # eval_episode_info_buf = utl_eval.evaluate(args=self.args, 
                #                                           step=self.iter_idx, 
                #                                           actor_critic=self.policy.actor_critic,
                #                                           eval_log_dir=self.eval_log_dir,
                #                                           device=device)          

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
                           'losses/val_loss': value_loss,
                           'losses/action_loss': action_loss,
                           'losses/dist_entropy': dist_entropy}, step=self.iter_idx)
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
            test_dir = os.path.join(wandb.run.dir, 'test')   
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)  
            test_train_dir = test_dir+'/train'
            test_test_dir  = test_dir+'/test'
        else:
            test_train_dir = None
            test_test_dir  = None           

        # put actor-critic into evaluation mode 
        self.policy.actor_critic.eval()

        # store mean episdic returns from train and test envs
        mean_episodic_return = list()

        start_levels = [self.args.train_start_level, self.args.test_start_level]
        num_levels   = [self.args.train_num_levels, self.args.test_num_levels]  
        log_dirs     = [test_train_dir, test_test_dir]

        # iterate through train and then test envs
        for i in range(2):
            start_level = start_levels[i]
            nlevels = num_levels[i]
            log_dir = log_dirs[i]

            # initialise buffer for calculating means
            episode_returns = list()      

            # iterate through levels sequentially
            for level in range(start_level, start_level+nlevels):

                # initialise single env
                env = make_vec_envs(env_name=self.args.env_name, 
                                    start_level=level, 
                                    num_levels=1, 
                                    distribution_mode=self.args.distribution_mode, 
                                    paint_vel_info=self.args.paint_vel_info,
                                    seed=self.args.seed,                                                                        
                                    num_processes=1, 
                                    log_dir=log_dir,
                                    device=device, 
                                    recurrent=self.args.recurrent_policy,
                                    num_frame_stack=self.args.num_frame_stack)  

                # reset env
                obs = env.reset()
                obs = obs.to(device)

                # initialisations 
                recurrent_hidden_states = torch.zeros(1, 
                                                      self.policy.actor_critic.recurrent_hidden_state_size, 
                                                      device=device)
                masks = torch.zeros(1, 1, device=device)                        

                # take steps in env until episode terminates
                while True:
                    # sample actions from policy
                    with torch.no_grad():
                        _, action, _, recurrent_hidden_states = self.policy.actor_critic.act(obs, 
                                                                                             recurrent_hidden_states,
                                                                                             masks,
                                                                                             deterministic=True)                                                                                     
                                                                                                            
                    # observe rewards and next obs
                    obs, _, done, infos = env.step(action)

                    # create mask for episode ends
                    masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)                      

                    if done[0]:
                        if self.args.log:
                            # log episode info if finished, need logging on for VecMonitor
                            episode_returns.append(infos[0]['episode']['r']) 
                            env.close()                       
                        break  
            
            # compute mean episodic return
            mean_episodic_return.append(np.mean(episode_returns))

        wandb.run.summary["train_mean_episodic_return"] = mean_episodic_return[0]
        wandb.run.summary["test_mean_episodic_return"]  = mean_episodic_return[1]
