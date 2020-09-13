"""
Based on: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
"""
import os
import argparse
import warnings
import time
import datetime
from collections import deque

import torch

import wandb

from configs import args_ppo, args_ibac, args_ibac_sni, args_dist_match

from models.policy import ACModel
from algorithms.ppo import PPO
from algorithms.storage import RolloutStorage

from utils.make_envs import make_vec_envs, make_rep_analysis_envs
from utils import helpers as utl
from utils import math as utl_math
from utils import evaluation as utl_eval
from utils import test as utl_test
from utils import representation as utl_rep


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# environment variable sets the number of threads to use for parallel regions
# https://github.com/ray-project/ray/issues/6962
# os.environ["OMP_NUM_THREADS"] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='coinrun',
                        help='name of the environment to train on.')
    parser.add_argument('--model', type=str, default='ppo',
                        help='the model to use for training. {ppo, ibac, ibac_sni, dist_match}')
    args, rest_args = parser.parse_known_args()
    env_name = args.env_name
    model = args.model

    # --- ARGUMENTS ---

    if model == 'ppo':
        args = args_ppo.get_args(rest_args)
    elif model == 'ibac':
        args = args_ibac.get_args(rest_args)
    elif model == 'ibac_sni':
        args = args_ibac_sni.get_args(rest_args)
    elif model == 'dist_match':
        args = args_dist_match.get_args(rest_args)
    else:
        raise NotImplementedError

    # place other args back into argparse.Namespace
    args.env_name = env_name
    args.model = model
    args.num_train_envs = args.num_processes-args.num_val_envs if args.num_val_envs>0 else args.num_processes

    # warnings
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, run it with num_processes=1.'
                               'Warning: This will slow things down and might break A2C if '
                               'policy_num_steps < env._max_episode_steps.')

    elif args.num_val_envs>0 and (args.num_val_envs>=args.num_processes or not args.percentage_levels_train<1.0):
        raise ValueError('If --args.num_val_envs>0 then you must also have'
                         '--num_val_envs < --num_processes and  0 < --percentage_levels_train < 1.')

    elif args.num_val_envs>0 and not args.use_dist_matching and args.dist_matching_coef!=0:
        raise ValueError('If --num_val_envs>0 and --use_dist_matching=False then you must also have'
                         '--dist_matching_coef=0.')

    elif args.use_dist_matching and not args.num_val_envs>0:
        raise ValueError('If --use_dist_matching=True then you must also have'
                         '0 < --num_val_envs < --num_processes and 0 < --percentage_levels_train < 1.')

    elif args.analyse_rep and not args.use_bottleneck:
        raise ValueError('If --analyse_rep=True then you must also have'
                         '--use_bottleneck=True.')

    # --- TRAINING ---
    print("Setting up Wandb logging")

    # Weights & Biases logger
    if args.run_name is None:
        # make run name as {env_name}_{TIME}
        now = datetime.datetime.now().strftime('_%d-%m_%H:%M:%S')
        args.run_name = args.env_name+'_'+args.algo+now
    # initialise wandb
    wandb.init(project=args.proj_name,
               name=args.run_name,
               group=args.group_name,
               config=args,
               monitor_gym=False)
    # save wandb dir path
    args.run_dir = wandb.run.dir
    # make directory for saving models
    save_dir = os.path.join(wandb.run.dir, 'models')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set random seed of random, torch and numpy
    utl.set_global_seed(args.seed, args.deterministic_execution)

    # initialise environments for training
    print("Setting up Environments")
    if args.num_val_envs>0:
        train_num_levels = int(args.train_num_levels * args.percentage_levels_train)
        val_start_level  = args.train_start_level + train_num_levels
        val_num_levels   = args.train_num_levels - train_num_levels
        train_envs = make_vec_envs(env_name=args.env_name,
                                   start_level=args.train_start_level,
                                   num_levels=train_num_levels,
                                   distribution_mode=args.distribution_mode,
                                   paint_vel_info=args.paint_vel_info,
                                   num_processes=args.num_train_envs,
                                   num_frame_stack=args.num_frame_stack,
                                   device=device)
        val_envs = make_vec_envs(env_name=args.env_name,
                                 start_level=val_start_level,
                                 num_levels=val_num_levels,
                                 distribution_mode=args.distribution_mode,
                                 paint_vel_info=args.paint_vel_info,
                                 num_processes=args.num_val_envs,
                                 num_frame_stack=args.num_frame_stack,
                                 device=device)
    else:
        train_envs = make_vec_envs(env_name=args.env_name,
                                   start_level=args.train_start_level,
                                   num_levels=args.train_num_levels,
                                   distribution_mode=args.distribution_mode,
                                   paint_vel_info=args.paint_vel_info,
                                   num_processes=args.num_processes,
                                   num_frame_stack=args.num_frame_stack,
                                   device=device)
    # initialise environments for evaluation
    eval_envs = make_vec_envs(env_name=args.env_name,
                              start_level=0,
                              num_levels=0,
                              distribution_mode=args.distribution_mode,
                              paint_vel_info=args.paint_vel_info,
                              num_processes=args.num_processes,
                              num_frame_stack=args.num_frame_stack,
                              device=device)
    _ = eval_envs.reset()
    # initialise environments for analysing the representation
    if args.analyse_rep:
        analyse_rep_train1_envs, analyse_rep_train2_envs, analyse_rep_val_envs, analyse_rep_test_envs = make_rep_analysis_envs(args, device)

    print("Setting up Actor-Critic model and Training algorithm")
    # initialise policy network
    actor_critic = ACModel(obs_shape=train_envs.observation_space.shape,
                           action_space=train_envs.action_space,
                           hidden_size=args.hidden_size,
                           use_bottleneck=args.use_bottleneck,
                           sni_type=args.sni_type).to(device)

    # initialise policy training algorithm
    if args.algo == 'ppo':
        policy = PPO(actor_critic=actor_critic,
                     ppo_epoch=args.policy_ppo_epoch,
                     num_mini_batch=args.policy_num_mini_batch,
                     clip_param=args.policy_clip_param,
                     value_loss_coef=args.policy_value_loss_coef,
                     entropy_coef=args.policy_entropy_coef,
                     max_grad_norm=args.policy_max_grad_norm,
                     lr=args.policy_lr,
                     eps=args.policy_eps,
                     vib_coef=args.vib_coef,
                     sni_coef=args.sni_coef,
                     use_dist_matching=args.use_dist_matching,
                     dist_matching_loss=args.dist_matching_loss,
                     dist_matching_coef=args.dist_matching_coef,
                     num_train_envs=args.num_train_envs,
                     num_val_envs=args.num_val_envs)
    else:
        raise NotImplementedError

    # initialise rollout storage for the policy
    rollouts = RolloutStorage(num_steps=args.policy_num_steps,
                              num_processes=args.num_processes,
                              obs_shape=train_envs.observation_space.shape,
                              action_space=train_envs.action_space)

    # count number of frames and updates
    frames   = 0
    iter_idx = 0

    # Weights & Biases logger
    wandb.config.update(args)
    # wandb.watch(actor_critic, log="all") # to log gradients of actor-critic network

    update_start_time = time.time()

    # reset environments
    if args.num_val_envs>0:
        obs = torch.cat([train_envs.reset(), val_envs.reset()])  # obs.shape = (n_envs,C,H,W)
    else:
        obs = train_envs.reset()  # obs.shape = (n_envs,C,H,W)

    # insert initial observation to rollout storage
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    # initialise buffer for calculating mean episodic returns
    train_episode_info_buf = deque(maxlen=10)
    val_episode_info_buf   = deque(maxlen=10)

    # calculate number of updates
    # number of frames ÷ number of policy steps before update ÷ number of cpu processes
    args.num_batch = args.num_processes * args.policy_num_steps
    args.num_updates = int(args.num_frames) // args.num_batch
    print("Training beginning")
    print("Number of updates: ", args.num_updates)
    for iter_idx in range(args.num_updates):
        print("Iter: ", iter_idx)

        # put actor-critic into train mode
        actor_critic.train()

        # rollout policy to collect num_batch of experience and store in storage
        for step in range(args.policy_num_steps):

            # sample actions from policy
            with torch.no_grad():
                value, action, action_log_prob, _ = actor_critic.act(rollouts.obs[step])

            # observe rewards and next obs
            if args.num_val_envs>0:
                obs, reward, done, infos = train_envs.step(action[:args.num_train_envs, :])
                val_obs, val_reward, val_done, val_infos = val_envs.step(action[args.num_train_envs:, :])
                obs = torch.cat([obs, val_obs])
                reward = torch.cat([reward, val_reward])
                done, val_done = list(done), list(val_done)
                done.extend(val_done)
                infos.extend(val_infos)
            else:
                obs, reward, done, infos = train_envs.step(action)

            # log episode info if episode finished
            for i, info in enumerate(infos):
                if i<args.num_train_envs and 'episode' in info.keys():
                    train_episode_info_buf.append(info['episode'])
                elif i>=args.num_train_envs and 'episode' in info.keys():
                    val_episode_info_buf.append(info['episode'])

            # create mask for episode ends
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)

            # add experience to policy buffer
            rollouts.insert(obs,
                            reward,
                            action,
                            value,
                            action_log_prob,
                            masks)

            frames += args.num_processes

        # --- UPDATE ---

        # bootstrap next value prediction
        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()

        # compute returns for current rollouts
        rollouts.compute_returns(next_value,
                                 args.policy_gamma,
                                 args.policy_gae_lambda)

        # update actor-critic using policy gradient algo
        total_loss, value_loss, action_loss, dist_entropy, vib_kl, dist_matching_loss = policy.update(rollouts)

        # clean up after update
        rollouts.after_update()

        # --- LOGGING ---

        if iter_idx % args.log_interval == 0 or iter_idx == args.num_updates - 1:

            # --- EVALUATION ---
            eval_episode_info_buf = utl_eval.evaluate(eval_envs=eval_envs,
                                                      actor_critic=actor_critic,
                                                      device=device)

            # --- ANALYSE REPRESENTATION ---
            if args.analyse_rep:
                rep_measures = utl_rep.analyse_rep(args=args,
                                                   train1_envs=analyse_rep_train1_envs,
                                                   train2_envs=analyse_rep_train2_envs,
                                                   val_envs=analyse_rep_val_envs,
                                                   test_envs=analyse_rep_test_envs,
                                                   actor_critic=actor_critic,
                                                   device=device)

            # get stats for run
            update_end_time = time.time()
            num_interval_updates = 1 if iter_idx == 0 else args.log_interval
            fps = num_interval_updates * (args.num_processes * args.policy_num_steps) / (update_end_time - update_start_time)
            update_start_time = update_end_time
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = utl_math.explained_variance(utl.sf01(rollouts.value_preds),
                                             utl.sf01(rollouts.returns))

            wandb.log({'misc/timesteps': frames,
                       'misc/fps': fps,
                       'misc/explained_variance': float(ev),
                       'losses/total_loss': total_loss,
                       'losses/value_loss': value_loss,
                       'losses/action_loss': action_loss,
                       'losses/dist_entropy': dist_entropy,
                       'train/mean_episodic_return': utl_math.safe_mean([episode_info['r'] for episode_info in train_episode_info_buf]),
                       'train/mean_episodic_length': utl_math.safe_mean([episode_info['l'] for episode_info in train_episode_info_buf]),
                       'eval/mean_episodic_return': utl_math.safe_mean([episode_info['r'] for episode_info in eval_episode_info_buf]),
                       'eval/mean_episodic_length': utl_math.safe_mean([episode_info['l'] for episode_info in eval_episode_info_buf])}, step=iter_idx)
            if args.use_bottleneck:
                wandb.log({'losses/vib_kl': vib_kl}, step=iter_idx)
            if args.num_val_envs>0:
                wandb.log({'losses/dist_matching_loss': dist_matching_loss,
                           'val/mean_episodic_return': utl_math.safe_mean([episode_info['r'] for episode_info in val_episode_info_buf]),
                           'val/mean_episodic_length': utl_math.safe_mean([episode_info['l'] for episode_info in val_episode_info_buf])}, step=iter_idx)
            if args.analyse_rep:
                wandb.log({"analysis/"+key: val for key, val in rep_measures.items()}, step=iter_idx)

        # --- SAVE MODEL ---

        # save for every interval-th episode or for the last epoch
        if iter_idx !=0 and (iter_idx % args.save_interval == 0 or iter_idx == args.num_updates - 1):
            print("Saving Actor-Critic Model")
            torch.save(actor_critic.state_dict(), os.path.join(save_dir, "policy{0}.pt".format(iter_idx)))

    # close environments
    train_envs.close()
    eval_envs.close()

    # --- TEST ---

    if args.test:
        print("Testing beginning")
        episodic_return, latents_z = utl_test.test(args=args,
                                                   actor_critic=actor_critic,
                                                   device=device)

        # save returns from train and test levels to analyse using interactive mode
        train_levels = torch.arange(args.train_start_level, args.train_start_level+args.train_num_levels)
        for i, level in enumerate(train_levels):
            wandb.log({'test/train_levels': level,
                       'test/train_returns': episodic_return[0][i]})
        test_levels = torch.arange(args.test_start_level, args.test_start_level+args.test_num_levels)
        for i, level in enumerate(test_levels):
            wandb.log({'test/test_levels': level,
                       'test/test_returns': episodic_return[1][i]})
        # log returns from test envs
        wandb.run.summary["train_mean_episodic_return"] = utl_math.safe_mean(episodic_return[0])
        wandb.run.summary["test_mean_episodic_return"]  = utl_math.safe_mean(episodic_return[1])

        # plot latent representation
        if args.plot_pca:
            print("Plotting PCA of Latent Representation")
            utl_rep.pca(args, latents_z)


if __name__ == '__main__':
    main()
