"""
Based on: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
"""
import torch

from utils.make_envs import make_vec_envs


def evaluate(args,
             actor_critic,
             device):

    # put actor-critic into evaluation mode
    actor_critic.eval()

    # initialise environments for evaluation
    eval_envs = make_vec_envs(env_name=args.env_name,
                              start_level=0,
                              num_levels=0,
                              distribution_mode=args.distribution_mode,
                              paint_vel_info=args.paint_vel_info,
                              num_processes=1,
                              log_dir=None,
                              device=device,
                              num_frame_stack=args.num_frame_stack)

    # initialise buffer for calculating means
    eval_episode_info_buf = list()

    # reset environments
    obs = eval_envs.reset()  # obs.shape = (n_env,C,H,W)
    obs = obs.to(device)

    # collect returns from 10 full episodes
    while len(eval_episode_info_buf) < 10:
        # sample actions from policy
        with torch.no_grad():
            # determinism can lead to subpar performance
            _, action, _ = actor_critic.act(obs)

        # observe rewards and next obs
        obs, _, done, infos = eval_envs.step(action)

        # log episode info if finished
        # need logging on for VecMonitor
        for info in infos:
            if 'episode' in info.keys():
                eval_episode_info_buf.append(info['episode'])

    eval_envs.close()

    return eval_episode_info_buf
