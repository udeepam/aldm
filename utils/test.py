import torch

from utils.make_envs import make_vec_envs
from utils import math as utl_math


def test(args, obs_shape, actor_critic, device):

    # put actor-critic into evaluation mode
    actor_critic.eval()

    # store episodic returns from train and test envs
    episodic_returns = list()

    start_levels = [args.train_start_level, args.test_start_level]
    num_levels   = [args.train_num_levels, args.test_num_levels]

    # iterate through train and then test envs
    for i in range(2):
        start_level = start_levels[i]
        nlevels = num_levels[i]

        # initialise buffer for storing returns from episodes
        episode_returns = list()

        # iterate through levels sequentially
        for level in range(start_level, start_level+nlevels):

            # initialise single env
            env = make_vec_envs(env_name=args.env_name,
                                start_level=level,
                                num_levels=1,
                                distribution_mode=args.distribution_mode,
                                paint_vel_info=args.paint_vel_info,
                                num_processes=1,
                                log_dir=None,
                                device=device,
                                num_frame_stack=args.num_frame_stack)

            # reset env
            obs = env.reset()
            obs = obs.to(device)

            # take steps in env until episode terminates
            while True:
                # sample actions from policy
                with torch.no_grad():
                    _, action, _ = actor_critic.act(obs)

                # observe rewards and next obs
                obs, _, done, infos = env.step(action)

                if done[0]:
                    # log episode info if finished, need logging on for VecMonitor
                    episode_returns.append(infos[0]['episode']['r'])
                    env.close()
                    break

        # store episodic returns from train and test
        episodic_returns.append(episode_returns)

    return episodic_returns
