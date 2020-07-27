import torch

from utils.make_envs import make_vec_envs
from utils import math as utl_math


def test(args,
         actor_critic,
         test_log_dir,
         device):

    # Weights & Biases logger
    if args.log:
        # make directory for logging test envs
        test_train_dir = test_log_dir+'/train'
        test_test_dir  = test_log_dir+'/test'
    else:
        test_train_dir = None
        test_test_dir  = None

    # put actor-critic into evaluation mode
    actor_critic.eval()

    # store mean episdic returns from train and test envs
    mean_episodic_return = list()

    start_levels = [args.train_start_level, args.test_start_level]
    num_levels   = [args.train_num_levels, args.test_num_levels]
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
            env = make_vec_envs(env_name=args.env_name,
                                start_level=level,
                                num_levels=1,
                                distribution_mode=args.distribution_mode,
                                paint_vel_info=args.paint_vel_info,
                                num_processes=1,
                                log_dir=log_dir,
                                device=device,
                                num_frame_stack=args.num_frame_stack)

            # reset env
            obs = env.reset()
            obs = obs.to(device)

            # initialisations
            recurrent_hidden_states = torch.zeros(1,
                                                  actor_critic.recurrent_hidden_state_size,
                                                  device=device)
            masks = torch.zeros(1, 1, device=device)

            # take steps in env until episode terminates
            while True:
                # sample actions from policy
                with torch.no_grad():
                    _, action, _, recurrent_hidden_states = actor_critic.act(obs,
                                                                             recurrent_hidden_states,
                                                                             masks)

                # observe rewards and next obs
                obs, _, done, infos = env.step(action)

                # create mask for episode ends
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)

                if done[0]:
                    if args.log:
                        # log episode info if finished, need logging on for VecMonitor
                        episode_returns.append(infos[0]['episode']['r'])
                        env.close()
                    break

        # compute mean episodic return
        mean_episodic_return.append(utl_math.safe_mean(episode_returns))

    return mean_episodic_return
