import torch

from utils.make_envs import make_vec_envs


def test(args, actor_critic, device):

    # put actor-critic into evaluation mode
    actor_critic.eval()

    # store episodic returns from train and test envs
    episodic_returns = list()
    # create placeholder for latents
    if args.plot_pca:
        latents_z = {'Train Complete': torch.zeros(1, args.hidden_size).to(device),
                     'Train Fail': torch.zeros(1, args.hidden_size).to(device),
                     'Test Complete': torch.zeros(1, args.hidden_size).to(device),
                     'Test Fail': torch.zeros(1, args.hidden_size).to(device)}

    # iterate through train and then test envs
    for i in range(2):
        # counters
        num_complete_levels = num_fail_levels = 0

        # get level ranges
        start_level = args.train_start_level if i==0 else args.test_start_level
        num_levels  = args.train_num_levels if i==0 else args.test_num_levels

        # initialise buffer for storing returns from episodes
        episode_returns = list()

        # iterate through levels sequentially
        for level in range(start_level, start_level+num_levels):

            # initialise single env for train or test
            env = make_vec_envs(env_name=args.env_name,
                                start_level=level,
                                num_levels=1,
                                distribution_mode=args.distribution_mode,
                                paint_vel_info=args.paint_vel_info,
                                num_processes=1,
                                num_frame_stack=args.num_frame_stack,
                                device=device)

            # reset env
            obs = env.reset()
            obs = obs.to(device)

            # create placeholder for latents for current episode
            if args.plot_pca:
                latents = torch.zeros(1, args.hidden_size).to(device)

            # take steps in env until episode terminates
            while True:
                # sample actions from policy
                with torch.no_grad():
                    _, action, _, z = actor_critic.act(obs)

                # add latents to placeholder
                if args.plot_pca and num_complete_levels<args.num_levels2plot and num_fail_levels<args.num_levels2plot:
                    latents = torch.cat([latents, z], dim=0)

                # observe rewards and next obs
                obs, _, done, infos = env.step(action)

                if done[0]:
                    # log episode info if finished
                    episode_returns.append(infos[0]['episode']['r'])

                    level_outcome = infos[0]['prev_level_complete']
                    # save train latents
                    if args.plot_pca and i==0:
                        if level_outcome==1 and num_complete_levels<args.num_levels2plot:
                            latents_z['Train Complete'] = torch.cat([latents_z['Train Complete'], latents[1:, :]], dim=0)
                            num_complete_levels += 1
                        elif level_outcome==0 and num_fail_levels<args.num_levels2plot:
                            latents_z['Train Fail'] = torch.cat([latents_z['Train Fail'], latents[1:, :]], dim=0)
                            num_fail_levels += 1
                    # save test latents
                    elif args.plot_pca and i==1:
                        if level_outcome==1 and num_complete_levels<args.num_levels2plot:
                            latents_z['Test Complete'] = torch.cat([latents_z['Test Complete'], latents[1:, :]], dim=0)
                            num_complete_levels += 1
                        elif level_outcome==0 and num_fail_levels<args.num_levels2plot:
                            latents_z['Test Fail'] = torch.cat([latents_z['Test Fail'], latents[1:, :]], dim=0)
                            num_fail_levels += 1

                    env.close()
                    break

        # store episodic returns from train and test
        episodic_returns.append(episode_returns)

    # organise latents
    latents_z = {key: val[1:] for key, val in latents_z.items() if len(val)>1} if args.plot_pca else None

    return episodic_returns, latents_z
