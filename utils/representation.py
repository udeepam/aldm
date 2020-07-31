import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from matplotlib import pyplot as plt
import plotly.graph_objects as go

import wandb

from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA

from utils.make_envs import make_vec_envs


def analyse_representation(args, actor_critic, rollouts, num_updates, device, use_pca=True, use_t_sne=True):
    """
    Methods for analysing the representation:
    1. t-SNE or PCA
    2. JSD
    3. Cycle consistency
    """
    # --- GATHER TEST DATA ----

    # put actor-critic into evaluation mode
    actor_critic.eval()
    # initialise environments for gathering data for analysing representation
    test_envs = make_vec_envs(env_name=args.env_name,
                              start_level=args.test_start_level,
                              num_levels=args.test_num_levels,
                              distribution_mode=args.distribution_mode,
                              paint_vel_info=args.paint_vel_info,
                              num_processes=args.num_processes,
                              log_dir=None,
                              device=device,
                              num_frame_stack=args.num_frame_stack)
    # create list to store test env observations
    test_obs = torch.zeros(args.policy_num_steps, args.num_processes, *test_envs.observation_space.shape).to(device)
    # reset environments
    obs = test_envs.reset()  # obs.shape = (n_env,C,H,W)
    obs = obs.to(device)
    # collect returns from 10 full episodes
    for step in range(args.policy_num_steps):
        # sample actions from policy
        with torch.no_grad():
            # determinism can lead to subpar performance
            _, action, _ = actor_critic.act(obs)
        # observe rewards and next obs
        obs, _, _, _ = test_envs.step(action)
        # store obs
        test_obs[step].copy_(obs)
    test_envs.close()

    # --- GET LATENT REPRESENTATION ---

    # use most recent experience in storage
    train_obs = rollouts.obs[:-1].reshape(-1, *rollouts.obs.size()[2:])
    test_obs = test_obs.reshape(-1, *test_obs.size()[2:])
    # create train indices sampler
    sampler = BatchSampler(SequentialSampler(range(train_obs.shape[0])),
                           256,
                           drop_last=False)
    z = torch.zeros(2*args.num_processes*args.policy_num_steps, args.hidden_size)
    for i, indices in enumerate(sampler):
        # encode observations
        with torch.no_grad():
            train_z, _, _ = actor_critic.encode(train_obs[indices])
            test_z, _, _ = actor_critic.encode(test_obs[indices])
        z[i*256:(i+1)*256].copy_(train_z)
        z[args.num_processes*args.policy_num_steps+i*256:args.num_processes*args.policy_num_steps+(i+1)*256].copy_(test_z)

    # --- ANALYSE LATENT REPRESENTATION ----

    # pca
    if use_pca:
        pca(args, num_updates, z)
    # t-sne
    if use_t_sne:
        t_sne(args, num_updates, z)


def pca(args, num_updates, z):
    # project down latent space down to 2D
    embeddings = PCA(n_components=2).fit_transform(z)

    # plot embeddings
    train_obs = args.num_train_envs * args.policy_num_steps
    test_obs = args.num_processes * args.policy_num_steps

    # plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=embeddings[:train_obs, 0],
                             y=embeddings[:train_obs, 1],
                             mode='markers',
                             name='Train'))
    if args.use_distribution_matching:
        fig.add_trace(go.Scatter(x=embeddings[train_obs:test_obs, 0],
                                 y=embeddings[train_obs:test_obs, 1],
                                 mode='markers',
                                 name='Validation'))
    fig.add_trace(go.Scatter(x=embeddings[test_obs:, 0],
                             y=embeddings[test_obs:, 1],
                             mode='markers',
                             name='Test'))
    fig.update_layout(title={'text': "PCA plot of Latent Representation after "+str(num_updates)+" updates",
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'})

    # matplotlib
    plt.rcParams.update({'font.size': 13})
    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings[:train_obs, 0], embeddings[:train_obs, 1], color='b', marker='.', label='Train')
    if args.use_distribution_matching:
        plt.scatter(embeddings[train_obs:test_obs, 0], embeddings[train_obs:test_obs, 1], color='orange', marker='.', label='Validation')
    plt.scatter(embeddings[test_obs:, 0], embeddings[test_obs:, 1], color='r', marker='.', label='Test')
    plt.title("PCA plot of Latent Representation after "+str(num_updates)+" updates")
    plt.legend()
    plt.grid()

    # save plots
    wandb.log({"pca_plotly_"+str(num_updates): fig, "pca_plot_"+str(num_updates): wandb.Image(plt)})


def t_sne(args, num_updates, z):
    # project down latent space down to 2D
    embeddings = TSNE(n_jobs=args.num_processes).fit_transform(z)

    # plot embeddings
    train_obs = args.num_train_envs * args.policy_num_steps
    test_obs = args.num_processes * args.policy_num_steps

    # plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=embeddings[:train_obs, 0],
                             y=embeddings[:train_obs, 1],
                             mode='markers',
                             name='Train'))
    if args.use_distribution_matching:
        fig.add_trace(go.Scatter(x=embeddings[train_obs:test_obs, 0],
                                 y=embeddings[train_obs:test_obs, 1],
                                 mode='markers',
                                 name='Validation'))
    fig.add_trace(go.Scatter(x=embeddings[test_obs:, 0],
                             y=embeddings[test_obs:, 1],
                             mode='markers',
                             name='Test'))
    fig.update_layout(title={'text': "t-SNE plot of Latent Representation after "+str(num_updates)+" updates",
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'})

    # matplotlib
    plt.rcParams.update({'font.size': 13})
    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings[:train_obs, 0], embeddings[:train_obs, 1], color='b', marker='.', label='Train')
    if args.use_distribution_matching:
        plt.scatter(embeddings[train_obs:test_obs, 0], embeddings[train_obs:test_obs, 1], color='orange', marker='.', label='Validation')
    plt.scatter(embeddings[test_obs:, 0], embeddings[test_obs:, 1], color='r', marker='.', label='Test')
    plt.title("t-SNE plot of Latent Representation after "+str(num_updates)+" updates")
    plt.legend()
    plt.grid()

    # save plots
    wandb.log({"t_SNE_plotly_"+str(num_updates): fig, "t_sne_plot_"+str(num_updates): wandb.Image(plt)})
