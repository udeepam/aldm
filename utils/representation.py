import torch
from matplotlib import pyplot as plt
import plotly.graph_objects as go

import wandb

from MulticoreTSNE import MulticoreTSNE as TSNE


def analyse_representation(args, actor_critic, rollouts):
    """
    Methods for analysing the representation:
    1. t-SNE or PCA
    2. JSD
    3. Cycle consistency
    """
    # t-sne
    t_sne(args, actor_critic, rollouts)


def t_sne(args, actor_critic, rollouts):
    # use most recent experience in storage
    obs_batch = rollouts.obs[:-1].reshape(-1, *rollouts.obs.size()[2:])
    recurrent_hidden_states_batch = rollouts.recurrent_hidden_states[:-1].reshape(-1, rollouts.recurrent_hidden_states.size(-1))
    masks_batch = rollouts.masks[:-1].reshape(-1, 1)

    # encode observations
    with torch.no_grad():
        _, _, z, _, _ = actor_critic.encode(obs_batch,
                                            recurrent_hidden_states_batch,
                                            masks_batch)

    # project down latent space down to 2D
    embeddings = TSNE(n_jobs=args.num_processes).fit_transform(z.cpu())

    # plot embeddings
    train_obs = args.num_train_envs * args.policy_num_steps

    # plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=embeddings[:train_obs, 0],
                             y=embeddings[:train_obs, 1],
                             mode='markers',
                             name='Train'))
    if args.use_distribution_matching:
        fig.add_trace(go.Scatter(x=embeddings[train_obs:, 0],
                                 y=embeddings[train_obs:, 1],
                                 mode='markers',
                                 name='Validation'))
    fig.update_layout(title={'text': "t-SNE plot of Latent Representation",
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'})

    # matplotlib
    plt.rcParams.update({'font.size': 13})
    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings[:train_obs, 0], embeddings[:train_obs, 1], color='b', marker='.', label='Train')
    if args.use_distribution_matching:
        plt.scatter(embeddings[train_obs:, 0], embeddings[train_obs:, 1], color='orange', marker='.', label='Validation')
    plt.title("t-SNE plot of Latent Representation")
    plt.legend()
    plt.grid()

    if args.log:
        # save plots
        wandb.log({"t_SNE_plotly": fig, "t_sne_plot": wandb.Image(plt)})
