from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence

from matplotlib import pyplot as plt
import plotly.graph_objects as go

import wandb

from sklearn.decomposition import PCA

from utils import math as utl_math
from utils.helpers import reset_envs


def analyse_rep(args,
                train1_envs,
                train2_envs,
                val_envs,
                test_envs,
                actor_critic,
                device):
    """
    Analyse the latent representation using KL divergence.

    TODO: important to note the mini_batch_size for this as it means how we calculate the divergences.
          atm think its too small
    """
    # put actor-critic into evaluation mode
    actor_critic.eval()

    # --- GATHER TRAIN DATA ----

    num_train1_envs = train1_envs.num_envs
    # create list to store test env observations
    train_obs = torch.zeros(args.policy_num_steps, args.num_processes, *train1_envs.observation_space.shape).to(device)
    # reset environments
    if args.num_val_envs>0:
        obs = torch.cat([reset_envs(train1_envs, device), reset_envs(train2_envs, device), reset_envs(val_envs, device)])  # obs.shape = (n_envs,C,H,W)
    else:
        obs = torch.cat([reset_envs(train1_envs, device), reset_envs(train2_envs, device)])  # obs.shape = (n_envs,C,H,W)
    obs = obs.to(device)
    # rollout policy to collect experience
    for step in range(args.policy_num_steps):
        # sample actions from policy
        with torch.no_grad():
            _, action, _, _ = actor_critic.act(obs)
        # observe rewards and next obs
        if args.num_val_envs>0:
            obs, _, _, _ = train1_envs.step(action[:num_train1_envs, :])
            train2_obs, _, _, _ = train2_envs.step(action[num_train1_envs:args.num_train_envs, :])
            val_obs, _, _, _ = val_envs.step(action[args.num_train_envs:, :])
            obs = torch.cat([obs, train2_obs, val_obs])
        else:
            obs, _, _, _ = train1_envs.step(action[:num_train1_envs, :])
            train2_obs, _, _, _ = train2_envs.step(action[num_train1_envs:, :])
            obs = torch.cat([obs, train2_obs])
        # store obs
        train_obs[step].copy_(obs)

    # --- GATHER TEST DATA ----

    # create list to store test env observations
    test_obs = torch.zeros(args.policy_num_steps, args.num_train_envs, *test_envs.observation_space.shape).to(device)
    # reset environments
    obs = reset_envs(test_envs, device)  # obs.shape = (n_env,C,H,W)
    obs = obs.to(device)
    # rollout policy to collect experience
    for step in range(args.policy_num_steps):
        # sample actions from policy
        with torch.no_grad():
            _, action, _, _ = actor_critic.act(obs)
        # observe rewards and next obs
        obs, _, _, _ = test_envs.step(action)
        # store obs
        test_obs[step].copy_(obs)

    # --- GET LATENT REPRESENTATION ---

    train1_obs = train_obs[:, :num_train1_envs].reshape(-1, *train_obs[:, :num_train1_envs].size()[2:])
    train2_obs = train_obs[:, num_train1_envs:args.num_train_envs].reshape(-1, *train_obs[:, num_train1_envs:args.num_train_envs].size()[2:])
    test_obs = test_obs.reshape(-1, *test_obs.size()[2:])
    # create train indices sampler
    train_batch_size = num_train1_envs * args.policy_num_steps
    train_mini_batch_size = train_batch_size // args.policy_num_mini_batch
    train_sampler = BatchSampler(SequentialSampler(range(train_batch_size)),
                                 train_mini_batch_size,
                                 drop_last=True)
    # create test indices sampler
    test_batch_size = args.num_train_envs * args.policy_num_steps
    test_mini_batch_size = test_batch_size // args.policy_num_mini_batch
    test_sampler = BatchSampler(SequentialSampler(range(test_batch_size)),
                                test_mini_batch_size,
                                drop_last=True)
    # get validation
    if args.num_val_envs>0:
        val_obs = train_obs[:, args.num_train_envs:].reshape(-1, *train_obs[:, args.num_train_envs:].size()[2:])
        val_batch_size = (args.num_processes - args.num_train_envs) * args.policy_num_steps
        val_mini_batch_size = val_batch_size // args.policy_num_mini_batch
        val_sampler = BatchSampler(SequentialSampler(range(val_batch_size)),
                                   val_mini_batch_size,
                                   drop_last=True)
    else:
        val_sampler = torch.zeros(int(train_batch_size/train_mini_batch_size)+1)

    # initialise values
    measures = defaultdict(list)
    for i, (train_indices, val_indices, test_indices) in enumerate(zip(train_sampler, val_sampler, test_sampler)):
        with torch.no_grad():
            # encode train and test observations
            train1_action, train1_z, train1_mu, train1_std = actor_critic.get_analysis(train1_obs[train_indices])
            train2_action, train2_z, train2_mu, train2_std = actor_critic.get_analysis(train2_obs[train_indices])
            test_action, test_z, test_mu, test_std = actor_critic.get_analysis(test_obs[test_indices])
            # create full train
            train_z = torch.cat([train1_z, train2_z])
            train_mu = torch.cat([train1_mu, train2_mu])
            train_std = torch.cat([train1_std, train2_std])
            train_action = torch.cat([train1_action, train2_action])
            # encode val observations
            if args.num_val_envs>0:
                val_action, val_z, val_mu, val_std = actor_critic.get_analysis(val_obs[val_indices])

        # --- ANALYSE LATENT REPRESENTATION ---

        # calculate KL between train and prior
        measures['train_prior_kl'].append(0.5 * torch.sum(train_mu.pow(2) + train_std.pow(2) - 2*train_std.log() - 1, dim=1).mean())
        # calculate KL between test and prior
        measures['test_prior_kl'].append(0.5 * torch.sum(test_mu.pow(2) + test_std.pow(2) - 2*test_std.log() - 1, dim=1).mean())
        # calculate KL between first half of train and second half of train
        train1_dist = MultivariateNormal(train1_z.mean(dim=0), scale_tril=torch.diag(train1_z.std(dim=0)))
        train2_dist = MultivariateNormal(train2_z.mean(dim=0), scale_tril=torch.diag(train2_z.std(dim=0)))
        measures['inter_train_kl'].append(kl_divergence(train1_dist, train2_dist).mean())
        # calculate KL between train and test
        train_dist = MultivariateNormal(train_z.mean(dim=0), scale_tril=torch.diag(train_z.std(dim=0)))
        test_dist  = MultivariateNormal(test_z.mean(dim=0), scale_tril=torch.diag(test_z.std(dim=0)))
        measures['train_test_kl'].append(kl_divergence(train_dist, test_dist).mean())
        # get KL between train and test for each action
        for j in range(test_envs.action_space.n):
            train_z_action = train_z[(train_action==j).squeeze(), :]
            test_z_action  = test_z[(test_action==j).squeeze(), :]
            if len(train_z_action)<2 or len(test_z_action)<2:
                measures['train_test_'+str(j)+"_kl"]
            else:
                train_dist = MultivariateNormal(train_z_action.mean(dim=0), scale_tril=torch.diag(train_z_action.std(dim=0)))
                test_dist  = MultivariateNormal(test_z_action.mean(dim=0), scale_tril=torch.diag(test_z_action.std(dim=0)))
                measures['train_test_'+str(j)+"_kl"].append(kl_divergence(train_dist, test_dist).mean())
        if args.num_val_envs>0:
            # calculate KL between val and prior
            measures['val_prior_kl'].append(0.5 * torch.sum(val_mu.pow(2) + val_std.pow(2) - 2*val_std.log() - 1, dim=1).mean())
            # calculate KL between train and val
            val_dist  = MultivariateNormal(val_z.mean(dim=0), scale_tril=torch.diag(val_z.std(dim=0)))
            measures['train_val_kl'].append(kl_divergence(train_dist, val_dist).mean())
            # get KL between train and val for each action
            for j in range(test_envs.action_space.n):
                train_z_action = train_z[(train_action==j).squeeze(), :]
                val_z_action  = val_z[(val_action==j).squeeze(), :]
                if len(train_z_action)<2 or len(val_z_action)<2:
                    measures['train_val_'+str(j)+"_kl"]
                else:
                    train_dist = MultivariateNormal(train_z_action.mean(dim=0), scale_tril=torch.diag(train_z_action.std(dim=0)))
                    test_dist  = MultivariateNormal(val_z_action.mean(dim=0), scale_tril=torch.diag(val_z_action.std(dim=0)))
                    measures['train_val_'+str(j)+"_kl"].append(kl_divergence(train_dist, test_dist).mean())
    # calculate mean of measures
    new_measures = dict()
    for key, val in measures.items():
        if len(val)!=0:
            new_measures[key] = utl_math.safe_torch_mean(torch.stack(val))
        else:
            new_measures[key] = np.nan

    return new_measures


def pca(args, latents_z):
    """
    latents_z : `dict` of `torch.Tensor`
    """
    # concatenate latents from all sources and bring to cpu
    z = torch.cat(list(latents_z.values())).cpu()

    # project down latent space down to 2D
    embeddings = PCA(n_components=2).fit_transform(z)

    # replace latents with the pca embeddings
    start = 0
    pca_z = dict()
    for key, val in latents_z.items():
        pca_z[key] = embeddings[start:start+val.shape[0]]
        start += val.shape[0]

    # plotly
    fig = go.Figure()
    for key, val in pca_z.items():
        fig.add_trace(go.Scatter(x=val[:, 0],
                                 y=val[:, 1],
                                 mode='markers',
                                 name=key))
    fig.update_layout(title={'text': "PCA plot of Train and Test Latent Representation",
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'})

    # matplotlib
    color = ['b', 'r'] if len(latents_z)==2 else ['b', 'orange', 'r'] if len(latents_z)==3 else ['b', 'y', 'g', 'r']
    plt.rcParams.update({'font.size': 13})
    plt.figure(figsize=(10, 10))
    for i, (key, val) in enumerate(pca_z.items()):
        plt.scatter(val[:, 0], val[:, 1], color=color[i], marker='.', label=key)
    plt.title("PCA plot of Train and Test Latent Representation")
    plt.legend()
    plt.grid()

    # log plots to wandb
    wandb.log({"pca_plotly": fig,
               "pca_plot": wandb.Image(plt)})
