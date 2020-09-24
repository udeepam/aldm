"""
Based on: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
"""
import torch

from utils.helpers import reset_envs


def evaluate(eval_envs,
             actor_critic,
             device):

    # put actor-critic into evaluation mode
    actor_critic.eval()

    # initialise buffer for calculating mean episodic returns
    eval_episode_info_buf = list()

    # reset environments
    obs = reset_envs(eval_envs, device)  # obs.shape = (n_env,C,H,W)
    obs = obs.to(device)

    # collect returns from 10 full episodes
    while len(eval_episode_info_buf) < 10:
        # sample actions from policy
        with torch.no_grad():
            _, action, _, _ = actor_critic.act(obs)

        # observe rewards and next obs
        obs, _, _, infos = eval_envs.step(action)

        # log episode info if finished
        for info in infos:
            if 'episode' in info.keys():
                eval_episode_info_buf.append(info['episode'])

    return eval_episode_info_buf
