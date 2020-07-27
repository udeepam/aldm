"""
Code taken from: https://github.com/microsoft/IBAC-SNI/tree/master/torch_rl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def xavier_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        m.bias.data.zero_()


class Bottleneck(nn.Module):

    def __init__(self, input_size, output_size):
        super(Bottleneck, self).__init__()
        self.output_size = output_size
        # build encoder for mean and std
        self.encode = nn.Linear(input_size, 2 * output_size)

        # intialise weights
        self.weight_init()

        # put model into train mode
        self.train()

    def forward(self, x):
        stats = self.encode(x)
        mu  = stats[:, :self.output_size]
        std = F.softplus(stats[:, self.output_size:]-5)
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        # reparameterisation trick
        z = mu + std*eps
        return z, mu, std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])


# from torch.distributions import Normal, MultivariateNormal
# from torch.distributions.kl import kl_divergence
# from utils.math import kl_divergence as kl_divergence_
# import time
# device = x.device

# start_time = time.time()
# # approach 1
# prior = Normal(torch.zeros(self.output_size).to(device), torch.ones(self.output_size).to(device))
# dist  = Normal(mu, std)
# kld = kl_divergence(dist, prior)
# kld = kld.mean()
# print(kld)
# print(time.time()-start_time)

# # approach 2
# start_time = time.time()
# cov = torch.diag_embed(std.pow(2)).to(device)
# prior = MultivariateNormal(torch.zeros(self.output_size).to(device), torch.eye(self.output_size).to(device))
# dist  = MultivariateNormal(mu, cov)
# kld = kl_divergence(dist, prior)
# kld = kld.mean()
# print(kld)
# print(time.time()-start_time)

# # approach 3
# start_time = time.time()
# kld = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1, dim=1)
# kld = kld.mean()
# print(kld)
# print(time.time()-start_time)

# # approach 4
# start_time = time.time()
# kld = kl_divergence_(dist, prior)
# kld = kld.mean()
# print(kld)
# print(time.time()-start_time)
# print()
