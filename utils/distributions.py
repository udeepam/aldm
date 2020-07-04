"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
         https://github.com/harry-uglow/Curriculum-Reinforcement-Learning
         https://github.com/lmzintgraf/varibad
"""
import torch


"""
Modify standard PyTorch distributions so they are compatible with this code.
- modifies action shape
- This way also allows wandb logging for the model.
"""

#
# Standardize distribution interfaces
#

# ---Categorical---


class FixedCategorical(torch.distributions.Categorical):

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return super().log_prob(actions.squeeze(-1)).unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)