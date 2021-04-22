import torch
from torch import nn


class Network(torch.nn.Module):

    def __init__(self, input_dim=2, n_actions=4, prob_uniform=0.01, track_mean=True):
        super().__init__()

        self.p_network = nn.Sequential(
            *[nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, n_actions)])
        self.v_network = nn.Sequential(
            *[nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1)])
        self.prob_uniform = prob_uniform
        self.track_mean = track_mean

        self.register_buffer('running_mean', torch.zeros(input_dim, dtype=torch.float32), persistent=True)
        self.register_buffer('running_sq_mean', torch.ones(input_dim, dtype=torch.float32), persistent=True)
        self.running_n = 0

    def forward(self, x):

        if self.track_mean:
            if not self.training:
                add_n = x.shape[0]
                add_mean = x.mean(dim=0)
                add_sq_mean = (x ** 2).mean(dim=0)
                new_n = self.running_n + add_n
                self.running_mean = self.running_mean + add_n / new_n * (add_mean - self.running_mean)
                self.running_sq_mean = self.running_sq_mean + add_n / new_n * (add_sq_mean - self.running_sq_mean)
                self.running_n = new_n
            std = (self.running_sq_mean - self.running_mean ** 2).sqrt() + 1e-6
            x = (x - self.running_mean) / std

        logit_p = self.p_network(x)
        v = self.v_network(x)
        distr = torch.distributions.Categorical(logits=logit_p)
        prob_uniform = torch.ones_like(distr.probs) / distr.probs.shape[1]
        final_probs = distr.probs * (1 - self.prob_uniform) + prob_uniform * self.prob_uniform
        return v, torch.distributions.Categorical(probs=final_probs)

    def v(self, x):
        return self.v_network(x)
