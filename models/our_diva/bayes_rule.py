import torch
from torch import nn
import torch.distributions as dist


class BayesRule(nn.Module):
    def __init__(self, posterior_dim, likelihood, device):
        super(BayesRule, self).__init__()
        self.likelihood = likelihood
        self.posterior_dim = posterior_dim
        self.device = device

    def forward(self, zd: torch.Tensor):
        d_one_eye = torch.eye(self.posterior_dim, device=self.device)
        zd_r = zd.repeat(self.posterior_dim, 1)
        d_one_eye_r = d_one_eye.repeat(1, zd.shape[0]).view(-1, self.posterior_dim)
        zd_p_loc, zd_p_scale = self.likelihood(d_one_eye_r)
        pz = dist.Normal(zd_p_loc, zd_p_scale)
        prob = pz.log_prob(zd_r)
        prob = torch.sum(prob, dim=1)
        prob = torch.t(prob.view(self.posterior_dim, -1))
        max_prob = torch.max(prob, dim=1).values
        prob -= max_prob[:, None]
        log_prob_sum = torch.log(torch.sum(torch.exp(prob), dim=1))
        prob = prob - log_prob_sum[:, None]
        return prob
