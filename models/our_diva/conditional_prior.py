from torch import nn


class ConditionalPrior(nn.Module):
    def __init__(self, cond_dim, rand_var_dim):
        super(ConditionalPrior, self).__init__()
        self.fc_mean = nn.Sequential(nn.Linear(cond_dim, rand_var_dim, bias=False))
        self.fc_logvar = nn.Sequential(nn.Linear(cond_dim, rand_var_dim, bias=False), nn.Softplus())

    def forward(self, y):
        zy_loc = self.fc_mean(y)
        zy_scale = self.fc_logvar(y) + 1e-7
        return zy_loc, zy_scale
