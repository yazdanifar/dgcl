from torch import nn


class Encoder(nn.Module):
    def __init__(self, x_dim, latent_dim, hidden1, hidden2):
        super(Encoder, self).__init__()
        self.x_dim = x_dim
        self.fc1 = nn.Sequential(nn.Linear(self.x_dim, hidden1), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hidden1, hidden2), nn.ReLU())
        self.fc_mean = nn.Sequential(nn.Linear(hidden2, latent_dim))
        self.fc_logvar = nn.Sequential(nn.Linear(hidden2, latent_dim), nn.Softplus())

    def forward(self, x):
        h = x.view(-1, self.x_dim)
        h = self.fc1(h)
        h = self.fc2(h)
        zd_loc = self.fc_mean(h)
        zd_scale = self.fc_logvar(h) + 1e-7
        return zd_loc, zd_scale
