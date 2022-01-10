import torch
from torch import nn


class Px(nn.Module):
    def __init__(self, zd_dim, zx_dim, zy_dim, x_w, x_h, x_c):
        super(Px, self).__init__()
        self.zd_dim, self.zx_dim, self.zy_dim, self.x_w, self.x_h, self.x_c = zd_dim, zx_dim, zy_dim, x_w, x_h, x_c
        self.fc1 = nn.Sequential(nn.Linear(self.zd_dim + self.zx_dim + self.zy_dim, 2000), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(2000, 2000), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(2000, self.x_w * self.x_h), nn.Sigmoid())

    def forward(self, zd, zx, zy):
        if zx is None:
            zdzxzy = torch.cat((zd, zy), dim=-1)
        else:
            zdzxzy = torch.cat((zd, zx, zy), dim=-1)
        h = self.fc1(zdzxzy)
        h = self.fc2(h)
        h = self.fc3(h)
        batch_size = zy.size(0)
        return h.view(batch_size, self.x_c, self.x_h, self.x_w)
