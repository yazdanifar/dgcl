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
        ans=h.view(batch_size, self.x_c, self.x_h, self.x_w)
        return ans


class convpx(nn.Module):
    def __init__(self, d_dim, x_w, x_h, x_c, y_dim, zd_dim, zx_dim, zy_dim):
        super(convpx, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(zd_dim + zx_dim + zy_dim, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU())
        self.up1 = nn.Upsample(8)
        self.de1 = nn.Sequential(nn.ConvTranspose2d(64, 128, kernel_size=5, stride=1, padding=0, bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.up2 = nn.Upsample(24)
        self.de2 = nn.Sequential(nn.ConvTranspose2d(128, 256, kernel_size=5, stride=1, padding=0, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self.de3 = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1, stride=1), nn.Sigmoid())

    def forward(self, zd, zx, zy):
        if zx is None:
            zdzxzy = torch.cat((zd, zy), dim=-1)
        else:
            zdzxzy = torch.cat((zd, zx, zy), dim=-1)
        h = self.fc1(zdzxzy)
        h = h.view(-1, 64, 4, 4)
        h = self.up1(h)
        h = self.de1(h)
        h = self.up2(h)
        h = self.de2(h)
        loc_img = self.de3(h)

        return loc_img