import torch
from tensorboardX import SummaryWriter
from torch import nn

from models.our_diva.our_diva import OurDIVA


class ClOf(OurDIVA):

    def __init__(self, args, writer: SummaryWriter, device):
        super(ClOf, self).__init__(args, writer, device)
        self.name = "ClOfPx"
        self.device = device
        self.use_KL_close = True
        self.use_bayes = False
        self.freeze_classifiers = False
        self.freeze_priors = False

        self.px = ClOfPx(self.zd_dim, self.zx_dim, self.zy_dim, self.x_w, self.x_h, self.x_c)

        self.qzd = CalOffQzd(self.x_dim, self.zd_dim)
        if self.zx_dim != 0:
            self.qzx = CalOffQzx(self.x_dim, self.zx_dim)
        self.qzy = CalOffQzy(self.x_dim, self.zy_dim)


class ClOfPx(nn.Module):
    def __init__(self, zd_dim, zx_dim, zy_dim, x_w, x_h, x_c):
        super(ClOfPx, self).__init__()
        self.zd_dim, self.zx_dim, self.zy_dim, self.x_w, self.x_h, self.x_c = zd_dim, zx_dim, zy_dim, x_w, x_h, x_c
        self.fc = nn.Sequential(nn.Linear(self.zd_dim + self.zx_dim + self.zy_dim, self.x_w * self.x_h), nn.Sigmoid())

    def forward(self, zd, zx, zy):
        if zx is None:
            zdzxzy = torch.cat((zd, zy), dim=-1)
        else:
            zdzxzy = torch.cat((zd, zx, zy), dim=-1)
        h = self.fc(zdzxzy)
        batch_size = zy.size(0)
        return h.view(batch_size, self.x_c, self.x_h, self.x_w)


class CalOffEncoder(nn.Module):
    def __init__(self, x_dim, latent_dim):
        super(CalOffEncoder, self).__init__()
        self.x_dim = x_dim
        self.fc_mean = nn.Sequential(nn.Linear(self.x_dim, latent_dim))
        self.fc_logvar = nn.Sequential(nn.Linear(self.x_dim, latent_dim), nn.Softplus())

    def forward(self, x):
        h = x.view(-1, self.x_dim)
        zd_loc = self.fc_mean(h)
        zd_scale = self.fc_logvar(h) + 1e-7
        return zd_loc, zd_scale


class CalOffQzd(CalOffEncoder):
    def __init__(self, x_dim, zd_dim):
        super(CalOffQzd, self).__init__(x_dim, zd_dim)


class CalOffQzy(CalOffEncoder):
    def __init__(self, x_dim, zy_dim):
        super(CalOffQzy, self).__init__(x_dim, zy_dim)


class CalOffQzx(CalOffEncoder):
    def __init__(self, x_dim, zx_dim):
        super(CalOffQzx, self).__init__(x_dim, zx_dim)
