import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
from tensorboardX import SummaryWriter


### Follows model as seen in LEARNING ROBUST REPRESENTATIONS BY PROJECTING SUPERFICIAL STATISTICS OUT

# Decoders
class px(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(px, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(zd_dim + zx_dim + zy_dim, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU())
        self.up1 = nn.Upsample(8)
        self.de1 = nn.Sequential(nn.ConvTranspose2d(64, 128, kernel_size=5, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(128), nn.ReLU())
        self.up2 = nn.Upsample(24)
        self.de2 = nn.Sequential(nn.ConvTranspose2d(128, 256, kernel_size=5, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(256), nn.ReLU())
        self.de3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1))

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.de1[0].weight)
        torch.nn.init.xavier_uniform_(self.de2[0].weight)
        torch.nn.init.xavier_uniform_(self.de3[0].weight)
        self.de3[0].bias.data.zero_()

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


class pzd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, device):
        super(pzd, self).__init__()
        self.device = device
        self.fc1 = nn.Sequential(nn.Linear(d_dim, zd_dim, bias=False), nn.BatchNorm1d(zd_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(zd_dim, zd_dim))
        self.fc22 = nn.Sequential(nn.Linear(zd_dim, zd_dim), nn.Softplus())
        self.d_dim = d_dim
        self.zd_dim = zd_dim

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

        self.learned_domain = torch.zeros(d_dim, device=self.device)
        self.learned_loc = torch.zeros((d_dim, zd_dim), device=self.device)
        self.learned_scale = torch.zeros((d_dim, zd_dim), device=self.device)

    def forward(self, d):
        dd = torch.argmax(d, dim=1, keepdim=True)
        loaded_loc = torch.gather(self.learned_loc, dim=0, index=dd.repeat(1, self.zd_dim)).detach()
        loaded_scale = torch.gather(self.learned_scale, dim=0, index=dd.repeat(1, self.zd_dim)).detach()

        hidden = self.fc1(d)
        zd_loc = self.fc21(hidden)
        zd_scale = self.fc22(hidden) + 1e-7

        loc = torch.cat((torch.unsqueeze(zd_loc, 0), torch.unsqueeze(loaded_loc, 0)), 0)
        scale = torch.cat((torch.unsqueeze(zd_scale, 0), torch.unsqueeze(loaded_scale, 0)), 0)
        which_one = torch.gather(self.learned_domain, dim=0, index=torch.squeeze(dd, dim=1)).long()
        which_one = which_one.view(1, -1, 1).repeat(1, 1, self.zd_dim)

        zd_loc = torch.squeeze(torch.gather(loc, dim=0, index=which_one), dim=0)
        zd_scale = torch.squeeze(torch.gather(scale, dim=0, index=which_one), dim=0)
        return zd_loc, zd_scale

    def learn(self, d):
        with torch.no_grad():
            input_d = torch.zeros((2, self.d_dim), device=self.device)
            input_d[0, d] = 1
            loc, scale = self(input_d)
            self.learned_loc[d], self.learned_scale[d] = loc[0].detach(), scale[0].detach()
            self.learned_domain[d] = 1


class pzy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(pzy, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(y_dim, zy_dim, bias=False), nn.BatchNorm1d(zy_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(zy_dim, zy_dim))
        self.fc22 = nn.Sequential(nn.Linear(zy_dim, zy_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, y):
        hidden = self.fc1(y)
        zy_loc = self.fc21(hidden)
        zy_scale = self.fc22(hidden) + 1e-7

        return zy_loc, zy_scale


# Encoders
class qzd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qzd, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        )

        self.fc11 = nn.Sequential(nn.Linear(1024, zd_dim))
        self.fc12 = nn.Sequential(nn.Linear(1024, zd_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[4].weight)
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(-1, 1024)
        zd_loc = self.fc11(h)
        zd_scale = self.fc12(h) + 1e-7

        return zd_loc, zd_scale


class qzx(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qzx, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        )

        self.fc11 = nn.Sequential(nn.Linear(1024, zx_dim))
        self.fc12 = nn.Sequential(nn.Linear(1024, zx_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[4].weight)
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(-1, 1024)
        zx_loc = self.fc11(h)
        zx_scale = self.fc12(h) + 1e-7

        return zx_loc, zx_scale


class qzy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qzy, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        )

        self.fc11 = nn.Sequential(nn.Linear(1024, zy_dim))
        self.fc12 = nn.Sequential(nn.Linear(1024, zy_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[4].weight)
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(-1, 1024)
        zy_loc = self.fc11(h)
        zy_scale = self.fc12(h) + 1e-7

        return zy_loc, zy_scale


# Auxiliary tasks
class qd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qd, self).__init__()

        self.fc1 = nn.Linear(zd_dim, d_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zd):
        h = F.relu(zd)
        loc_d = self.fc1(h)

        return loc_d


class qy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qy, self).__init__()

        self.fc1 = nn.Linear(zy_dim, y_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zy):
        h = F.relu(zy)
        loc_y = self.fc1(h)

        return loc_y


class DIVA(nn.Module):
    def __init__(self, args, writer: SummaryWriter, device):
        super(DIVA, self).__init__()
        self.name = "DIVA"
        self.device = device

        self.zd_dim = args['zd_dim']
        self.zx_dim = args['zx_dim']
        self.zy_dim = args['zy_dim']
        self.d_dim = args['d_dim']
        self.x_dim = args['x_dim']
        self.y_dim = args['y_dim']

        self.class_num = args['y_dim']
        self.writer = writer

        self.start_zx = self.zd_dim
        self.start_zy = self.zd_dim + self.zx_dim

        self.px = px(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.pzd = pzd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, device)
        self.pzy = pzy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        self.qzd = qzd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        if self.zx_dim != 0:
            self.qzx = qzx(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.qzy = qzy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        self.use_aux_domain = args['use_aux_domain']
        self.use_aux_class = args['use_aux_class']
        self.freeze_latent_domain = args['freeze_latent_domain']

        if not self.use_aux_domain:
            self.qd = OurQd(self.d_dim, self.zd_dim, self.pzd, self.device)
        else:
            self.qd = qd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        if not self.use_aux_class:
            self.qy = OurQy(self.y_dim, self.zy_dim, self.pzy, self.device)
        else:
            self.qy = qy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        self.aux_loss_multiplier_y = args['aux_loss_multiplier_y']
        self.aux_loss_multiplier_d = args['aux_loss_multiplier_d']

        self.beta_d = args['beta_d']
        self.beta_x = args['beta_x']
        self.beta_y = args['beta_y']

        self.learned_domain = []

        self.cuda()

    def get_image_by_recon(self, x):
        x_shape = x.shape
        batch_size, h, w = x_shape[0], x_shape[2], x_shape[3]
        recon_batch = x.view(-1, 1, h, w, 256)  # TODO: make it more general channel!=1

        sample = torch.argmax(recon_batch, dim=4).float() / 255
        return sample

    def get_replay_batch(self, domain, batch_size):
        # we could replay based on some thing different
        # (if we forgot a task we could put more sample
        #  from that task in the batch) p!=uniform
        y = torch.randint(low=0, high=self.y_dim, size=batch_size, device=self.device)
        d = (torch.ones(batch_size, device=self.device) * domain).long()

        with torch.no_grad():
            x = self.generate_supervised_image(d, y)

        return x, y, d

    def generate_supervised_image(self, d, y):
        assert self.zx_dim != 0, "currently zx_dim=0 is not supported"
        d_eye = torch.eye(self.d_dim)
        y_eye = torch.eye(self.y_dim)
        y = y_eye[y]
        d = d_eye[d]
        y, d = y.to(self.device), d.to(self.device)
        batch_size = len(d)
        zd_p_loc, zd_p_scale = self.pzd(d)
        zx_p_loc, zx_p_scale = torch.zeros(batch_size, self.zx_dim, device=self.device), torch.ones(batch_size,
                                                                                                    self.zx_dim,
                                                                                                    device=self.device)
        zy_p_loc, zy_p_scale = self.pzy(y)

        # Reparameterization trick
        pzd = dist.Normal(zd_p_loc, zd_p_scale)
        zd_p = pzd.rsample()

        pzx = dist.Normal(zx_p_loc, zx_p_scale)
        zx_p = pzx.rsample()

        pzy = dist.Normal(zy_p_loc, zy_p_scale)
        zy_p = pzy.rsample()

        x_recon = self.px(zd_p, zx_p, zy_p)
        s_t = time.time()
        x_recon = self.get_image_by_recon(x_recon)
        # print("REAL USELESS:", round((time.time()-s_t)*100,3))
        return x_recon

    def forward(self, d, x, y):
        # Encode
        zd_q_loc, zd_q_scale = self.qzd(x)
        if self.zx_dim != 0:
            zx_q_loc, zx_q_scale = self.qzx(x)
        zy_q_loc, zy_q_scale = self.qzy(x)

        # Reparameterization trick
        qzd = dist.Normal(zd_q_loc, zd_q_scale)
        zd_q = qzd.rsample()
        if self.zx_dim != 0:
            qzx = dist.Normal(zx_q_loc, zx_q_scale)
            zx_q = qzx.rsample()
        else:
            qzx = None
            zx_q = None

        qzy = dist.Normal(zy_q_loc, zy_q_scale)
        zy_q = qzy.rsample()

        # Decode
        x_recon = self.px(zd_q, zx_q, zy_q)

        zd_p_loc, zd_p_scale = self.pzd(d)

        if self.zx_dim != 0:
            zx_p_loc, zx_p_scale = torch.zeros(zd_p_loc.size()[0], self.zx_dim, device=self.device), \
                                   torch.ones(zd_p_loc.size()[0], self.zx_dim, device=self.device)
        zy_p_loc, zy_p_scale = self.pzy(y)

        # Reparameterization trick
        pzd = dist.Normal(zd_p_loc, zd_p_scale)
        if self.zx_dim != 0:
            pzx = dist.Normal(zx_p_loc, zx_p_scale)
        else:
            pzx = None
        pzy = dist.Normal(zy_p_loc, zy_p_scale)

        # Auxiliary losses
        d_hat = self.qd(zd_q)
        y_hat = self.qy(zy_q)

        return x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q

    def loss_function(self, d, x, y=None):
        if y is None:  # unsupervised
            # Do standard forward pass for everything not involving y
            batch_size = d.shape[0]
            zd_q_loc, zd_q_scale = self.qzd(x)

            if self.zx_dim != 0:
                zx_q_loc, zx_q_scale = self.qzx(x)
            zy_q_loc, zy_q_scale = self.qzy(x)

            qzd = dist.Normal(zd_q_loc, zd_q_scale)

            zd_q = qzd.rsample()
            if self.zx_dim != 0:
                qzx = dist.Normal(zx_q_loc, zx_q_scale)
                zx_q = qzx.rsample()
            else:
                zx_q = None
            qzy = dist.Normal(zy_q_loc, zy_q_scale)
            zy_q = qzy.rsample()
            zd_p_loc, zd_p_scale = self.pzd(d)
            if self.zx_dim != 0:
                zx_p_loc, zx_p_scale = torch.zeros(zd_p_loc.size()[0], self.zx_dim, device=self.device), \
                                       torch.ones(zd_p_loc.size()[0], self.zx_dim, device=self.device)

            pzd = dist.Normal(zd_p_loc, zd_p_scale)

            if self.zx_dim != 0:
                pzx = dist.Normal(zx_p_loc, zx_p_scale)
            else:
                pzx = None

            d_hat = self.qd(zd_q)
            x_recon = self.px(zd_q, zx_q, zy_q)
            x_recon = x_recon.view(-1, 256)  # what is 256?(pixel domain)
            x_target = (x.view(-1) * 255).long()  # 255 is pixel scale
            CE_x = F.cross_entropy(x_recon, x_target, reduction='sum')
            zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))

            if self.zx_dim != 0:
                KL_zx = torch.sum(pzx.log_prob(zx_q) - qzx.log_prob(zx_q))
            else:
                KL_zx = 0

            _, d_target = d.max(dim=1)
            CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')

            # Create labels and repeats of zy_q and qzy
            y_onehot = torch.eye(self.class_num, device=self.device)  # what is 10?(class num)
            y_onehot = y_onehot.repeat(1, batch_size)  # what is 100?(repeat)
            y_onehot = y_onehot.view(self.class_num * batch_size,
                                     self.class_num)  # what is 1000,10?(class*batch, class)

            zy_q = zy_q.repeat(self.y_dim, 1)  # what is 10?(batch * class = class * batch)(B)
            # print(self.repeatB)
            zy_q_loc, zy_q_scale = zy_q_loc.repeat(self.y_dim, 1), zy_q_scale.repeat(self.y_dim,
                                                                                       1)  # what is 10?(B)
            qzy = dist.Normal(zy_q_loc, zy_q_scale)

            # Do forward pass for everything involving y
            zy_p_loc, zy_p_scale = self.pzy(
                y_onehot)  # TODO: why you should first repeat 100 times then forward pass it not in reverse?!

            # Reparameterization trick
            pzy = dist.Normal(zy_p_loc, zy_p_scale)

            # Auxiliary losses
            y_hat = self.qy(zy_q)

            # Marginals
            alpha_y = F.softmax(y_hat, dim=-1)
            qy = dist.OneHotCategorical(alpha_y)
            prob_qy = torch.exp(qy.log_prob(y_onehot))

            zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q), dim=-1)

            marginal_zy_p_minus_zy_q = torch.sum(prob_qy * zy_p_minus_zy_q)

            prior_y = torch.tensor(1 / self.class_num, device=self.device)  # what is 10?(class num) need change!
            prior_y_minus_qy = torch.log(prior_y) - qy.log_prob(y_onehot)
            marginal_prior_y_minus_qy = torch.sum(prob_qy * prior_y_minus_qy)
            return CE_x \
                   - self.beta_d * zd_p_minus_zd_q \
                   - self.beta_x * KL_zx \
                   - self.beta_y * marginal_zy_p_minus_zy_q \
                   - marginal_prior_y_minus_qy \
                   + self.aux_loss_multiplier_d * CE_d, 0

        else:  # supervised
            x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q = self.forward(d, x, y)

            x_recon = x_recon.view(-1, 256)  # what is 256? (pixel domain)
            x_target = (x.view(-1) * 255).long()  # 255 is pixel scale
            CE_x = F.cross_entropy(x_recon, x_target, reduction='sum')

            zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
            if self.zx_dim != 0:
                KL_zx = torch.sum(pzx.log_prob(zx_q) - qzx.log_prob(zx_q))
            else:
                KL_zx = 0

            zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q))

            _, d_target = d.max(dim=1)
            CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')

            _, y_target = y.max(dim=1)
            CE_y = F.cross_entropy(y_hat, y_target, reduction='sum')

            return CE_x \
                   - self.beta_d * zd_p_minus_zd_q \
                   - self.beta_x * KL_zx \
                   - self.beta_y * zy_p_minus_zy_q \
                   + self.aux_loss_multiplier_d * CE_d \
                   + self.aux_loss_multiplier_y * CE_y, \
                   CE_y

    def classifier(self, x):
        """
        classify an image (or a batch of images)
        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        with torch.no_grad():
            zd_q_loc, zd_q_scale = self.qzd(x)
            zd = zd_q_loc
            alpha = F.softmax(self.qd(zd), dim=1)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            res, ind = torch.topk(alpha, 1)

            # convert the digit(s) to one-hot tensor(s)
            d = x.new_zeros(alpha.size())
            d = d.scatter_(1, ind, 1.0)

            zy_q_loc, zy_q_scale = self.qzy.forward(x)
            zy = zy_q_loc
            alpha = F.softmax(self.qy(zy), dim=1)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            res, ind = torch.topk(alpha, 1)

            # convert the digit(s) to one-hot tensor(s)
            y = x.new_zeros(alpha.size())
            y = y.scatter_(1, ind, 1.0)

        return d, y

    def learn_task(self, stage_num, dataset):
        print(f"task {stage_num} learned!")
        learned_domain = []
        for d in range(self.d_dim):
            if len(dataset.stage_classes(stage_num, d)) > 0:
                learned_domain.append(d)
            self.learned_domain.append(d)

        if self.freeze_latent_domain:  # TODO: only work for domain incremental case
            for d in learned_domain:
                self.pzd.learn(d)


class OurQd(nn.Module):
    def __init__(self, d_dim, zd_dim, pzd, device):
        super(OurQd, self).__init__()
        self.pzd = pzd
        self.d_dim = d_dim
        self.zd_dim = zd_dim
        self.device = device

    def forward(self, zd: torch.Tensor):
        d_one_eye = torch.eye(self.d_dim, device=self.device)
        zd_r = zd.repeat(self.d_dim, 1)
        d_one_eye_r = d_one_eye.repeat(1, zd.shape[0]).view(-1, self.d_dim)
        zd_p_loc, zd_p_scale = self.pzd(d_one_eye_r)
        pz = dist.Normal(zd_p_loc, zd_p_scale)
        prob = pz.log_prob(zd_r)
        prob = torch.sum(prob, dim=1)
        prob = torch.t(prob.view(self.d_dim, -1))
        max_prob = torch.max(prob, dim=1).values
        prob -= max_prob[:, None]
        log_prob_sum = torch.log(torch.sum(torch.exp(prob), dim=1))
        prob = prob - log_prob_sum[:, None]
        return prob


class OurQy(nn.Module):
    def __init__(self, y_dim, zy_dim, pzy, device):
        super(OurQy, self).__init__()
        self.pzy = pzy
        self.y_dim = y_dim
        self.zy_dim = zy_dim
        self.device = device

    def forward(self, zy: torch.Tensor):
        y_one_eye = torch.eye(self.y_dim, device=self.device)
        zy_r = zy.repeat(self.y_dim, 1)
        y_one_eye_r = y_one_eye.repeat(1, zy.shape[0]).view(-1, self.y_dim)
        zy_p_loc, zy_p_scale = self.pzy(y_one_eye_r)
        pz = dist.Normal(zy_p_loc, zy_p_scale)
        prob = pz.log_prob(zy_r)
        prob = torch.sum(prob, dim=1)
        prob = torch.t(prob.view(self.y_dim, -1))
        max_prob = torch.max(prob, dim=1).values
        prob -= max_prob[:, None]
        log_prob_sum = torch.log(torch.sum(torch.exp(prob), dim=1))
        prob = prob - log_prob_sum[:, None]
        return prob
