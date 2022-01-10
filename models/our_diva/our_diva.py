import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
from tensorboardX import SummaryWriter

from models.our_diva.bayes_rule import BayesRule
from models.our_diva.encoder import Encoder
from models.our_diva.freezable_conditional_prior import FreezableConditionalPrior
from models.our_diva.px import Px


class FreezablePzd(FreezableConditionalPrior):

    def __init__(self, d_dim, zd_dim, learned_domain):
        super(FreezablePzd, self).__init__(d_dim, zd_dim)
        self.learned_domain = learned_domain

    def freeze_grad_hook(self, grad):
        mask = torch.unsqueeze(1 - self.learned_domain, 0)
        mask = mask.repeat(grad.shape[0], 1)
        return grad * mask


class FreezablePzy(FreezableConditionalPrior):

    def __init__(self, y_dim, zy_dim, learned_domain):
        super(FreezablePzy, self).__init__(y_dim, zy_dim)
        self.learned_domain = learned_domain

    def freeze_grad_hook(self, grad):
        if torch.any(self.learned_domain):
            return grad * 0
        return grad


class Qzd(Encoder):
    def __init__(self, x_dim, zd_dim):
        super(Qzd, self).__init__(x_dim, zd_dim, 2000, 2000)


class Qzy(Encoder):
    def __init__(self, x_dim, zy_dim):
        super(Qzy, self).__init__(x_dim, zy_dim, 2000, 2000)


class Qzx(Encoder):
    def __init__(self, x_dim, zx_dim):
        super(Qzx, self).__init__(x_dim, zx_dim, 2000, 2000)


class Qd(BayesRule):
    def __init__(self, d_dim, pzd, device):
        super(Qd, self).__init__(d_dim, pzd, device)


class Qy(BayesRule):
    def __init__(self, y_dim, pzy, device):
        super(Qy, self).__init__(y_dim, pzy, device)


class OurDIVA(nn.Module):

    def __init__(self, args, writer: SummaryWriter, device):
        super(OurDIVA, self).__init__()
        self.name = "OurDIVA"
        self.device = device

        model_config = args['model']
        self.zd_dim = model_config['zd_dim']
        self.zx_dim = model_config['zx_dim']
        self.zy_dim = model_config['zy_dim']
        self.x_c = args['x_c']
        self.d_dim = args['d_dim']
        self.x_h, self.x_w = args['x_h'], args['x_w']
        self.x_dim = args['x_h'] * args['x_w']
        self.y_dim = args['y_dim']

        self.class_num = args['y_dim']
        self.writer = writer

        self.start_zx = self.zd_dim
        self.start_zy = self.zd_dim + self.zx_dim

        self.learned_domain = nn.Parameter(torch.zeros(self.d_dim, device=self.device), requires_grad=False)

        self.px = Px(self.zd_dim, self.zx_dim, self.zy_dim, self.x_w, self.x_h, self.x_c)
        self.pzd = FreezablePzd(self.d_dim, self.zd_dim, self.learned_domain)
        self.pzy = FreezablePzy(self.y_dim, self.zy_dim, self.learned_domain)

        self.qzd = Qzd(self.x_dim, self.zd_dim)
        if self.zx_dim != 0:
            self.qzx = Qzx(self.x_dim, self.zx_dim)
        self.qzy = Qzy(self.x_dim, self.zy_dim)

        self.qd = Qd(self.d_dim, self.pzd, self.device)
        self.qy = Qy(self.y_dim, self.pzy, self.device)

        self.aux_loss_multiplier_y = model_config['aux_loss_multiplier_y']
        self.aux_loss_multiplier_d = model_config['aux_loss_multiplier_d']

        self.beta_d = model_config['beta_d']
        self.beta_x = model_config['beta_x']
        self.beta_y = model_config['beta_y']

    def generate_replay_batch(self, batch_size):
        if not torch.any(self.learned_domain):
            return None
        y = torch.randint(low=0, high=self.y_dim, size=(batch_size,), device=self.device)
        d = self.learned_domain.multinomial(num_samples=batch_size, replacement=True).to(self.device)

        with torch.no_grad():
            x = self.generate_supervised_image(d, y)

        return x, y, d

    def generate_supervised_image(self, d, y):
        d_eye = torch.eye(self.d_dim, device=self.device)
        y_eye = torch.eye(self.y_dim, device=self.device)
        y = y_eye[y]
        d = d_eye[d]
        y, d = y.to(self.device), d.to(self.device)
        batch_size = len(d)

        zd_p_loc, zd_p_scale = self.pzd(d)

        if self.zx_dim != 0:
            zx_p_loc, zx_p_scale = torch.zeros(batch_size, self.zx_dim, device=self.device), \
                                   torch.ones(batch_size, self.zx_dim, device=self.device)

        zy_p_loc, zy_p_scale = self.pzy(y)

        # Reparameterization trick
        pzd = dist.Normal(zd_p_loc, zd_p_scale)
        zd_p = pzd.rsample()

        if self.zx_dim != 0:
            pzx = dist.Normal(zx_p_loc, zx_p_scale)
            zx_p = pzx.rsample()
        else:
            zx_p = None

        pzy = dist.Normal(zy_p_loc, zy_p_scale)
        zy_p = pzy.rsample()

        x_recon = self.px(zd_p, zx_p, zy_p)

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
            CE_x = F.binary_cross_entropy(x_recon, x, reduction='sum')
            zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))

            if self.zx_dim != 0:
                KL_zx = torch.sum(pzx.log_prob(zx_q) - qzx.log_prob(zx_q))
            else:
                KL_zx = 0

            _, d_target = d.max(dim=1)
            CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')

            # Create labels and repeats of zy_q and qzy
            y_onehot = torch.eye(self.class_num, device=self.device)
            y_onehot = y_onehot.repeat(1, batch_size)
            y_onehot = y_onehot.view(self.class_num * batch_size, self.class_num)

            zy_q = zy_q.repeat(self.y_dim, 1)
            zy_q_loc, zy_q_scale = zy_q_loc.repeat(self.y_dim, 1), zy_q_scale.repeat(self.y_dim, 1)
            qzy = dist.Normal(zy_q_loc, zy_q_scale)

            # Do forward pass for everything involving y
            zy_p_loc, zy_p_scale = self.pzy(y_onehot)

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

            prior_y = torch.tensor(1 / self.class_num, device=self.device)
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

            CE_x = F.binary_cross_entropy(x_recon, x, reduction='sum')

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

    def task_learned(self, stage_num, dataset):
        print(f"task {stage_num} learned!")
        for d in range(self.d_dim):
            if len(dataset.stage_classes(stage_num, d)) > 0:
                self.learned_domain[d] = 1

    def train_with_replayed_data(self, d, x, y):
        return self.loss_function(d, x, y)
