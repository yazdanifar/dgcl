import abc
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
from tensorboardX import SummaryWriter

from models.our_diva.bayes_rule import BayesRule
from models.our_diva.conditional_prior import ConditionalPrior
from models.our_diva.encoder import Encoder
from models.our_diva.freezable_conditional_prior import FreezableConditionalPrior
from models.our_diva.px import Px


class Pzd(ConditionalPrior):
    def __init__(self, d_dim, zd_dim):
        super(Pzd, self).__init__(d_dim, zd_dim)


class FreezablePzd(FreezableConditionalPrior):

    def __init__(self, d_dim, zd_dim, learned_domain):
        super(FreezablePzd, self).__init__(d_dim, zd_dim)
        self.learned_domain = learned_domain

    def freeze_grad_hook(self, grad):
        mask = torch.unsqueeze(1 - self.learned_domain, 0)
        mask = mask.repeat(grad.shape[0], 1)
        return grad * mask


class Pzy(ConditionalPrior):

    def __init__(self, y_dim, zy_dim):
        super(Pzy, self).__init__(y_dim, zy_dim)


class FreezablePzy(FreezableConditionalPrior):

    def __init__(self, y_dim, zy_dim, learned_domain):
        super(FreezablePzy, self).__init__(y_dim, zy_dim)
        self.learned_domain = learned_domain

    def freeze_grad_hook(self, grad):
        if torch.any(self.learned_domain):
            return grad * 0
        return grad


class FreezableLinearClassifier(nn.Module, abc.ABC):
    def __init__(self, input_dim, class_num):
        super(FreezableLinearClassifier, self).__init__()
        self.linear_layer = nn.Linear(input_dim, class_num)
        self.net = nn.Sequential(self.linear_layer, nn.Softmax(dim=1))
        self.linear_layer.weight.register_hook(self.freeze_grad_hook)

    @abc.abstractmethod
    def freeze_grad_hook(self, grad):
        pass

    def forward(self, x):
        return self.net(x)


class FreezableDomainClassifier(FreezableLinearClassifier):
    def __init__(self, input_dim, class_num, learned_domain):
        super(FreezableDomainClassifier, self).__init__(input_dim, class_num)
        self.learned_domain = learned_domain

    def freeze_grad_hook(self, grad):
        mask = torch.unsqueeze(1 - self.learned_domain, 0)
        mask = mask.repeat(grad.shape[1], 1).t()
        return grad * mask


class FreezableLabelClassifier(FreezableLinearClassifier):
    def __init__(self, input_dim, class_num, learned_domain):
        super(FreezableLabelClassifier, self).__init__(input_dim, class_num)
        self.learned_domain = learned_domain

    def freeze_grad_hook(self, grad):
        if torch.any(self.learned_domain):
            return grad * 0
        return grad


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, class_num):
        super(LinearClassifier, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, class_num), nn.Softmax(dim=1))

    def forward(self, x):
        return self.net(x)


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
        self.use_KL_close = True
        self.use_bayes = False
        self.freeze_classifiers = True
        self.freeze_priors = True

        model_config = args['model']

        self.recon_loss = model_config['recon_loss'] #"MSE" "cross_entropy"
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
        if self.freeze_priors:
            self.pzd = FreezablePzd(self.d_dim, self.zd_dim, self.learned_domain)
            self.pzy = FreezablePzy(self.y_dim, self.zy_dim, self.learned_domain)
        else:
            self.pzd = Pzd(self.d_dim, self.zd_dim)
            self.pzy = Pzy(self.y_dim, self.zy_dim)

        self.qzd = Qzd(self.x_dim, self.zd_dim)
        if self.zx_dim != 0:
            self.qzx = Qzx(self.x_dim, self.zx_dim)
        self.qzy = Qzy(self.x_dim, self.zy_dim)

        if self.use_bayes:
            self.qd = Qd(self.d_dim, self.pzd, self.device)
            self.qy = Qy(self.y_dim, self.pzy, self.device)
        else:
            if self.freeze_classifiers:
                self.qd = FreezableDomainClassifier(self.zd_dim, self.d_dim, self.learned_domain)
                self.qy = FreezableLabelClassifier(self.zy_dim, self.y_dim, self.learned_domain)
            else:
                self.qd = LinearClassifier(self.zd_dim, self.d_dim)
                self.qy = LinearClassifier(self.zy_dim, self.y_dim)

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

    @staticmethod
    def reparameterize(mu, std):
        '''Perform "reparametrization trick" to make these stochastic variables differentiable.'''
        eps = std.new_empty(std.shape).normal_()  # .requires_grad_()
        return eps.mul(std).add_(mu)

    @staticmethod
    def log_likelihood_dif(point, mu1, std1, mu2, std2):
        '''calculate in a differentiable way log p(point| mu1, std1)-log p(point| mu2, std2)'''
        return torch.sum(
            torch.log(std2) - torch.log(std1) + (((point - mu2) / std2) ** 2 - ((point - mu1) / std1) ** 2) / 2)

    @staticmethod
    def kl_distribution(mu1, std1, mu2, std2):
        '''calculate in a differentiable KL (N(mu1,std1) || N(mu2, std2) all inputs are '''
        return torch.sum((torch.log(std2) - torch.log(std1)) + ((std1 ** 2 + (mu1 - mu2) ** 2) / (2 * std2 ** 2)) - 0.5)

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
        zd_p = OurDIVA.reparameterize(zd_p_loc, zd_p_scale)

        if self.zx_dim != 0:
            zx_p = OurDIVA.reparameterize(zx_p_loc, zx_p_scale)

        else:
            zx_p = None

        zy_p = OurDIVA.reparameterize(zy_p_loc, zy_p_scale)

        x_recon = self.px(zd_p, zx_p, zy_p)

        return x_recon

    def forward(self, d, x):
        zd_q_loc, zd_q_scale, zx_q_loc, zx_q_scale, zy_q_loc, zy_q_scale = self.infer_latent(x)

        # Reparameterization trick
        zd_q = OurDIVA.reparameterize(zd_q_loc, zd_q_scale)
        if self.zx_dim != 0:
            zx_q = OurDIVA.reparameterize(zx_q_loc, zx_q_scale)
        else:
            zx_q = None

        zy_q = OurDIVA.reparameterize(zy_q_loc, zy_q_scale)

        # Decode
        x_recon = self.px(zd_q, zx_q, zy_q)

        zd_p_loc, zd_p_scale = self.pzd(d)

        # Auxiliary losses
        d_hat = self.qd(zd_q)
        y_hat = self.qy(zy_q)

        return x_recon, d_hat, y_hat, (zd_q_loc, zd_q_scale), (zd_p_loc, zd_p_scale), zd_q, (zx_q_loc, zx_q_scale) \
            , zx_q, (zy_q_loc, zy_q_scale), zy_q

    def infer_latent(self, x, disable_qzx=False):
        # Encode
        zd_q_loc, zd_q_scale = self.qzd(x)
        if self.zx_dim != 0 and not disable_qzx:
            zx_q_loc, zx_q_scale = self.qzx(x)
        else:
            zx_q_loc, zx_q_scale = None, None
        zy_q_loc, zy_q_scale = self.qzy(x)
        return zd_q_loc, zd_q_scale, zx_q_loc, zx_q_scale, zy_q_loc, zy_q_scale

    def prior_px(self, batch_size):
        zx_p_loc, zx_p_scale = torch.zeros(batch_size, self.zx_dim, device=self.device), \
                               torch.ones(batch_size, self.zx_dim, device=self.device)

        return zx_p_loc, zx_p_scale

    def loss_function(self, d, x, y=None):
        if y is None:  # unsupervised
            # Do standard forward pass for everything not involving y
            batch_size = d.shape[0]
            x_recon, d_hat, y_hat, (zd_q_loc, zd_q_scale), (zd_p_loc, zd_p_scale), zd_q, (zx_q_loc, zx_q_scale) \
                , zx_q, (zy_q_loc, zy_q_scale), zy_q = self.forward(d, x)

            if self.zx_dim != 0:
                zx_p_loc, zx_p_scale = self.prior_px(zd_p_loc.size()[0])
            else:
                zx_p_loc, zx_p_scale = None, None

            if self.recon_loss=="cross_entropy":
                CE_x = F.binary_cross_entropy(x_recon, x, reduction='sum')
            elif self.recon_loss=="MSE":
                CE_x = torch.nn.MSELoss(reduction='sum')(x_recon, x)
            else:
                raise NotImplementedError

            if self.use_KL_close:
                zd_p_minus_zd_q = -OurDIVA.kl_distribution(zd_q_loc, zd_q_scale, zd_p_loc, zd_p_scale)

                if self.zx_dim != 0:
                    KL_zx = -OurDIVA.kl_distribution(zx_q_loc, zx_q_scale, zx_p_loc, zx_p_scale)
                else:
                    KL_zx = 0
            else:
                zd_p_minus_zd_q = OurDIVA.log_likelihood_dif(zd_q, zd_p_loc, zd_p_scale, zd_q_loc, zd_q_scale)

                if self.zx_dim != 0:
                    KL_zx = OurDIVA.log_likelihood_dif(zx_q, zx_p_loc, zx_p_scale, zx_q_loc, zx_q_scale)
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

            # Do forward pass for everything involving y
            zy_p_loc, zy_p_scale = self.pzy(y_onehot)

            # Marginals
            prob_qy = torch.exp(y_hat).t().reshape(-1)

            zy_p_minus_zy_q = torch.sum(torch.log(zy_q_scale) - torch.log(zy_p_scale) + (
                    ((zy_q - zy_q_loc) / zy_q_scale) ** 2 - ((zy_q - zy_p_loc) / zy_p_scale) ** 2) / 2, dim=1)
            marginal_zy_p_minus_zy_q = torch.sum(prob_qy * zy_p_minus_zy_q)

            prior_y_minus_qy = -math.log(self.class_num) - y_hat.t().reshape(-1)
            marginal_prior_y_minus_qy = torch.sum(prob_qy * prior_y_minus_qy)
            return CE_x \
                   - self.beta_d * zd_p_minus_zd_q \
                   - self.beta_x * KL_zx \
                   - self.beta_y * marginal_zy_p_minus_zy_q \
                   - marginal_prior_y_minus_qy \
                   + self.aux_loss_multiplier_d * CE_d, 0

        else:  # supervised
            x_recon, d_hat, y_hat, (zd_q_loc, zd_q_scale), (zd_p_loc, zd_p_scale), zd_q, (zx_q_loc, zx_q_scale) \
                , zx_q, (zy_q_loc, zy_q_scale), zy_q = self.forward(d, x)
            zy_p_loc, zy_p_scale = self.pzy(y)

            if self.recon_loss == "cross_entropy":
                CE_x = F.binary_cross_entropy(x_recon, x, reduction='sum')
            elif self.recon_loss == "MSE":
                CE_x = torch.nn.MSELoss(reduction='sum')(x_recon, x)
            else:
                raise NotImplementedError


            if self.use_KL_close:
                zd_p_minus_zd_q = -OurDIVA.kl_distribution(zd_q_loc, zd_q_scale, zd_p_loc, zd_p_scale)
                zy_p_minus_zy_q = -OurDIVA.kl_distribution(zy_q_loc, zy_q_scale, zy_p_loc, zy_p_scale)
                if self.zx_dim != 0:
                    zx_p_loc, zx_p_scale = self.prior_px(zd_p_loc.size()[0])
                    KL_zx = -OurDIVA.kl_distribution(zx_q_loc, zx_q_scale, zx_p_loc, zx_p_scale)
                else:
                    zx_p_loc, zx_p_scale = None, None
                    KL_zx = 0
            else:
                zd_p_minus_zd_q = OurDIVA.log_likelihood_dif(zd_q, zd_p_loc, zd_p_scale, zd_q_loc, zd_q_scale)
                zy_p_minus_zy_q = OurDIVA.log_likelihood_dif(zy_q, zy_p_loc, zy_p_scale, zy_q_loc, zy_q_scale)

                if self.zx_dim != 0:
                    zx_p_loc, zx_p_scale = self.prior_px(zd_p_loc.size()[0])
                    KL_zx = OurDIVA.log_likelihood_dif(zx_q, zx_p_loc, zx_p_scale, zx_q_loc, zx_q_scale)
                else:
                    zx_p_loc, zx_p_scale = None, None
                    KL_zx = 0

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
            zd_q_loc, zd_q_scale, _, _, zy_q_loc, zy_q_scale = self.infer_latent(x, disable_qzx=True)

            zd = zd_q_loc
            alpha = self.qd(zd)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            res, ind = torch.topk(alpha, 1)

            # convert the digit(s) to one-hot tensor(s)
            d = x.new_zeros(alpha.size())
            d = d.scatter_(1, ind, 1.0)

            zy = zy_q_loc
            alpha = self.qy(zy)

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
