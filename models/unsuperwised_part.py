if y is None:  # unsupervised
    batch_size = x.shape[0]
    # Do standard forward pass for everything not involving y
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
        zx_p_loc, zx_p_scale = torch.zeros(zd_p_loc.size()[0], self.zx_dim).cuda(), \
                               torch.ones(zd_p_loc.size()[0], self.zx_dim).cuda()

    pzd = dist.Normal(zd_p_loc, zd_p_scale)

    if self.zx_dim != 0:
        pzx = dist.Normal(zx_p_loc, zx_p_scale)
    else:
        pzx = None

    d_hat = self.qd(zd_q)

    x_recon = self.px(zd_q, zx_q, zy_q)

    # print(x_recon.shape, x.shape)
    # x_recon = x_recon.view(-1, 256)
    # x_target = (x.view(-1) * 255).long()
    if self.recon_loss == "cross_entropy":
        CE_x = F.binary_cross_entropy(x_recon, x, reduction='sum')
    elif self.recon_loss == "MSE":
        CE_x = torch.nn.MSELoss(reduction='sum')(x_recon, x)
    else:
        raise NotImplementedError

    zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
    if self.zx_dim != 0:
        KL_zx = torch.sum(pzx.log_prob(zx_q) - qzx.log_prob(zx_q))
    else:
        KL_zx = 0

    _, d_target = d.max(dim=1)
    CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')

    # Create labels and repeats of zy_q and qzy
    y_onehot = torch.eye(10)
    y_onehot = y_onehot.repeat(1, batch_size)
    y_onehot = y_onehot.view(batch_size * 10, 10).cuda()

    zy_q = zy_q.repeat(10, 1)
    zy_q_loc, zy_q_scale = zy_q_loc.repeat(10, 1), zy_q_scale.repeat(10, 1)
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

    prior_y = torch.tensor(1 / 10).cuda()
    prior_y_minus_qy = torch.log(prior_y) - qy.log_prob(y_onehot)
    marginal_prior_y_minus_qy = torch.sum(prob_qy * prior_y_minus_qy)

    return CE_x \
           - self.beta_d * zd_p_minus_zd_q \
           - self.beta_x * KL_zx \
           - self.beta_y * marginal_zy_p_minus_zy_q \
           - marginal_prior_y_minus_qy \
           + self.aux_loss_multiplier_d * CE_d, 0

