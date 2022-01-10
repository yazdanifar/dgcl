from abc import abstractmethod

from models.our_diva.conditional_prior import ConditionalPrior


class FreezableConditionalPrior(ConditionalPrior):
    def __init__(self, cond_dim, rand_var_dim):
        super(FreezableConditionalPrior, self).__init__(cond_dim, rand_var_dim)

        self.fc_mean[0].weight.register_hook(self.freeze_grad_hook)
        self.fc_logvar[0].weight.register_hook(self.freeze_grad_hook)

    @abstractmethod
    def freeze_grad_hook(self, grad):
        pass
