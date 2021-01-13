import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.loss import _Loss
from pytorch_toolbelt.losses.functional import sigmoid_focal_loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, country_city_mat: torch.tensor, smooth_factor: float = 0.4) -> None:
        super().__init__()
        self.smooth_factor = smooth_factor
        self.country_city_mat = country_city_mat

    def _smooth_labels(self, num_classes, target):
        target_one_hot = F.one_hot(target, num_classes).float()
        with torch.no_grad():
            mask = self.country_city_mat[torch.argmax(self.country_city_mat[:, target], axis=0)]
            noise = self.smooth_factor / (mask.sum(axis=1) - 1)
            for i in range(target.size(0)):
                target_one_hot[i, mask[i].type(torch.bool)] = noise[i]
                target_one_hot[i, target[i]] = 1 - self.smooth_factor
        return target_one_hot

    def forward(self, input, target):
        logp = F.log_softmax(input, dim=1)
        target_one_hot = self._smooth_labels(input.size(1), target)
        return F.kl_div(logp, target_one_hot, reduction='sum')


def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.

    print(index)
    return mask.scatter_(1, index, ones)


# https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
class FocalLossWithOneHot(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLossWithOneHot, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = one_hot(target, input.size(-1))

        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.sum()


class FocalLossWithOutOneHot(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLossWithOutOneHot, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
        logit = logit.clamp(self.eps, 1. - self.eps)
        logit_ls = torch.log(logit)
        loss = F.nll_loss(logit_ls, target, reduction="none")
        view = target.size() + (1,)
        index = target.view(*view)
        loss = loss * (1 - logit.gather(1, index).squeeze(1)) ** self.gamma # focal loss

        return loss.sum()
