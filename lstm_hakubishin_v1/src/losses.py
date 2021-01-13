import torch
import torch.nn.functional as F
from torch import nn


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


