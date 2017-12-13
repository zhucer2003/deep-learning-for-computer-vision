import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropy2d:

    def __init__(self):
        super(CrossEntropy2d, self).__init__()

    def wrap(self, inputs, targets, weight=None, pixel_average=True):
        n, c, h, w = inputs.size()

        # after this we have n, h, w, c = inputs.size()
        inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous()
        # expand targets to a tensor with depth of c
        inputs = inputs[targets.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

        # also exclude background and unlabeled
        targets_mask = targets >= 0
        targets = targets[targets_mask]

        loss = F.cross_entropy(inputs, targets, weight=weight, size_average=False)
        if pixel_average:
            loss /= targets_mask.data.sum()
        return loss