from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn


class CTCLoss(nn.Module):
    def __init__(self, blank=0, reduction="none", use_focal_loss=False):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(
            blank=blank, reduction=reduction, zero_infinity=True
        )
        self.use_focal_loss = use_focal_loss

    def forward(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]

        if len(predicts.shape) == 4:
            predicts = predicts.flatten(1, 2)

        predicts = predicts.permute(1, 0, 2).log_softmax(2)
        N, B, _ = predicts.shape
        preds_lengths = torch.tensor([N] * B, dtype=torch.long)

        labels = batch[1].long().to(predicts.device)
        label_lengths = batch[2].long()

        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        if self.use_focal_loss:
            weight = torch.exp(-loss)
            weight = 1 - weight
            weight = torch.square(weight)
            loss = loss.mul(weight)
        loss = loss.mean()
        return {"loss": loss}
