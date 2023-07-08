import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(self, epsilon=None):
        super().__init__()
        if epsilon is not None and (epsilon <= 0 or epsilon >= 1):
            epsilon = None
        self.epsilon = epsilon

    def _labelsmoothing(self, target, class_num):
        if target.shape[-1] != class_num:
            one_hot_target = F.one_hot(target, class_num)
        else:
            one_hot_target = target
        soft_target = F.label_smooth(one_hot_target, epsilon=self.epsilon)
        soft_target = torch.reshape(soft_target, shape=(-1, class_num))
        return soft_target

    def forward(self, x, label):
        loss_dict = {}
        if self.epsilon is not None:
            class_num = x.shape[-1]
            label = self._labelsmoothing(label, class_num)
            x = -F.log_softmax(x, dim=-1)
            loss = torch.sum(x * label, dim=-1)
        else:
            if label.shape[-1] == x.shape[-1]:
                label = F.softmax(label, dim=-1)
                soft_label = True
            else:
                soft_label = False
            loss = F.cross_entropy(x, label=label, soft_label=soft_label)
        return loss


class ClsLoss(nn.Module):
    def __init__(self, **kwargs):
        super(ClsLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, predicts, batch):
        label = batch[1].long().to(predicts.device)
        loss = self.loss_func(predicts, label)
        return {"loss": loss}
