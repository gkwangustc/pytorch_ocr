import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvCTCHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=6625,
        return_feats=False,
        use_guide=False,
        **kwargs
    ):
        super(ConvCTCHead, self).__init__()
        self.out_channels = out_channels
        self.embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.last_bn = nn.BatchNorm2d(out_channels)

        self.use_guide = use_guide
        self.return_feats = return_feats

    def forward(self, feat, targets=None):
        if self.use_guide:
            z = feat.clone()
            z.stop_gradient = True
        else:
            z = feat
        if len(z.shape) == 3:
            z = z.permute(0, 2, 1)
            z = z.unsqueeze(2)

        x = self.last_bn(self.embedding(z))
        predicts = x.permute(0, 2, 3, 1)

        if self.return_feats:
            result = (feat, predicts)
        else:
            result = predicts

        if not self.training:
            predicts = F.softmax(predicts, dim=2)
            result = predicts

        return result
