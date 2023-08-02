# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn


class SARLoss(nn.Module):
    def __init__(self, **kwargs):
        super(SARLoss, self).__init__()
        ignore_index = kwargs.get("ignore_index", 92)  # 6626
        self.loss_func = torch.nn.CrossEntropyLoss(
            reduction="mean", ignore_index=ignore_index
        )

    def forward(self, predicts, batch):
        predict = predicts[
            :, :-1, :
        ]  # ignore last index of outputs to be in same seq_len with targets
        label = batch[1].long()[ :, 1: ]  # ignore first index of target in loss calculation
        batch_size, num_steps, num_classes = (
            predict.shape[0],
            predict.shape[1],
            predict.shape[2],
        )
        assert (
            len(label.shape) == len(list(predict.shape)) - 1
        ), "The target's shape and inputs's shape is [N, d] and [N, num_steps]"

        inputs = torch.reshape(predict, [-1, num_classes])
        targets = torch.reshape(label, [-1])
        loss = self.loss_func(inputs, targets)
        return {"loss": loss}
