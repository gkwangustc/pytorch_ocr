from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
from core.modeling.transforms import build_transform
from core.modeling.backbones import build_backbone
from core.modeling.necks import build_neck
from core.modeling.heads import build_head
from .base_model import BaseModel
from core.utils.save_load import load_pretrained_params

__all__ = ['DistillationModel']


class DistillationModel(nn.Module):

    def __init__(self, config):
        """
        the module for OCR distillation.
        args:
            config (dict): the super parameters for module.
        """
        super().__init__()
        self.model_name_list = []
        model_list = []
        for key in config["Models"]:
            model_config = config["Models"][key]
            freeze_params = False
            pretrained = None
            if "freeze_params" in model_config:
                freeze_params = model_config.pop("freeze_params")
            if "pretrained" in model_config:
                pretrained = model_config.pop("pretrained")
            model = BaseModel(model_config)
            if pretrained is not None:
                load_pretrained_params(model, pretrained)
            if freeze_params:
                for param in model.parameters():
                    param.requires_grad= False
            model_list.append(model)
            self.model_name_list.append(key)
        self.model_list = nn.ModuleList(model_list)

    def forward(self, x, data=None):
        result_dict = dict()
        for idx, model_name in enumerate(self.model_name_list):
            result_dict[model_name] = self.model_list[idx](x, data)
        return result_dict
