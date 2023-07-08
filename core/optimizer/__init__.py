from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import copy
import torch
from torch import optim
import ipdb


__all__ = ["build_optimizer"]


def build_optimizer(config, epochs, step_each_epoch, model, loss=None):
    config = copy.deepcopy(config)

    # step1 build lr
    lr = config["lr"]["learning_rate"]

    # step2 build regularization

    if "regularizer" in config and config["regularizer"] is not None:
        support_dict = ["L2"]
        reg_name = config["regularizer"]["name"]
        assert reg_name in support_dict, Exception(
            "regularizer only support {}".format(support_dict)
        )

        reg = config["regularizer"]["factor"]
    else:
        reg = None
    skip = {}
    skip_keywords = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()
    if hasattr(model, "no_weight_decay_keywords"):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords, loss)

    # step3 build optimizer
    optim_name = config["name"]
    lr_scheduler = lr
    if optim_name == "Adam":
        beta1 = config["beta1"]
        beta2 = config["beta2"]
        optimizer = optim.Adam(parameters, lr=lr, betas=(beta1, beta2))
    elif optim_name == "AdamW":
        beta1 = config["beta1"]
        beta2 = config["beta2"]
        optimizer = optim.AdamW(parameters, lr=lr, betas=(beta1, beta2))
    elif optim_name == "Adadelta":
        optimizer = optim.Adadelta(parameters)
    elif optim_name == "SGD":
        optimizer = optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else:
        optimizer = optim.RMSprop(parameters, lr=lr)

    return optimizer, lr_scheduler


def set_weight_decay(model, skip_list=(), skip_keywords=(), loss=None):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or (name in skip_list)
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    if loss != None:
        has_decay += list(loss.parameters())
    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
