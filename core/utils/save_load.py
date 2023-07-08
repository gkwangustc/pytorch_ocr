from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import os

import torch
import torch.nn as nn

from core.utils.logging import get_logger

__all__ = ["load_model", "save_model"]


def _mkdir_if_not_exist(path, logger):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    "be happy if some process has already created {}".format(path)
                )
            else:
                raise OSError("Failed to mkdir {}".format(path))


def load_model(config, model, optimizer=None):
    """
    load model from checkpoint or pretrained_model
    """
    logger = get_logger()
    global_config = config["Global"]
    checkpoints = global_config.get("checkpoints")
    pretrained_model = global_config.get("pretrained_model")
    best_model_dict = {}
    if checkpoints:
        model_name = checkpoints
    elif pretrained_model:
        model_name = pretrained_model
    else:
        logger.info("train from scratch")
        return best_model_dict

    if model_name.endswith(".pth"):
        model_name = model_name.replace(".pth", "")
    assert os.path.exists(model_name + ".pth"), "The {}.pth does not exists!".format(
        model_name
    )

    # load params from trained model
    params = torch.load(model_name + ".pth")
    save_model_dict = params["model_state_dict"]

    current_model_dict = model.state_dict()
    new_state_dict = {
        k: v if v.size() == current_model_dict[k].size() else current_model_dict[k]
        for k, v in zip(current_model_dict.keys(), save_model_dict.values())
    }

    model.load_state_dict(new_state_dict, strict=False)

    if optimizer is not None:
        optimizer.load_state_dict(params["optimizer_state_dict"])

    logger.info("resume from {}".format(model_name))
    return best_model_dict


def save_model(
    model,
    optimizer,
    model_path,
    logger,
    config,
    is_best=False,
    prefix="pt_base",
    **kwargs
):
    """
    save model to the target path
    """
    _mkdir_if_not_exist(model_path, logger)
    model_prefix = os.path.join(model_path, prefix)
    if isinstance(model, nn.parallel.DistributedDataParallel):
        torch.save(
            {
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            model_prefix + ".pth",
        )
    else:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            model_prefix + ".pth",
        )

    if is_best:
        logger.info("save best model is to {}".format(model_prefix))
    else:
        logger.info("save model in {}".format(model_prefix))
