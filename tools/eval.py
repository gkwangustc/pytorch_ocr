from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, __dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import torch
from core.data import build_dataloader
from core.modeling import build_model
from core.postprocess import build_post_process
from core.metrics import build_metric
from core.utils.save_load import load_model
import tools.program as program


def main(config, device, logger, log_writer):
    global_config = config["Global"]
    # build dataloader
    valid_dataloader = build_dataloader(config, "Eval", device, logger)

    # build post process
    post_process_class = build_post_process(config["PostProcess"], global_config)

    # build model
    model = build_model(config["Architecture"])
    model.to(device)

    # build metric
    eval_class = build_metric(config["Metric"])

    best_model_dict = load_model(config, model)
    if len(best_model_dict):
        logger.info("metric in ckpt ***************")
        for k, v in best_model_dict.items():
            logger.info("{}:{}".format(k, v))

    # start eval
    metric = program.eval(model, valid_dataloader, post_process_class, eval_class, device)
    logger.info("metric eval ***************")
    for k, v in metric.items():
        logger.info("{}:{}".format(k, v))


if __name__ == "__main__":
    config, device, logger, log_writer= program.preprocess()
    main(config, device, logger, log_writer)
