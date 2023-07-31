from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import platform
import yaml
import time
import datetime
import torch
import torch.distributed as dist
from tqdm import tqdm
import cv2
import numpy as np
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from core.utils.stats import TrainingStats
from core.utils.save_load import save_model
from core.utils.utility import print_dict, AverageMeter
from core.utils.logging import get_logger
from core.utils.loggers import WandbLogger, Loggers
from core.data import build_dataloader


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c", "--config", help="configuration file to use")
        self.add_argument("-o", "--opt", nargs="+", help="set configuration options")
        self.add_argument(
            "--world-size", default=1, type=int, help="number of distributed process"
        )
        self.add_argument(
            "--dist-url",
            default="env://",
            type=str,
            help="url used to set up distributed training",
        )
        self.add_argument(
            "--dist-backend", default="nccl", type=str, help="distributed backend"
        )
        self.add_argument("--local_rank", default=0, type=int, help="local_rank")
        self.add_argument("--global_rank", default=0, type=int, help="global_rank")

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split("=")
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config


def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in [".yml", ".yaml"], "only support yaml files for now"
    config = yaml.load(open(file_path, "rb"), Loader=yaml.Loader)
    return config


def merge_config(config, opts):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in opts.items():
        if "." not in key:
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        else:
            sub_keys = key.split(".")
            assert sub_keys[0] in config, (
                "the sub_keys can only be one of global_config: {}, but get: "
                "{}, please check your running command".format(
                    config.keys(), sub_keys[0]
                )
            )
            cur = config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]
    return config


def check_gpu(use_gpu):
    """
    Log error and exit when set use_gpu=true in
    cpu version.
    """
    err = (
        "Config {} cannot be set as true while your pytorch"
        "is not compiled with {} ! \nPlease try: \n"
        "\t1. Install pytorch to run model on {} \n"
        "\t2. Set {} as false in config file to run "
        "model on CPU"
    )

    try:
        if use_gpu and not torch.cuda.is_available():
            print(err.format("use_gpu", "cuda", "gpu", "use_gpu"))
            sys.exit(1)
    except Exception as e:
        pass


def train(
    config,
    train_dataloader,
    valid_dataloader,
    device,
    model,
    loss_class,
    optimizer,
    lr_scheduler,
    post_process_class,
    eval_class,
    pre_best_model_dict,
    logger,
    log_writer=None,
):
    cal_metric_during_train = config["Global"].get("cal_metric_during_train", False)
    calc_epoch_interval = config["Global"].get("calc_epoch_interval", 1)
    log_smooth_window = config["Global"]["log_smooth_window"]
    epoch_num = config["Global"]["epoch_num"]
    print_batch_step = config["Global"]["print_batch_step"]
    eval_batch_step = config["Global"]["eval_batch_step"]

    global_step = 0
    if "global_step" in pre_best_model_dict:
        global_step = pre_best_model_dict["global_step"]
    start_eval_step = 0
    if type(eval_batch_step) == list and len(eval_batch_step) >= 2:
        start_eval_step = eval_batch_step[0]
        eval_batch_step = eval_batch_step[1]
        if len(valid_dataloader) == 0:
            logger.info(
                "No Images in eval dataset, evaluation during training "
                "will be disabled"
            )
            start_eval_step = 1e111
        logger.info(
            "During the training process, after the {}th iteration, "
            "an evaluation is run every {} iterations".format(
                start_eval_step, eval_batch_step
            )
        )
    save_epoch_step = config["Global"]["save_epoch_step"]
    save_model_dir = config["Global"]["save_model_dir"]
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    main_indicator = eval_class.main_indicator
    best_model_dict = {main_indicator: 0}
    best_model_dict.update(pre_best_model_dict)
    train_stats = TrainingStats(log_smooth_window, ["lr"])
    model.train()

    extra_input_models = ["SAR", "SVTR"]
    extra_input = False
    if config["Architecture"]["algorithm"] == "Distillation":
        for key in config["Architecture"]["Models"]:
            extra_input = (
                extra_input
                or config["Architecture"]["Models"][key]["algorithm"]
                in extra_input_models
            )
    else:
        extra_input = config["Architecture"]["algorithm"] in extra_input_models
    try:
        model_type = config["Architecture"]["model_type"]
    except:
        model_type = None

    start_epoch = (
        best_model_dict["start_epoch"] if "start_epoch" in best_model_dict else 1
    )

    total_samples = 0
    train_reader_cost = 0.0
    train_batch_cost = 0.0
    reader_start = time.time()
    eta_meter = AverageMeter()

    max_iter = (
        len(train_dataloader) - 1
        if platform.system() == "Windows"
        else len(train_dataloader)
    )

    for epoch in range(start_epoch, epoch_num + 1):
        if config["Global"]["distributed"]:
            train_dataloader.sampler.set_epoch(epoch)

        for idx, batch in enumerate(train_dataloader):
            train_reader_cost += time.time() - reader_start
            if idx >= max_iter:
                break
            if not isinstance(lr_scheduler, float):
                lr = lr_scheduler.get_last_lr()
            else:
                lr = lr_scheduler

            model.zero_grad()
            batch = [item.to(device) for item in batch]
            images = batch[0]
            if extra_input:
                preds = model(images, data=batch[1:])
            else:
                preds = model(images)
            loss = loss_class(preds, batch)
            avg_loss = loss["loss"]
            avg_loss.backward()
            optimizer.step()

            if (
                cal_metric_during_train and epoch % calc_epoch_interval == 0
            ):  # only rec and cls need
                batch = [item.detach().cpu().numpy() for item in batch]
                if config["Loss"]["name"] in ["MultiLoss"]:  # for multi head loss
                    post_result = post_process_class(
                        preds["ctc"], batch[1]
                    )  # for CTC head out
                else:
                    post_result = post_process_class(preds, batch[1])
                eval_class(post_result, batch)
                metric = eval_class.get_metric()
                train_stats.update(metric)

            train_batch_time = time.time() - reader_start
            train_batch_cost += train_batch_time
            eta_meter.update(train_batch_time)
            global_step += 1
            total_samples += len(images)

            if not isinstance(lr_scheduler, float):
                lr_scheduler.step()

            # logger and visualdl
            stats = {k: v.detach().cpu().numpy().mean() for k, v in loss.items()}
            stats["lr"] = lr
            train_stats.update(stats)

            if log_writer is not None and config["rank"] == 0:
                log_writer.log_metrics(
                    metrics=train_stats.get(), prefix="TRAIN", step=global_step
                )

            if config["rank"] == 0 and (
                (global_step > 0 and global_step % print_batch_step == 0)
                or (idx >= len(train_dataloader) - 1)
            ):
                logs = train_stats.log()

                eta_sec = (
                    (epoch_num + 1 - epoch) * len(train_dataloader) - idx - 1
                ) * eta_meter.avg
                eta_sec_format = str(datetime.timedelta(seconds=int(eta_sec)))
                strs = (
                    "epoch: [{}/{}], global_step: {}, {}, avg_reader_cost: "
                    "{:.5f} s, avg_batch_cost: {:.5f} s, avg_samples: {}, "
                    "ips: {:.5f} samples/s, eta: {}".format(
                        epoch,
                        epoch_num,
                        global_step,
                        logs,
                        train_reader_cost / print_batch_step,
                        train_batch_cost / print_batch_step,
                        total_samples / print_batch_step,
                        total_samples / train_batch_cost,
                        eta_sec_format,
                    )
                )
                logger.info(strs)

                total_samples = 0
                train_reader_cost = 0.0
                train_batch_cost = 0.0
            # eval
            if (
                global_step > start_eval_step
                and (global_step - start_eval_step) % eval_batch_step == 0
                and config["rank"] == 0
            ):
                cur_metric = eval(
                    model,
                    valid_dataloader,
                    post_process_class,
                    eval_class,
                    device,
                    extra_input=extra_input,
                )
                cur_metric_str = "cur metric, {}".format(
                    ", ".join(["{}: {}".format(k, v) for k, v in cur_metric.items()])
                )
                logger.info(cur_metric_str)

                # logger metric
                if log_writer is not None:
                    log_writer.log_metrics(
                        metrics=cur_metric, prefix="EVAL", step=global_step
                    )

                if cur_metric[main_indicator] >= best_model_dict[main_indicator]:
                    best_model_dict.update(cur_metric)
                    best_model_dict["best_epoch"] = epoch
                    save_model(
                        model,
                        optimizer,
                        save_model_dir,
                        logger,
                        config,
                        is_best=True,
                        prefix="best_accuracy",
                        best_model_dict=best_model_dict,
                        epoch=epoch,
                        global_step=global_step,
                    )
                best_str = "best metric, {}".format(
                    ", ".join(
                        ["{}: {}".format(k, v) for k, v in best_model_dict.items()]
                    )
                )
                logger.info(best_str)
                # logger best metric
                if log_writer is not None:
                    log_writer.log_metrics(
                        metrics={
                            "best_{}".format(main_indicator): best_model_dict[
                                main_indicator
                            ]
                        },
                        prefix="EVAL",
                        step=global_step,
                    )

                    log_writer.log_model(
                        is_best=True, prefix="best_accuracy", metadata=best_model_dict
                    )

            reader_start = time.time()
        if config["rank"] == 0:
            save_model(
                model,
                optimizer,
                save_model_dir,
                logger,
                config,
                is_best=False,
                prefix="latest",
                best_model_dict=best_model_dict,
                epoch=epoch,
                global_step=global_step,
            )

            if log_writer is not None:
                log_writer.log_model(is_best=False, prefix="latest")

        if config["rank"] == 0 and epoch > 0 and epoch % save_epoch_step == 0:
            save_model(
                model,
                optimizer,
                save_model_dir,
                logger,
                config,
                is_best=False,
                prefix="iter_epoch_{}".format(epoch),
                best_model_dict=best_model_dict,
                epoch=epoch,
                global_step=global_step,
            )
            if log_writer is not None:
                log_writer.log_model(
                    is_best=False, prefix="iter_epoch_{}".format(epoch)
                )

    best_str = "best metric, {}".format(
        ", ".join(["{}: {}".format(k, v) for k, v in best_model_dict.items()])
    )
    logger.info(best_str)
    if config["rank"] == 0 and log_writer is not None:
        log_writer.close()
    return


def eval(
    model,
    valid_dataloader,
    post_process_class,
    eval_class,
    device,
    extra_input=False,
):
    model.eval()
    with torch.no_grad():
        total_frame = 0.0
        total_time = 0.0
        pbar = tqdm(
            total=len(valid_dataloader), desc="eval model:", position=0, leave=True
        )
        max_iter = (
            len(valid_dataloader) - 1
            if platform.system() == "Windows"
            else len(valid_dataloader)
        )
        sum_images = 0
        for idx, batch in enumerate(valid_dataloader):
            if idx >= max_iter:
                break
            images = batch[0].to(device)
            start = time.time()

            if extra_input:
                preds = model(images, data=batch[1:])
            else:
                preds = model(images)

            batch_numpy = []
            for item in batch:
                if isinstance(item, torch.Tensor):
                    batch_numpy.append(item.cpu().detach().numpy())
                else:
                    batch_numpy.append(item)
            # Obtain usable results from post-processing methods
            total_time += time.time() - start
            # Evaluate the results of the current batch
            post_result = post_process_class(preds, batch_numpy[1])
            eval_class(post_result, batch_numpy)

            pbar.update(1)
            total_frame += len(images)
            sum_images += 1
        # Get final metric，eg. acc or hmean
        metric = eval_class.get_metric()

    pbar.close()
    model.train()
    metric["fps"] = total_frame / total_time
    return metric


def preprocess(is_train=False):
    FLAGS = ArgsParser().parse_args()
    config = load_config(FLAGS.config)
    config = merge_config(config, FLAGS.opt)

    use_gpu = config["Global"]["use_gpu"]
    check_gpu(use_gpu)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        config["rank"] = int(os.environ["RANK"])
        config["local_rank"] = int(os.environ["LOCAL_RANK"])
        config["world_size"] = int(os.environ["WORLD_SIZE"])
        config["Global"]["distributed"] = True
        dist.init_process_group(backend="nccl")
    else:
        config["rank"] = 0
        config["local_rank"] = 0
        config["Global"]["distributed"] = False

    device = "cuda:{}".format(config["local_rank"]) if use_gpu else "cpu"
    device = torch.device(device)

    if is_train:
        # save_config
        save_model_dir = config["Global"]["save_model_dir"]
        os.makedirs(save_model_dir, exist_ok=True)
        with open(os.path.join(save_model_dir, "config.yml"), "w") as f:
            yaml.dump(dict(config), f, default_flow_style=False, sort_keys=False)
        log_file = "{}/train.log".format(save_model_dir)
    else:
        log_file = None
    logger = get_logger(log_file=log_file, rank=config["rank"])

    loggers = []

    if (
        "use_wandb" in config["Global"] and config["Global"]["use_wandb"]
    ) or "wandb" in config:
        save_model_dir = config["Global"]["save_model_dir"]
        wandb_writer_path = "{}/wandb".format(save_model_dir)
        if "wandb" in config:
            wandb_params = config["wandb"]
        else:
            wandb_params = dict()
        wandb_params.update({"save_dir": save_model_dir})
        log_writer = WandbLogger(**wandb_params, config=config)
        loggers.append(log_writer)
    else:
        log_writer = None
    print_dict(config, logger)

    if loggers:
        log_writer = Loggers(loggers)
    else:
        log_writer = None

    logger.info("train with pytorch {} and device {}".format(torch.__version__, device))
    return config, device, logger, log_writer
