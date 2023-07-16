import copy

# basic_loss
from .basic_loss import CELoss, ClsLoss

from .rec_sar_loss import SARLoss
from .rec_ctc_loss import CTCLoss
from .rec_multi_loss import MultiLoss

from .combined_loss import CombinedLoss


def build_loss(config):
    support_dict = [
        "CELoss",
        "ClsLoss",
        "CTCLoss",
        "SARLoss",
        "MultiLoss",
        "CombinedLoss",
    ]
    config = copy.deepcopy(config)
    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "loss only support {}".format(support_dict)
    )
    module_class = eval(module_name)(**config)
    return module_class
