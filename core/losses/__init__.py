import copy

# basic_loss
from .basic_loss import CELoss, ClsLoss


def build_loss(config):
    support_dict = ["CELoss", "ClsLoss"]
    config = copy.deepcopy(config)
    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "loss only support {}".format(support_dict)
    )
    module_class = eval(module_name)(**config)
    return module_class
