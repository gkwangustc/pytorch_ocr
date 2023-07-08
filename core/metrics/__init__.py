from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy

__all__ = ["build_metric"]

from .cls_metric import ClsMetric


def build_metric(config):
    support_dict = ["ClsMetric"]

    config = copy.deepcopy(config)
    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "metric only support {}".format(support_dict)
    )
    module_class = eval(module_name)(**config)
    return module_class
