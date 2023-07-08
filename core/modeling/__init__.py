from .resnet import ResNet

__all__ = ["build_model"]


def build_model(config):
    support_dict = ['ResNet']
    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        "model only support {}".format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
