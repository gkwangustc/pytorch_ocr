__all__ = ['build_backbone']


def build_backbone(config, model_type):
    from .rec_mobilenet_v3 import MobileNetV3
    from .rec_resnet_vd import ResNet
    from .rec_svtrnet import SVTRNet
    support_dict = ['MobileNetV3', 'ResNet', 'SVTRNet']

    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'when model typs is {}, backbone only support {}'.format(
            model_type, support_dict))
    module_class = eval(module_name)(**config)
    return module_class
