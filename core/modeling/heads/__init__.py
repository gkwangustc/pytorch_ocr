__all__ = ["build_head"]


def build_head(config, **kwargs):
    # rec head
    from .rec_ctc_head import CTCHead
    from .rec_sar_head import SARHead
    from .rec_multi_head import MultiHead
    from .rec_conv_ctc_head import ConvCTCHead

    support_dict = ["CTCHead", "SARHead", "MultiHead", "ConvCTCHead"]

    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "head only support {}".format(support_dict)
    )
    print(config)
    module_class = eval(module_name)(**config, **kwargs)
    return module_class
