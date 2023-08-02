# Copyright (c) 2023 pytorch_ocr Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import torch
import time
import onnx

from onnxruntime import InferenceSession

import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))

from core.modeling.architectures import build_model
from core.postprocess import build_post_process
from core.utils.save_load import load_model
import tools.program as program


def main(config, device, logger, log_writer):
    global_config = config["Global"]
    # build dataloader
    post_process_class = build_post_process(config["PostProcess"], global_config)

    # build model
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["algorithm"] in [
            "Distillation",
        ]:  # distillation model
            is_distill = True
            for key in config["Architecture"]["Models"]:
                if (
                    config["Architecture"]["Models"][key]["Head"]["name"] == "MultiHead"
                ):  # for multi head
                    out_channels_list = {}
                    if config["PostProcess"]["name"] == "DistillationSARLabelDecode":
                        char_num = char_num - 2
                    out_channels_list["CTCLabelDecode"] = char_num
                    out_channels_list["SARLabelDecode"] = char_num + 2
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels_list"
                    ] = out_channels_list
                else:
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels"
                    ] = char_num
                config["Architecture"]["Models"][key]["return_all_feats"] = False
        elif config["Architecture"]["Head"]["name"] == "MultiHead":  # for multi head
            out_channels_list = {}
            if config["PostProcess"]["name"] == "SARLabelDecode":
                char_num = char_num - 2
            out_channels_list["CTCLabelDecode"] = char_num
            out_channels_list["SARLabelDecode"] = char_num + 2
            config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        else:  # base rec model
            config["Architecture"]["Head"]["out_channels"] = char_num

    model = build_model(config["Architecture"])
    load_model(config, model)

    arch_config = config["Architecture"]
    if arch_config["algorithm"] in [
        "Distillation",
    ]:
        for idx, name in enumerate(model.model_name_list):
            if name != "Student":
                continue
            model = model.model_list[idx]

    model.eval()
    model.to(device)

    # export model
    save_path = "{}/inference.onnx".format(config["Global"]["save_model_dir"])

    dummy_input = torch.randn(8, 3, 48, 320).to(device)

    # trt5
    # torch.onnx.export(model,
    #                  dummy_input,
    #                  save_path,
    #                  verbose=True,
    #                  export_params=True,
    #                  input_names=['input'],
    #                  output_names=['output'])
    # trt7
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        verbose=True,
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    # load onnx model to compare
    sess = InferenceSession(save_path)
    test_model = onnx.load(save_path)
    input_names = [node.name for node in test_model.graph.input]
    output_names = [node.name for node in test_model.graph.output]

    x = dummy_input.cpu().detach().numpy().astype("float32")

    start = time.time()
    ort_outs = sess.run(output_names=None, input_feed={"input": x})
    end = time.time()

    print("Exported model has been predicted by ONNXRuntime!")
    print("ONNXRuntime predict time: %.04f s" % (end - start))

    pytorch_outs = model(torch.tensor(x).to(device))

    diff = ort_outs[0] - pytorch_outs.cpu().detach().numpy()
    max_abs_diff = np.fabs(diff).max()
    if max_abs_diff < 1e-05:
        print("The difference of results between ONNXRuntime and Pytorch looks good!")
    else:
        relative_diff = (
            max_abs_diff / np.fabs(pytorch_outs.cpu().detach().numpy()).max()
        )
        if relative_diff < 1e-05:
            print(
                "The difference of results between ONNXRuntime and Pytorch looks good!"
            )
        else:
            print(
                "The difference of results between ONNXRuntime and Pytorch looks bad!"
            )
        print("relative_diff: ", relative_diff)
    print("max_abs_diff: ", max_abs_diff)


if __name__ == "__main__":
    config, device, logger, log_writer = program.preprocess()
    main(config, device, logger, log_writer)
