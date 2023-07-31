# pytorch_ocr

## 背景

使用pytorch_base框架，在OCR识别任务上来实现不同的OCR识别算法，OCR算法参考自PaddleOCR算法库。

## 安装

环境要求：

- torch >= 1.5.0
- torchvision >= 0.6.0

安装步骤

```shell
# 1. 下载源码
git clone --recursive https://github.com/gkwangustc/pytorch_ocr.git

# 2. 安装环境依赖
cd pytorch_ocr
pip install -r requirements.txt

```

## 使用

### 准备训练数据

以百度AI Studio提供的[中文街景数据集](https://aistudio.baidu.com/aistudio/datasetdetail/8429)作为训练数据，分别下载`train_images.tar.gz`以及`train.list`文件，放置在`dataset/rec/chinese/baidu`文件夹下，文件结构如下所示:

```shell
dataset/rec/chinese/baidu/
 ├── convert.py
 ├── train_images.tar.gz
 └── train.list
```

将压缩包解压后，执行文件转换脚本`convert.py`，即可完成训练集和验证集的拆分和转换

```shell
# cd至数据所在目录
cd dataset/rec/chinese/baidu/

# 解压训练数据
tar -xzvf train_images.tar.gz

# 执行数据拆分和转换，进行训练集/验证集的拆分，并转换标注文件的格式
python convert.py
```

### 配置网络

在`configs/rec`文件夹下提供四个样例配置文件，参考自PaddleOCR项目，配置分别如下：

- `ch_PY-OCR_rec_conv_ctc.yaml`: ResNet34为backbone，使用卷积层作为head，预测文字内容
- `ch_PY-OCR_rec_ctc.yaml`: ResNet34为backbone，使用`Im2Seq`作为neck，使用fc层作为head，预测文字内容
- `ch_PY-OCR_rec_gtc.yaml`: ResNet34为backbone，使用Attention模块(SAR Head)指导CTC训练，其中CTC分支使用SVTR + fc层作为head，预测文字内容
- `ch_PY-OCR_rec_gtc_distillation.yaml`: 使用UDML(联合互学习)策略，进一步提升GTC模型的准确率

### 训练

根据需要选择一个配置，进行单卡或多卡训练

- 单卡训练

```shell
python tools/train.py -c configs/rec/ch_PY-OCR_rec_ctc.yaml
```

- 多卡训练

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 61232 tools/train.py -c configs/rec/ch_PY-OCR_rec_ctc.yaml
```

### 评测

完成训练后，可以对模型进行评测，假设模型保存的文件夹为`output/rec_pyocr_ctc`，可以使用如下命令，对`ch_PY-OCR_rec_ctc.yaml`文件中的`Eval`数据集进行评测，得到准确率指标，

```shell
python tools/eval.py -c configs/rec/ch_PY-OCR_rec_ctc.yaml -o Global.pretrained_model=output/rec_pyocr_ctc/best_accuracy
```

### 模型导出

如有部署需求，可以将模型导出为`onnx`格式，以便进一步转换为`trt`等格式进行下一步的部署，可以使用一下脚本进行转换

```shell
python tools/export_onnx.py -c configs/rec/ch_PY-OCR_rec_ctc.yaml -o Global.pretrained_model=output/rec_pyocr_ctc/best_accuracy
```

## 性能

使用上述四个配置，分别进行训练和评测，得到的参考指标如下所示

## 参考项目

- [PaddlerOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [pytorch_base](https://github.com/gkwangustc/pytorch_base)