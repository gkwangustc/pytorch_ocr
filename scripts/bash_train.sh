#! /bin/bash
# for single gpu and cpu
python tools/train.py -c configs/rec/ch_PY-OCR_rec_ctc.yaml

# for mutli gpu
#CUDA_VISIBLE_DEVICES=0,1 torchrun \
#    --nproc_per_node=2 \
#    --master_port 61232 tools/train.py \
#    -c configs/rec/ch_PY-OCR_rec_ctc.yaml