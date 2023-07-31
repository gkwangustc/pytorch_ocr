#! /bin/bash
python tools/infer.py -c configs/rec/ch_PY-OCR_rec_ctc.yaml \
       -o Global.pretrained_model=output/rec_pyocr_ctc/best_accuracy \
       Global.infer_img=dataset/rec/chinese/baidu/train_images