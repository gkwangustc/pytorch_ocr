#! /bin/bash
python tools/export_onnx.py -c configs/rec/ch_PY-OCR_rec_ctc.yaml \
       -o Global.pretrained_model=output/rec_pyocr_ctc/best_accuracy