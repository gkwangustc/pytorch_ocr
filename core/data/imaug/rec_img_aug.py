import math
import cv2
import numpy as np
import random
import copy
from PIL import Image


class RecResizeImg(object):
    def __init__(
        self,
        image_shape,
        infer_mode=False,
        character_dict_path="./ppocr/utils/ppocr_keys_v1.txt",
        padding=True,
        **kwargs
    ):
        self.image_shape = image_shape
        self.infer_mode = infer_mode
        self.character_dict_path = character_dict_path
        self.padding = padding

    def __call__(self, data):
        img = data["image"]
        if self.infer_mode and self.character_dict_path is not None:
            norm_img, valid_ratio = resize_norm_img_chinese(img, self.image_shape)
        else:
            norm_img, valid_ratio = resize_norm_img(img, self.image_shape, self.padding)
        data["image"] = norm_img
        data["valid_ratio"] = valid_ratio
        return data


def resize_norm_img(img, image_shape, padding=True, interpolation=cv2.INTER_LINEAR):
    imgC, imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    if not padding:
        resized_image = cv2.resize(img, (imgW, imgH), interpolation=interpolation)
        resized_w = imgW
    else:
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype("float32")
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))
    return padding_im, valid_ratio


def resize_norm_img_chinese(img, image_shape):
    imgC, imgH, imgW = image_shape
    # todo: change to 0 and modified image shape
    max_wh_ratio = imgW * 1.0 / imgH
    h, w = img.shape[0], img.shape[1]
    ratio = w * 1.0 / h
    max_wh_ratio = max(max_wh_ratio, ratio)
    imgW = int(imgH * max_wh_ratio)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype("float32")
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))
    return padding_im, valid_ratio


class RecConAug(object):
    def __init__(
        self,
        prob=0.5,
        image_shape=(32, 320, 3),
        max_text_length=25,
        ext_data_num=1,
        **kwargs
    ):
        self.ext_data_num = ext_data_num
        self.prob = prob
        self.max_text_length = max_text_length
        self.image_shape = image_shape
        self.max_wh_ratio = self.image_shape[1] / self.image_shape[0]

    def merge_ext_data(self, data, ext_data):
        ori_w = round(
            data["image"].shape[1] / data["image"].shape[0] * self.image_shape[0]
        )
        ext_w = round(
            ext_data["image"].shape[1]
            / ext_data["image"].shape[0]
            * self.image_shape[0]
        )
        data["image"] = cv2.resize(data["image"], (ori_w, self.image_shape[0]))
        ext_data["image"] = cv2.resize(ext_data["image"], (ext_w, self.image_shape[0]))
        data["image"] = np.concatenate([data["image"], ext_data["image"]], axis=1)
        data["label"] += ext_data["label"]
        return data

    def __call__(self, data):
        rnd_num = random.random()
        if rnd_num > self.prob:
            return data
        for idx, ext_data in enumerate(data["ext_data"]):
            if len(data["label"]) + len(ext_data["label"]) > self.max_text_length:
                break
            concat_ratio = (
                data["image"].shape[1] / data["image"].shape[0]
                + ext_data["image"].shape[1] / ext_data["image"].shape[0]
            )
            if concat_ratio > self.max_wh_ratio:
                break
            data = self.merge_ext_data(data, ext_data)
        data.pop("ext_data")
        return data
