import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import cv2
from matplotlib import pyplot as plt
import numpy as np


def segement() :
    load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))
    bodypix_model = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))

    # s = 'images/person27.png'
    # frame = cv2.imread(s)
    #
    # # BodyPix Detections
    # result = bodypix_model.predict_single(frame)
    # mask = result.get_mask(threshold=0.5).numpy().astype(np.uint8)
    # masked_image = cv2.bitwise_and(frame, frame, mask=mask)
    #
    # # Show result to user on desktop
    # cv2.imshow('BodyPix', masked_image)
