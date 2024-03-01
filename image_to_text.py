import easyocr
import csv
import os
import numpy as np
import pandas as pd
import cv2
import re


def predict_number_plate(img, ocr):
    result = ocr.ocr(img, cls=True)
    result = result[0]
    texts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    if (scores[0]*100) >= 90:
        return re.sub(r'[^a-zA-Z0-9]', '', texts[0]), scores[0]
    else:
        return None, None
