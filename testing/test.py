from pathlib import Path
from utils.grabscreen import grab_screen
import matplotlib.pyplot as plt
import cv2 as cv
import time
import numpy as np
from utils.custom_ocr import match, save_templates
from environment import _ocr_preproc, ScreenParser
import torch

scr = ScreenParser()
scr.calibrate('./calibr')

# db_dir = Path(r"C:\Users\Mykhailo_Tkachuk\PycharmProjects\Brawl-Stars-AI\ocr_data\end_title")
# db = save_templates(db_dir)
# for f in db_dir.glob('*'):
#     print(f.name, match(cv.imread(str(f))[..., [0]], db, verbose=0, score_thresh=1.2E-4))