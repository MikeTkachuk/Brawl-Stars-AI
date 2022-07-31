from utils.grabscreen import grab_screen
import matplotlib.pyplot as plt
import cv2 as cv
import time
import pytesseract
import numpy as np

import torch
from Cloned.mae.models_vit import vit_base_patch16


def img_to_tensor(img):
    img = torch.tensor(img, dtype=torch.float)
    img = (img-torch.mean(img,dim=[0,1]))/torch.std(img,dim=[0,1])
    img = torch.unsqueeze(img.permute([2,0,1]),0)
    #img = torch.concat([img,img],0) #!!!
    return img


vit_dict = torch.load(r"mae_pretrain_vit_base.pth")
vit = vit_base_patch16()
print(vit.load_state_dict(vit_dict['model'], strict=False))

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
while True:
    start = time.time()
    region = grab_screen([60,35,400,170])
    cv.imwrite('../imt.png', region)
    region1 = cv.resize(region, (224,224))
    print(vit.forward_features(img_to_tensor(region1)))
    score = grab_screen([120, 170, 200, 230])
    region = cv.cvtColor(region, cv.COLOR_RGB2GRAY)
    score = cv.cvtColor(score, cv.COLOR_RGB2GRAY)
    _, region = cv.threshold(region, 150, 255, cv.THRESH_BINARY)
    region = cv.erode(region, np.ones((5,5),np.uint8))
    _, score = cv.threshold(score, 150, 255, cv.THRESH_BINARY)
    score = cv.erode(score, np.ones((3,3),np.uint8))
    region = cv.cvtColor(np.expand_dims(region,-1), cv.COLOR_GRAY2RGB)
    score = cv.cvtColor(np.expand_dims(score,-1), cv.COLOR_GRAY2RGB)

    score = cv.resize(score, None, fx=3., fy=3.)

    print(pytesseract.image_to_string(region, config='--psm 8'))
    print(pytesseract.image_to_string(score, config='-c tessedit_char_whitelist=0123456789+- --psm 7'))


    print(f'FPS: {1/(time.time()-start)}')
    cv.imwrite('../img1.png', region)
    cv.imwrite('../score1.png', score)
