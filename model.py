import torch
import cv2 as cv

from Cloned.mae.models_vit import vit_base_patch16


def img_to_tensor(img):
    img = cv.resize(img, (224, 224))
    img = torch.tensor(img, dtype=torch.float)
    img = (img - torch.mean(img, dim=[0, 1])) / torch.std(img, dim=[0, 1])
    img = torch.unsqueeze(img.permute([2, 0, 1]), 0)
    # img = torch.concat([img,img],0) #!!!
    return img


def get_model():
    model = vit_base_patch16()
    model_dict = torch.load(r"mae_pretrain_vit_base.pth")
    model.load_state_dict(model_dict['model'], strict=False)
    return model
