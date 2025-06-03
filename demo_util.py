import numpy as np
import skimage.color as color
import torch
from PIL import Image
from skimage.transform import resize


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = np.asarray(img) / 255.0
    return img

def preprocess_img(img_rgb, HW=(256, 256)):
    img_lab = color.rgb2lab(img_rgb)
    img_l = img_lab[:, :, 0]
    tens_l_orig = torch.Tensor(img_l)[None, None, :, :]
    img_rs = np.array(Image.fromarray(img_l).resize(HW[::-1], resample=Image.BICUBIC))
    tens_l_rs = torch.Tensor(img_rs)[None, None, :, :].float()
    tens_l_rs = (tens_l_rs - 50.) / 50.
    return tens_l_orig, tens_l_rs

def postprocess_tens(tens_l_orig, out_ab):
    img_l = tens_l_orig[0, 0].numpy()
    out_ab_np = out_ab[0].detach().numpy()
    out_lab = np.zeros((img_l.shape[0], img_l.shape[1], 3))
    out_lab[:, :, 0] = img_l
    out_ab_resized = resize(out_ab_np.transpose((1, 2, 0)), img_l.shape, order=1, mode='reflect', anti_aliasing=True)
    out_lab[:, :, 1:] = out_ab_resized
    img_rgb = color.lab2rgb(out_lab)
    img_rgb = np.clip(img_rgb, 0, 1)
    return img_rgb
