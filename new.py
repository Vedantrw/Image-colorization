import argparse
import cv2
import matplotlib.pyplot as plt
import torch
from colorizers import *


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img_path', type=str, required=True, help='Path to black and white image')
parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')
parser.add_argument('-o', '--save_prefix', type=str, default='output', help='Prefix for saved color images')
opt = parser.parse_args()


colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if opt.use_gpu:
    colorizer_eccv16.cuda()
    colorizer_siggraph17.cuda()


img = load_img(opt.img_path)  # from original repo
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
if opt.use_gpu:
    tens_l_rs = tens_l_rs.cuda()


out_eccv16 = colorizer_eccv16(tens_l_rs).cpu()
out_siggraph17 = colorizer_siggraph17(tens_l_rs).cpu()


img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
out_img_eccv16 = postprocess_tens(tens_l_orig, out_eccv16)
out_img_siggraph17 = postprocess_tens(tens_l_orig, out_siggraph17)

eccv_path = f'{opt.save_prefix}_eccv16.png'
sigg_path = f'{opt.save_prefix}_siggraph17.png'
cv2.imwrite(eccv_path, cv2.cvtColor((out_img_eccv16 * 255).astype('uint8'), cv2.COLOR_RGB2BGR))
cv2.imwrite(sigg_path, cv2.cvtColor((out_img_siggraph17 * 255).astype('uint8'), cv2.COLOR_RGB2BGR))

print(f"Colorized images saved as:\n - {eccv_path}\n - {sigg_path}")

# Display with OpenCV
import numpy as np

# Convert all images to BGR for OpenCV
img_bw_disp = cv2.cvtColor((img_bw * 255).astype('uint8'), cv2.COLOR_RGB2BGR)
eccv_disp = cv2.imread(eccv_path)
siggraph_disp = cv2.imread(sigg_path)

# Resize all to the same height (optional, to ensure alignment)
target_height = 256
img_bw_disp = cv2.resize(img_bw_disp, (int(img_bw_disp.shape[1] * target_height / img_bw_disp.shape[0]), target_height))
eccv_disp = cv2.resize(eccv_disp, (int(eccv_disp.shape[1] * target_height / eccv_disp.shape[0]), target_height))
siggraph_disp = cv2.resize(siggraph_disp, (int(siggraph_disp.shape[1] * target_height / siggraph_disp.shape[0]), target_height))

# Concatenate horizontally
combined = np.hstack((img_bw_disp, eccv_disp, siggraph_disp))

# Display all in one window
cv2.imshow('Colorization Results (L to R: Original BW, ECCV16, SIGGRAPH17)', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

