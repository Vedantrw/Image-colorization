from flask import Flask, request, render_template, send_from_directory
import os
import cv2
import torch
import uuid
from colorizers import eccv16, siggraph17
from demo_util import load_img, preprocess_img, postprocess_tens  # same utils from repo

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
use_gpu = torch.cuda.is_available()

if use_gpu:
    colorizer_eccv16.cuda()
    colorizer_siggraph17.cuda()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        img = load_img(filepath)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
        if use_gpu:
            tens_l_rs = tens_l_rs.cuda()

        out_eccv16 = colorizer_eccv16(tens_l_rs).cpu()
        out_siggraph17 = colorizer_siggraph17(tens_l_rs).cpu()

        out_img_eccv16 = postprocess_tens(tens_l_orig, out_eccv16)
        out_img_siggraph17 = postprocess_tens(tens_l_orig, out_siggraph17)

        # Save colorized images
        out1_path = os.path.join(OUTPUT_FOLDER, f"{filename}_eccv16.png")
        out2_path = os.path.join(OUTPUT_FOLDER, f"{filename}_siggraph17.png")
        cv2.imwrite(out1_path, cv2.cvtColor((out_img_eccv16 * 255).astype('uint8'), cv2.COLOR_RGB2BGR))
        cv2.imwrite(out2_path, cv2.cvtColor((out_img_siggraph17 * 255).astype('uint8'), cv2.COLOR_RGB2BGR))

        return render_template('index.html', original=filepath, eccv=out1_path, sigg=out2_path)

    return render_template('index.html', original=None)

if __name__ == '__main__':
    app.run(debug=True)
