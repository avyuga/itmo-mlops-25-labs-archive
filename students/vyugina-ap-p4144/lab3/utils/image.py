import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def preprocess_one_image(img_path):

    raw_image = Image.open(img_path).convert("RGB")

    input_image = np.array(raw_image.resize((640, 640)))
    input_image = input_image / 255.
    input_image = input_image.astype(np.float32).transpose((2, 0, 1))[None, ...] 

    return np.array(raw_image), input_image


def preprocess_all_images(directory: str) -> list:
    processed_images = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and os.path.splitext(file_path)[1] in [".png", ".jpeg", ",jpg"]:
            img = Image.open(file_path).convert('RGB')
            img_resized = img.resize((640, 640))
            img_array = np.array(img_resized)
            img_array = img_array.transpose(2, 0, 1)
            img_array = img_array / 255.0
            processed_images.append(img_array)
    return processed_images


def postprocess_detections(dets, init_h=640, init_w=640, confidence_threshold=0.7):

    dets_ = dets.copy()
    dets_ = dets_[dets_[:, 4] > confidence_threshold]
    dets_[:, 0] = dets_[:, 0] / 640 * init_w
    dets_[:, 2] = dets_[:, 2] / 640 * init_w

    dets_[:, 1] = dets_[:, 1] / 640 * init_h
    dets_[:, 3] = dets_[:, 3] / 640 * init_h

    return dets_


def visualize_result(raw_image, detections, coco_classes=None):
    draw = ImageDraw.Draw(raw_image)

    fontsize = 16
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", fontsize)

    for detection in detections:
        x0, y0, x1, y1 = detection[:4]
        score = detection[4]
        cls_id = detection[5]
        
        draw.rectangle(((x0, y0), (x1, y1)), outline="red", width=3)
        
        if coco_classes is not None:
            label_text = f"{coco_classes[str(int(cls_id)+1)]} [{score:.2f}]"
        else:
            label_text = f"{int(cls_id)} [{score:.2f}]"

        
        text_position = (x0, y0-fontsize) 
        draw.text(text_position, label_text, fill='red', font=font)
    
    return raw_image