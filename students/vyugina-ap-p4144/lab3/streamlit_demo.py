import json

import numpy as np
import pandas as pd
import streamlit as st
import tritonclient.http as httpclient
from PIL import Image

from utils.image import postprocess_detections, visualize_result


client = httpclient.InferenceServerClient(url="localhost:9000")

with open("assets/coco_classes.json", 'r') as f:
    coco_classes = json.load(f)

st.set_page_config(layout="wide")


def infere_image(raw_image):
    input_image = np.array(raw_image.resize((640, 640), Image.Resampling.LANCZOS)) / 255.
    input_image = input_image.transpose((2, 0, 1)).astype(np.float32)[None, ...] 
    
    infer_input = httpclient.InferInput("images", input_image.shape, datatype="FP32")
    infer_input.set_data_from_numpy(input_image, binary_data=True)

    responce = client.infer(model_name="yolov10s", inputs=[infer_input])
    output = responce.as_numpy('output0')[0]

    init_w, init_h = raw_image.size
    output = postprocess_detections(output, init_h=init_h, init_w=init_w, confidence_threshold=0.7)

    visualized_image = visualize_result(raw_image, output, coco_classes=coco_classes)

    table = pd.DataFrame(output, columns=["x0", "y0", "x1", "y1", "score", "class"])
    table[["x0", "y0", "x1", "y1"]] = table[["x0", "y0", "x1", "y1"]].astype(np.uint32)
    table["class"] = table["class"].apply(lambda x: coco_classes[str(int(x+1))])

    return visualized_image, table


def main():
    st.title("Nvidia Triton Inference Server")
    
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file).convert("RGB")
        processed_image, result_table = infere_image(raw_image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Result Visualization")
            st.image(processed_image, caption="Processed Image", use_container_width=True)
        
        with col2:
            st.subheader("Found Objects")
            st.dataframe(result_table, hide_index=True)

if __name__ == "__main__":
    main()
