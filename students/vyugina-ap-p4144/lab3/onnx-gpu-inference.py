import random
import time

import numpy as np
import onnxruntime as ort

from utils.image import preprocess_all_images
from utils.inference import warmup_onnx, MAX_BATCH_SIZE


if __name__ == "__main__":
    session = ort.InferenceSession(
        "triton-model-repository/yolov10s/1/model.onnx", 
        providers=['CUDAExecutionProvider'],
        provider_options=[{"device_id": 0}]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    warmup_onnx(session)

    image_batch = preprocess_all_images("assets")
    batch = np.stack(image_batch, axis=0).astype(np.float32)


    times = {"onnx": []}

    for batch_size in range(1, MAX_BATCH_SIZE+1):
        random_idxs = random.sample(range(batch.shape[0]), batch_size)
        batch_sample = batch[random_idxs]

        t3 = time.time()
        session.run([output_name], {input_name: batch_sample.astype(np.float32)})[0]
        t4 = time.time()
        times["onnx"] += [(t4-t3)*1000]

        print(f"Batch size={batch_size}: " + \
            f"ONNX[GPU]: {times['onnx'][-1] / batch_size:.3f} ms/img")