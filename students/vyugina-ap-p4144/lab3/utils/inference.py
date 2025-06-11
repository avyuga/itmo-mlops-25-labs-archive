import numpy as np

MAX_BATCH_SIZE = 8

def warmup_onnx(session, N_attempts=1):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    for _ in range(N_attempts):
        for batch_size in range(1, MAX_BATCH_SIZE+1):
            batch = np.random.uniform(low=0, high=1, size=(batch_size, 3, 640, 640))
            session.run([output_name], {input_name: batch.astype(np.float32)})[0]
    print("finished warming up ONNX Runtime...")


def warmup_triton(client):
    import tritonclient.http as httpclient

    for batch_size in range(1, MAX_BATCH_SIZE+1):
        batch = np.random.uniform(low=0, high=1, size=(batch_size, 3, 640, 640)).astype(np.float32)
        infer_input = httpclient.InferInput("images", batch.shape, datatype="FP32")
        infer_input.set_data_from_numpy(batch, binary_data=True)
        client.infer(model_name="yolov10s", inputs=[infer_input])

    print("finished warming up NVidia Triton Runtime...")