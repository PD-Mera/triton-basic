import tritonclient.grpc.aio
from tritonclient.utils import np_to_triton_dtype
import logging
import numpy as np

def main():
    MODEL_NAME = "test_model"
    URL = "172.16.10.240:5555"
    client = tritonclient.grpc.InferenceServerClient(URL)

    inputs_tensor = np.float32(np.random.randn(1, 3, 224, 224))

    inputs = [
        tritonclient.grpc.InferInput("input", inputs_tensor.shape, np_to_triton_dtype(np.float32)),
    ]
    inputs[0].set_data_from_numpy(inputs_tensor)
    outputs = [tritonclient.grpc.InferRequestedOutput("output")]

    res = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
    out = res.as_numpy("output")
    print(out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

