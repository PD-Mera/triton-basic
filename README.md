# Basic Usage of Triton with Pytorch-ONNX Models

## Tritonserver

Create simple `.onnx` models with 

```bash 
pip install timm torch
python create_model.py
```

It will create a `mobilenetv3` onnx classification model name `output.onnx`, input is `BSx3x224x224` and output is `BSx1000`

Deploy models and config have structure

```
deploy_models
    | model_test
    |   | 1
    |   |   | model.onnx
    |   | config.pbtxt
```

Sample `config.pbtxt`

``` txt
name: "model_test"
platform: "onnxruntime_onnx"
max_batch_size: 4
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [1000]
  }
]
```

Start tritonserver docker

``` bash
docker pull nvcr.io/nvidia/tritonserver:22.12
docker run --gpus '"device=0,1"' -it --name triton_demo -p8187:8000 -p8188:8001 -p8189:8002 -v/.../triton-basic:/workspace/ --shm-size=16G nvcr.io/nvidia/tritonserver:22.12-py3
```

In tritonserver docker, run

``` bash
tritonserver --model-repository=/workspace/deploy_models/
```

When you see something like this, that mean your tritonserver is running properly

``` txt
I0328 02:32:49.029540 188 grpc_server.cc:4819] Started GRPCInferenceService at 0.0.0.0:8001
I0328 02:32:49.029872 188 http_server.cc:3477] Started HTTPService at 0.0.0.0:8000
I0328 02:32:49.071148 188 http_server.cc:184] Started Metrics Service at 0.0.0.0:8002
```

## Tritonclient

`parser_trion.py` is a sample parser for tritonclient

``` bash
pip install tritonclient[all]
python parser_triton.py
```
