import torch
from torch import nn
import timm

class NewModel(nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()
        self.backbone = timm.create_model("mobilenetv3_small_050.lamb_in1k", pretrained = True)
        self.head = nn.Linear(1000, 2)


    def forward(self, x):
        return self.head(self.backbone(x))

def main():
    model = NewModel()
    dummy_input = torch.randn(4, 3, 224, 224)

    out = model(dummy_input)
    print(out.size())

    torch.onnx.export(model,               # model being run
                        dummy_input,                         # model input (or a tuple for multiple inputs)
                        "output.onnx",   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=11,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

if __name__ == "__main__":
    main()