import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


torch.manual_seed(42)
model = TinyMLP()
# Inference mode
model.eval()


torch.manual_seed(42)
dummy_input = torch.randn(2, 4)


with torch.no_grad():
    output = model(dummy_input)
    print("Output shape:", output.shape)
    print("Output: ", output)

torch.onnx.export(model, (dummy_input,), "onnx/tiny_mlp.onnx", dynamo=True)
print("Exported model.onnx")
