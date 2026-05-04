import torch
import torch.nn as nn
import json
from pathlib import Path

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(16, 4) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(16))
        self.w2 = nn.Parameter(torch.randn(2, 16) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        x = torch.matmul(x, self.w1.t()) + self.b1
        x = torch.relu(x)
        x = torch.matmul(x, self.w2.t()) + self.b2
        return x

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(16, 4) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(16))
        self.w2 = nn.Parameter(torch.randn(2, 16) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(2))
        # self.net = nn.Sequential(
        #     nn.Linear(4, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 2),
        # )

    def forward(self, x):
        x = torch.matmul(x, self.w1.t()) + self.b1
        x = torch.relu(x)
        x = torch.matmul(x, self.w2.t()) + self.b2
        return x
        # return self.net(x)


torch.manual_seed(42)
model = TinyMLP()
model.eval()

dummy_input = torch.randn(1, 4)


root = Path.cwd()
input_data = {"input_data": dummy_input.numpy().tolist()}
output_dir = root / "zk-torch" / "generated"
output_dir.parent.mkdir(parents=True, exist_ok=True)
with open(output_dir / "tiny_mlp_input.json", "w") as f:
    json.dump(input_data, f)

# with torch.no_grad():
#     output = model(dummy_input)
#     print("Output shape:", output.shape)
#     print("Output: ", output)

torch.onnx.export(
    model, (dummy_input,), "onnx/tiny_mlp.onnx", dynamo=False, opset_version=11
)
print("Exported model.onnx")
