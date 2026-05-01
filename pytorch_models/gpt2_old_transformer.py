# With old versions of transformers
#    "onnx>=1.15.0",
#    "torch>=2.1.0",
#    "transformers==4.37.0",

import torch
import torch.nn as nn
from transformers import GPT2Model
from pathlib import Path
import onnx
import json

class GPT2Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = GPT2Model.from_pretrained("gpt2")
        # disable caching and new masking logic
        self.model.config.use_cache = False
        self.model.config._attn_implementation = "eager"

    def forward(self, input_ids):
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                use_cache=False,
            )
        return out.last_hidden_state

model = GPT2Wrapper()
model.eval()
dummy_input = torch.zeros(1, 64, dtype=torch.long)
root = Path.cwd()

input_data = {"input_data": dummy_input.numpy().tolist()}
output_dir = root / "zk-torch" / "generated"

output_dir.parent.mkdir(parents=True, exist_ok=True)
with open(output_dir / "gpt2_input.json", "w") as f:
    json.dump(input_data, f)

onnx_path = str(root / "onnx" / "gpt2.onnx")
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=13,
    dynamo=False,
    input_names=["input_ids"],
    output_names=["last_hidden_state"],
)

m = onnx.load(onnx_path)
ops = {n.op_type for n in m.graph.node}
print("Opset:", m.opset_import)
print("Ops:", ops)
assert "LayerNormalization" not in ops
