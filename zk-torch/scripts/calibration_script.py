import torch
import numpy as np
from transformers import GPT2Model

def hook_fn(module, input, output):
    if isinstance(output, torch.Tensor):
        val = output.abs().max().item()
        activations.append(val)

model = GPT2Model.from_pretrained("gpt2")
model.eval()

# hook to capture all intermediate activations
activations = []

# register hooks on all layers
hooks = []
for name, module in model.named_modules():
    hooks.append(module.register_forward_hook(hook_fn))

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokens = tokenizer("hello world", return_tensors="pt")
dummy = tokens["input_ids"]  # real tokens, short sequence

# run forward pass
with torch.no_grad():
    #dummy = torch.zeros(1, 16, dtype=torch.long)  # 16 tokens
    model(dummy)

# remove hooks
for h in hooks:
    h.remove()

# compute required range
scale_factor_log = 12
max_activation = max(activations)
max_quantized = max_activation * (2 ** scale_factor_log)
cq_range_log = int(np.ceil(np.log2(max_quantized))) + 1  # +1 safety margin

print(f"scale_factor_log:.{scale_factor_log}")
print(f"Max activation value: {max_activation:.4f}")
print(f"Max quantized value: {max_quantized:.2f}")
print(f"Recommended cq_range_log: {cq_range_log}")
print(f"Recommended pow_len_log: {cq_range_log + 1}")
# then print sorted
for val in sorted(activations, reverse=True)[:10]:
    print(f"{val:.4f}")

