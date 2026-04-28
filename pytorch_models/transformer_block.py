import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


def attention(queries, keys, values):
    d = queries.shape[-1]
    # scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(d)
    scale = torch.full((1,), 1.0 / math.sqrt(d))
    scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale
    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, values)


# Custom multihead for ezkl compatibility
class ManualMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim, self.num_heads = embed_dim, num_heads
        assert embed_dim % num_heads == 0
        self.projection_dim = embed_dim // num_heads

        self.W_q = nn.ModuleList([nn.Linear(embed_dim, self.projection_dim)  for _ in range(num_heads)])
        self.W_k = nn.ModuleList([nn.Linear(embed_dim, self.projection_dim)  for _ in range(num_heads)])
        self.W_v = nn.ModuleList([nn.Linear(embed_dim, self.projection_dim)  for _ in range(num_heads)])
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        heads = [attention(self.W_q[i](x), self.W_k[i](x), self.W_v[i](x)) for i in range(self.num_heads)]
        return self.W_o(torch.cat(heads, dim=-1))


class ManualTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.att = ManualMultiHeadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.layernorm1(x + self.att(x))
        x = self.layernorm2(x + self.ffn(x))
        return x



def create_onnx_model(d_model, nheads, d_ffn, batch_size, seq_len) -> tuple[str, nn.Module]:
    layer = ManualTransformerBlock(
        d_model,
        nheads,
        d_ffn,
    )

    layer.eval()

    torch.manual_seed(42)
    dummy_input = torch.randn(batch_size, seq_len, d_model)

    model_name = f"transformer_block_dmod_{d_model}_nheads_{nheads}_dffn_{d_ffn}_batch_{batch_size}_seq_{seq_len}"
    fname = f"onnx/{model_name}.onnx"
    with torch.no_grad():
        torch.onnx.export(
            layer, (dummy_input,), fname, dynamic_axes=None, opset_version=13, dynamo=False
        )
        print(f"Exported {model_name}.onnx")
    return (model_name, layer)


# Test our custom transfomer
def test_transformer_block():
    embed_dim, n_heads, d_ffn = 16, 4, 32

    ref = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=n_heads,
        dim_feedforward=d_ffn,
        dropout=0.0,
        batch_first=True,
    )
    ref.eval()

    custom = ManualTransformerBlock(embed_dim, n_heads, d_ffn)
    custom.eval()

    q_w, k_w, v_w = ref.self_attn.in_proj_weight.chunk(3, dim=0)
    q_b, k_b, v_b = ref.self_attn.in_proj_bias.chunk(3, dim=0)
    with torch.no_grad():
        for i in range(n_heads):
            start = i * custom.att.projection_dim
            end = (i+1) * custom.att.projection_dim
            custom.att.W_q[i].weight.copy_(q_w[start:end])
            custom.att.W_q[i].bias.copy_(q_b[start:end])
            custom.att.W_k[i].weight.copy_(k_w[start:end])
            custom.att.W_k[i].bias.copy_(k_b[start:end])
            custom.att.W_v[i].weight.copy_(v_w[start:end])
            custom.att.W_v[i].bias.copy_(v_b[start:end])

        custom.att.W_o.weight.copy_(ref.self_attn.out_proj.weight)
        custom.att.W_o.bias.copy_(ref.self_attn.out_proj.bias)
        # FFN weights
        custom.ffn[0].weight.copy_(ref.linear1.weight)
        custom.ffn[0].bias.copy_(ref.linear1.bias)
        custom.ffn[2].weight.copy_(ref.linear2.weight)
        custom.ffn[2].bias.copy_(ref.linear2.bias)

    for _ in range(10):
        x = torch.randn(1, 1, embed_dim)
        with torch.no_grad():
            ref_out = ref(x)
            custom_out = custom(x)

        print(f"Max diff: {(ref_out - custom_out).abs().max().item()}")
        assert torch.allclose(ref_out, custom_out, atol=1e-5), "Outputs don't match!"
    print("Test passed!")


# test_transformer_block()
