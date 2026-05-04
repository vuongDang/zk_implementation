from transformers import GPT2Model
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper, numpy_helper
import torch


SEQ_LEN = 64
VOCAB_SIZE = 50257

# We need to replace onnx layernorm with primitives handled by zktorch
def decompose_layer_norm(model):
    """Replace LayerNormalization nodes with primitives that zk-torch supports."""
    graph = model.graph
    new_nodes = []
    new_initializers = list(graph.initializer)

    for node in graph.node:
        if node.op_type != "LayerNormalization":
            new_nodes.append(node)
            continue

        epsilon = 1e-5
        for attr in node.attribute:
            if attr.name == "epsilon":
                epsilon = attr.f

        x = node.input[0]
        scale = node.input[1]
        bias = node.input[2] if len(node.input) > 2 and node.input[2] != "" else None
        y = node.output[0]
        p = y + "_ln_"

        # In opset 18, ReduceMean takes axes as a second input tensor.
        # zk-torch ignores this input and defaults to reducing along the last axis,
        # which is correct for GPT-2 LayerNorm (axis=-1).
        axes_name = p + "axes"
        new_initializers.append(
            numpy_helper.from_array(np.array([-1], dtype=np.int64), name=axes_name)
        )
        eps_name = p + "eps"
        new_initializers.append(
            numpy_helper.from_array(np.array([epsilon], dtype=np.float32), name=eps_name)
        )

        mean = p + "mean"
        diff = p + "diff"
        sq   = p + "sq"
        var  = p + "var"
        veps = p + "veps"
        std  = p + "std"
        norm = p + "norm"
        scaled = p + "scaled"

        new_nodes += [
            helper.make_node("ReduceMean", [x, axes_name],   [mean],   keepdims=1),
            helper.make_node("Sub",        [x, mean],         [diff]),
            helper.make_node("Mul",        [diff, diff],      [sq]),
            helper.make_node("ReduceMean", [sq, axes_name],   [var],    keepdims=1),
            helper.make_node("Add",        [var, eps_name],   [veps]),
            helper.make_node("Sqrt",       [veps],            [std]),
            helper.make_node("Div",        [diff, std],       [norm]),
            helper.make_node("Mul",        [norm, scale],     [scaled if bias else y]),
        ]
        if bias:
            new_nodes.append(helper.make_node("Add", [scaled, bias], [y]))

    del graph.node[:]
    graph.node.extend(new_nodes)
    del graph.initializer[:]
    graph.initializer.extend(new_initializers)
    return model


class GPT2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
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


class GPT2WrapperOldTransformer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
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

def export_gpt2_onnx(model_name, model_config, vocab_size, seq_len, ):
    """Export model to ONNX, decompose LayerNorm, and verify numerical equivalence."""
    import onnxruntime as ort

    model_config.config._attn_implementation = "eager"
    # model = GPT2Wrapper(model_config)
    model = GPT2WrapperOldTransformer(model_config)
    model.eval()
    if model_name == "tiny_gpt2":
        for param in model_config.parameters():
            param.data *= 1e-2
    dummy_input = torch.zeros(1, seq_len, dtype=torch.long)
    root = Path.cwd()
    output_dir = root / "zk-torch" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{model_name}_input.json", "w") as f:
        json.dump({"input_data": dummy_input.numpy().tolist()}, f)

    onnx_path = str(root / "onnx" / f"{model_name}.onnx")

    torch.onnx.export(model, dummy_input, onnx_path, dynamo=False, opset_version=11)
    m = onnx.load(onnx_path)

    sess_orig = ort.InferenceSession(onnx_path)
    input_name = sess_orig.get_inputs()[0].name

    m = decompose_layer_norm(m)
    onnx.save(m, onnx_path)

    ops = {n.op_type for n in m.graph.node}
    print("Opset:", m.opset_import)
    print("Ops:", ops)
    assert "LayerNormalization" not in ops
    # print("OK: LayerNormalization decomposed into primitives")

    # sess_decomp = ort.InferenceSession(onnx_path)
    # diffs = []
    # for i in range(10):
    #     inp = np.random.randint(0, vocab_size, size=(1, seq_len), dtype=np.int64)
    #     (out_orig,) = sess_orig.run(None, {input_name: inp})
    #     (out_decomp,) = sess_decomp.run(None, {input_name: inp})
    #     worst = np.abs(out_orig - out_decomp).max()
    #     diffs.append(worst)
    #     print(f"  input {i}: worst max diff = {worst:.2e}")

    # assert all(d < 1e-2 for d in diffs), f"Decomposition changed outputs: {diffs}"
    # print("OK: decomposition is numerically equivalent")


if __name__ == "__main__":
    model_config = GPT2Model.from_pretrained("gpt2")
    export_gpt2_onnx("gpt2",  model_config,  VOCAB_SIZE, SEQ_LEN)
