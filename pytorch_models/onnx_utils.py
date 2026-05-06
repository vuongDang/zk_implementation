from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper, numpy_helper
import torch


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


def get_paths(model_name):
    """Return (output_dir, onnx_path), creating directories as needed."""
    root = Path.cwd()
    output_dir = root / "zk-torch" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = root / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, str(onnx_dir / f"{model_name}.onnx")


def save_input_json(output_dir, model_name, dummy_input):
    with open(output_dir / f"{model_name}_input.json", "w") as f:
        json.dump({"input_data": dummy_input.numpy().tolist()}, f)


def export_onnx(model, dummy_input, onnx_path, opset_version=11):
    """Export to ONNX, load, print ops, and return the ONNX model."""
    torch.onnx.export(model, dummy_input, onnx_path, dynamo=False, opset_version=opset_version)
    m = onnx.load(onnx_path)
    print("Opset:", m.opset_import)
    print("Ops:", {n.op_type for n in m.graph.node})
    return m
