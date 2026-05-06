"""
Calibrate cq_range_log against an actual ONNX graph.

Unlike the original calibration_script.py (which hooks PyTorch module outputs
and misses SF^2 intermediates), this script:
  1. Loads the ONNX file directly.
  2. Exposes every intermediate tensor as a graph output.
  3. Runs onnxruntime to capture every tensor's runtime values.
  4. For ops whose raw output sits at SF^2 scale before ChangeSF
     (MatMul, Mul, Pow, Gemm), uses 2*SF in the cq computation.
     For everything else, uses SF.
  5. Reports the worst offenders and the required cq_range_log / pow_len_log.
"""
import sys
import argparse
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
from transformers import AutoTokenizer

SF2_OPS = {"MatMul", "Mul", "Pow", "Gemm"}


def find_gelu_internal_outputs(onnx_path):
    """Re-run the replace_gelu.py pattern match (without modifying the graph)
    and return the set of tensor names that would be eliminated by fusion."""
    m = onnx.load(onnx_path, load_external_data=False)
    nodes = list(m.graph.node)
    producers = {}
    for n in nodes:
        for o in n.output:
            producers[o] = n

    fused_outputs = set()
    for mul_node in nodes:
        if mul_node.op_type != "Mul":
            continue
        try:
            mul_node_1 = producers.get(mul_node.input[0])
            if mul_node_1 is None or mul_node_1.op_type != "Mul":
                continue
            add_node_0 = producers.get(mul_node.input[1])
            if add_node_0 is None or add_node_0.op_type != "Add":
                continue
            tanh_node = producers.get(add_node_0.input[0])
            if tanh_node is None or tanh_node.op_type != "Tanh":
                continue
            mul_node_2 = producers.get(tanh_node.input[0])
            if mul_node_2 is None or mul_node_2.op_type != "Mul":
                continue
            add_node_1 = producers.get(mul_node_2.input[0])
            if add_node_1 is None or add_node_1.op_type != "Add":
                continue
            mul_node_3 = producers.get(add_node_1.input[1])
            if mul_node_3 is None or mul_node_3.op_type != "Mul":
                continue
            pow_node = producers.get(mul_node_3.input[0])
            if pow_node is None or pow_node.op_type != "Pow":
                continue
            for n in (mul_node, mul_node_1, add_node_0, tanh_node,
                      mul_node_2, add_node_1, mul_node_3, pow_node):
                for o in n.output:
                    fused_outputs.add(o)
        except IndexError:
            continue
    return fused_outputs


def expose_all_intermediates(model_path, out_path):
    m = onnx.load(model_path, load_external_data=False)
    # Run shape inference so we can read the dtype of every intermediate tensor
    m = onnx.shape_inference.infer_shapes(m)
    existing_outputs = {o.name for o in m.graph.output}
    initializer_names = {i.name for i in m.graph.initializer}
    input_names = {i.name for i in m.graph.input}

    # value_info now contains type/shape for intermediates
    vi_by_name = {vi.name: vi for vi in m.graph.value_info}

    intermediate_names = []
    for node in m.graph.node:
        for o in node.output:
            if not o or o in existing_outputs or o in initializer_names or o in input_names:
                continue
            vi = vi_by_name.get(o)
            if vi is None or vi.type.tensor_type.elem_type == TensorProto.UNDEFINED:
                # No inferred type — skip (rare; usually shape/dynamic)
                continue
            m.graph.output.append(vi)
            intermediate_names.append(o)
    onnx.save(m, out_path)
    return intermediate_names


def producer_op_map(model_path):
    m = onnx.load(model_path, load_external_data=False)
    pmap = {}
    for node in m.graph.node:
        for o in node.output:
            if o:
                pmap[o] = (node.op_type, node.name)
    return pmap


def run_calibration(onnx_path, sf, prompt, seq_len, tokenizer_id="gpt2"):
    instrumented_path = onnx_path.replace(".onnx", "_allout.onnx")
    intermediate_names = expose_all_intermediates(onnx_path, instrumented_path)
    pmap = producer_op_map(onnx_path)

    sess = ort.InferenceSession(instrumented_path, providers=["CPUExecutionProvider"])
    input_meta = sess.get_inputs()
    print(f"Model inputs: {[(i.name, i.shape, i.type) for i in input_meta]}")

    # Build inputs
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    enc = tokenizer(prompt, return_tensors="np")
    ids = enc["input_ids"].astype(np.int64)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
    )
    if seq_len is not None:
        if ids.shape[1] < seq_len:
            pad = np.full((1, seq_len - ids.shape[1]), pad_id, dtype=np.int64)
            ids = np.concatenate([ids, pad], axis=1)
        else:
            ids = ids[:, :seq_len]

    feed = {}
    for inp in input_meta:
        if inp.name in ("input_ids",) or "int64" in inp.type:
            # Treat the first int64 input as token ids regardless of the name —
            # legacy ONNX exporter sometimes calls it onnx::Reshape_0 etc.
            feed[inp.name] = ids
        elif inp.name == "attention_mask":
            feed[inp.name] = np.ones_like(ids)
        elif inp.name == "position_ids":
            feed[inp.name] = np.arange(ids.shape[1], dtype=np.int64).reshape(1, -1)
        else:
            print(f"WARNING: unknown input {inp.name}, skipping")

    print(f"Running ORT with input_ids shape {ids.shape}...")
    output_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(output_names, feed)
    name_to_val = dict(zip(output_names, outs))
    print(f"Captured {len(name_to_val)} intermediate tensors.")

    # Per-op-type stats
    per_op_max = {}  # op_type -> (max_abs, tensor_name, node_name)
    per_op_required_log = {}  # op_type -> (max required cq_range_log, tensor_name, node_name)
    all_records = []  # (required_log, op_type, node_name, tensor_name, max_abs, scale_used)

    for name, val in name_to_val.items():
        if not isinstance(val, np.ndarray) or val.size == 0:
            continue
        if not np.issubdtype(val.dtype, np.floating):
            continue
        max_abs = float(np.max(np.abs(val)))
        if max_abs == 0 or not np.isfinite(max_abs):
            continue
        op_type, node_name = pmap.get(name, ("?", "?"))
        scale_log = (2 * sf) if op_type in SF2_OPS else sf
        max_quantized = max_abs * (1 << scale_log)
        required_log = int(np.ceil(np.log2(max_quantized))) + 1  # +1 safety margin
        all_records.append((required_log, op_type, node_name, name, max_abs, scale_log))

        prev = per_op_max.get(op_type, (-1.0, None, None))
        if max_abs > prev[0]:
            per_op_max[op_type] = (max_abs, name, node_name)
        prev_log = per_op_required_log.get(op_type, (-1, None, None))
        if required_log > prev_log[0]:
            per_op_required_log[op_type] = (required_log, name, node_name)

    return all_records, per_op_max, per_op_required_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_path")
    ap.add_argument("--sf", type=int, default=8, help="scale_factor_log")
    ap.add_argument("--prompt", default="hello world")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="optional fixed seq len (pad with eos)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--tokenizer", default="gpt2",
                    help="HF tokenizer id (e.g. gpt2, bert-base-uncased)")
    ap.add_argument("--simulate-gelu-fusion", action="store_true",
                    help="exclude tensors that would be absorbed by replace_gelu.py")
    args = ap.parse_args()

    all_records, per_op_max, per_op_required_log = run_calibration(
        args.onnx_path, args.sf, args.prompt, args.seq_len, args.tokenizer
    )

    fused = set()
    if args.simulate_gelu_fusion:
        fused = find_gelu_internal_outputs(args.onnx_path)
        print(f"\nFusion simulation: {len(fused)} tensors will be excluded "
              f"(absorbed into Gelu basic blocks).")

    def report(records, label):
        if not records:
            print(f"\n[{label}] no records.")
            return
        print(f"\n=== {label} (SF = {args.sf}) ===")
        print(f"\nTop {args.top} offenders by required cq_range_log:")
        print(f"  {'cq':>4}  {'op':<10} {'scale':>5}  {'max_abs':>14}  {'node_name'}")
        for rec in sorted(records, reverse=True)[: args.top]:
            rlog, op, nname, tname, mabs, scale = rec
            print(f"  {rlog:>4}  {op:<10} {scale:>5}  {mabs:>14.4g}  {nname}")

        per_op_log = {}
        for rec in records:
            rlog, op = rec[0], rec[1]
            if rlog > per_op_log.get(op, (-1, "", ""))[0]:
                per_op_log[op] = (rlog, rec[3], rec[2])
        print(f"\nPer-op worst-case required cq_range_log:")
        for op_type in sorted(per_op_log, key=lambda k: -per_op_log[k][0]):
            rlog, tname, nname = per_op_log[op_type]
            scale_log = (2 * args.sf) if op_type in SF2_OPS else args.sf
            print(f"  {op_type:<12} cq={rlog}  scale=2^{scale_log}  worst: {nname}")

        overall = max(r[0] for r in records)
        print(f"\n[{label}] Overall recommended cq_range_log: {overall}")
        print(f"[{label}] Overall recommended pow_len_log:  {overall + 1}")

    report(all_records, "BEFORE fusion (current ONNX)")
    if args.simulate_gelu_fusion:
        kept = [r for r in all_records if r[3] not in fused]
        report(kept, "AFTER GELU fusion")


if __name__ == "__main__":
    main()
