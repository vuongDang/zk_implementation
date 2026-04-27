import sys
import json
import time
from contextlib import contextmanager
from pathlib import Path
import os
from typing import Any

import ezkl_pipeline as ezkl_pl

sys.path.append(str(Path(__file__).parent.parent / "pytorch_models"))

from transformer_block import create_onnx_model


@contextmanager
def timer(label, timings):
    start = time.perf_counter()
    yield
    time_passed = time.perf_counter() - start
    print(f"{label}: {time_passed:3f}s")
    timings[label.lower().replace(" ", "_")] = time_passed


def get_stats(d_model, n_heads, d_ffn, batch_size, seq_len) -> dict[str, Any]:
    onnx_file, model = create_onnx_model(d_model, n_heads, d_ffn, batch_size, seq_len)
    ezkl_pl.setup(onnx_file, d_model, batch_size, seq_len)

    stats = {
          "d_model": d_model, "n_heads": n_heads, "d_ffn": d_ffn,
          "batch_size": batch_size, "seq_len": seq_len,
      }
    stats["nb_parameters"] = sum(p.numel() for p in model.parameters())
    with open(ezkl_pl.SETTINGS_FILE) as f:
        stats["logrows"] = json.load(f)["run_args"]["logrows"]

    with timer("Circuit compilation time", stats):
        ezkl_pl.compile_circuit(reuse=False)

    with timer("Proof generation time", stats):
        proof = ezkl_pl.gen_proof(d_model, batch_size, seq_len, reuse=False)
        stats["proof_size"] = os.path.getsize(proof)

    with timer("Proof verification time", stats):
        res, commit_hash = ezkl_pl.verif_proof()
        print(f"Verification result: {res}")
        print(f"Commitment hash: {commit_hash}")

    return stats


stats = get_stats(d_model=4, n_heads=1, d_ffn=16, batch_size=1, seq_len=1)
print(stats)
