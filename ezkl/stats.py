import sys
import json
import csv
import time
import logging
import signal
from contextlib import contextmanager
from pathlib import Path
import os
from typing import Any
import itertools

PROOF_TIMEOUT_S = 600  # 10 minutes per config

import ezkl_pipeline as ezkl_pl

sys.path.append(str(Path(__file__).parent.parent / "pytorch_models"))

from transformer_block import create_onnx_model


@contextmanager
def timer(label, timings):
    log.info(f"Starting: {label}")
    start = time.perf_counter()
    yield
    time_passed = time.perf_counter() - start
    log.info(f"{label}: {time_passed:3f}s")
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

sweep = list(itertools.product(
    # [4],  # d_model
    [4, 8, 16, 32],  # d_model
    [1, 2, 4],   # n_heads
    # [16],    # d_fnn
    [16, 32, 64, 128],    # d_fnn
    [1],         # batch_size
    [1, 2]      # seq_len
))

# Init file
fieldnames = ["d_model", "n_heads", "d_ffn", "batch_size", "seq_len",
              "nb_parameters", "logrows", "circuit_compilation_time",
              "proof_generation_time", "proof_verification_time", "proof_size"]

timestamp = time.strftime("%Y%m%d_%H%M%S")
results_dir = Path("results").joinpath(timestamp)
results_dir.mkdir(exist_ok=True)
results_file = results_dir / f"results.csv"
logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(results_dir, f"run.log")),
        #logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)



with open(results_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

for params in sweep:
    d_model, n_heads, d_ffn, batch_size, seq_len = params
    if d_model % n_heads != 0 and d_model <= d_ffn:
        continue
    try:
        signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError(f"timed out after {PROOF_TIMEOUT_S}s")))
        signal.alarm(PROOF_TIMEOUT_S)
        stats = get_stats(d_model, n_heads, d_ffn, batch_size, seq_len)
        signal.alarm(0)
        with open(results_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(stats)
    except Exception as e:
        signal.alarm(0)
        log.warning(f"FAILED {params}: {e}")
