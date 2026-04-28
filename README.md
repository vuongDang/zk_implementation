# ZK Proof Generation for ML Models

Zero-knowledge proof system for neural network inference using [EZKL](https://github.com/zkonduit/ezkl). Proves correct execution of model inference without revealing inputs, using Halo2 with KZG commitments.

## Overview

Currently supports:
- **Transformer blocks** — custom multi-head attention implementation compatible with EZKL's ONNX constraints
- **Tiny MLP** — simple 2-layer perceptron for baseline testing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Run benchmarks

Sweeps transformer block configurations and records proof generation metrics to `results/`:

```bash
cd ezkl
python stats.py
```

## Project Structure

```
zk_implementation/
├── ezkl/
│   ├── ezkl_pipeline.py      # Core ZK workflow (calibrate → compile → prove → verify)
│   ├── stats.py              # Benchmark runner
│   └── generated/            # Circuit artifacts: SRS, keys, witnesses, proofs
├── pytorch_models/
│   ├── transformer_block.py  # EZKL-compatible transformer (no TransformerEncoderLayer)
│   └── tiny_mlp.py           # Simple MLP baseline
├── onnx/
│   ├── check_onnx.py         # Validate ONNX model structure
│   └── exec_onnx.py          # Run inference via onnxruntime
├── results/                  # CSV benchmark outputs
└── notes.md                  # EZKL workflow notes and known constraints
```

## Workflow

1. Define PyTorch model and export to ONNX (opset ≤ 16)
2. Generate calibration data for quantization
3. Calibrate circuit settings and download SRS
4. Compile circuit and generate proving/verification keys
5. Run inference on test input to produce witness
6. Generate ZK proof and verify — outputs validity + model commitment hash

## Requirements

- Python 3.x
- ezkl 23.0.5
- PyTorch 2.11.0
- onnx 1.21.0 / onnxruntime 1.25.0
- CUDA (optional, for GPU acceleration)
