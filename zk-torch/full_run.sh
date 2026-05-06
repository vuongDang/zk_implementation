#!/usr/bin/env bash
set -euo pipefail

#### PARAMETERS

# GPT2
MODEL="gtp2"
PTAU_LOG=24
FEATURES="fold,mock_prove"  # replace with "fold" for real proofs

# # Tiny GPT2
# MODEL="tiny_gtp2"
# PTAU_LOG=17
# FEATURES="fold,mock_prove"

# # Tiny MLP
# MODEL="tiny_mlp"
# PTAU_LOG=7
# FEATURES="fold"


#### SCRIPT

MANIFEST="zk-torch/zk-torch-repo/Cargo.toml"
PTAU_DIR="zk-torch/generated"
CONFIGS_DIR="zk-torch/configs"
SCRIPTS_DIR="zk-torch/scripts"
MODEL_SCRIPT="pytorch_models/$MODEL.py"
CONFIG="$CONFIGS_DIR/$MODEL_config.yaml"


rustup override set nightly
mkdir -p "$PTAU_DIR/prover/"
export OMP_NUM_THREADS=$(nproc)

uv run "$MODEL_SCRIPT"
[ -f "$PTAU_DIR/challenge_$PTAU_LOG" ] || uv run "$SCRIPTS_DIR/generate_ptau.sh" "$PTAU_LOG"
cargo run --release --manifest-path "$MANIFEST" --bin zk_torch --features "$FEATURES" -- "$CONFIG"
