#!/usr/bin/env bash
set -euo pipefail

#### PARAMETERS

# # GPT2
# MODEL="gtp2"
# PTAU_LOG=24
# FEATURES="fold,mock_prove"  # replace with "fold" for real proofs

# # Tiny GPT2
# MODEL="tiny_gtp2"
# PTAU_LOG=17
# FEATURES="fold,mock_prove"

# Tiny BERT
MODEL="bert_tiny"
PTAU_LOG=24
FEATURES="fold,mock_prove"

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
CONFIG="$CONFIGS_DIR/${MODEL}_config.yaml"


rustup override set nightly
mkdir -p "$PTAU_DIR/prover/"
export OMP_NUM_THREADS=$(nproc)

uv run "$MODEL_SCRIPT"

# Model-specific ONNX rewriters (authors' scripts under scratch/)
case "$MODEL" in
  bert_tiny|bert_base)
    # Fuse Reshape→Transpose (multi-head attention split) into ReshapeTrans.
    # The script hardcodes Bert.onnx → Bert_replaced.onnx; symlink + rename around it.
    pushd onnx > /dev/null
    ln -sf "${MODEL}.onnx" Bert.onnx
    uv run python ../zk-torch/zk-torch-repo/scratch/bert/replace_reshape_trans.py
    mv Bert_replaced.onnx "${MODEL}_rt.onnx"
    rm Bert.onnx
    popd > /dev/null
    ;;
esac

[ -f "$PTAU_DIR/challenge_$PTAU_LOG" ] || uv run "$SCRIPTS_DIR/generate_ptau.sh" "$PTAU_LOG"
cargo run --release --manifest-path "$MANIFEST" --bin zk_torch --features "$FEATURES" -- "$CONFIG"
