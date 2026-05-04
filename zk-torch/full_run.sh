# rustup override set nightly
mkdir -p zk-torch/generated/prover/

# uv run pytorch_models/tiny_mlp.py
# uv run zk-torch/scripts/generate_ptau.sh 7
# cargo run --release --manifest-path zk-torch/zk-torch-repo/Cargo.toml --bin zk_torch --features fold -- zk-torch/configs/tiny_mlp.yaml

# uv run pytorch_models/gpt2_old_transformer.py
# uv run zk-torch/scripts/generate_ptau.sh 24
# cargo run --manifest-path zk-torch/zk-torch-repo/Cargo.toml --bin zk_torch --features fold -- zk-torch/gpt2.yaml

uv run pytorch_models/tiny_gpt2.py
[ -f zk-torch/generated/challenge_17 ] || uv run zk-torch/scripts/generate_ptau.sh 17
cargo run --release --manifest-path zk-torch/zk-torch-repo/Cargo.toml --bin zk_torch --features "fold,mock_prove" -- zk-torch/configs/tiny_gpt2_config.yaml
