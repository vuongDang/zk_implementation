#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <power>" >&2
    exit 1
fi

POWER="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATED_DIR="$SCRIPT_DIR/../generated"

mkdir -p "$GENERATED_DIR"

npm install -g snarkjs

# snarkjs powersoftau new bn128 "$POWER" "$GENERATED_DIR/pot${POWER}_0000.ptau" -v
# snarkjs powersoftau contribute "$GENERATED_DIR/pot${POWER}_0000.ptau" "$GENERATED_DIR/pot${POWER}_0001.ptau" \
#     --name="contribution" -v -e="some random entropy"
# snarkjs powersoftau prepare phase2 "$GENERATED_DIR/pot${POWER}_0001.ptau" "$GENERATED_DIR/pot${POWER}_final.ptau" -v
snarkjs powersoftau export challenge "$GENERATED_DIR/pot${POWER}_final.ptau" "$GENERATED_DIR/challenge_${POWER}"
