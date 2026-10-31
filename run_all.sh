#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
config_path="${1:-configs/default.yaml}"
if [[ ! -x .venv/bin/packmorph-train || ! -x .venv/bin/packmorph-obfuscate || ! -x .venv/bin/packmorph-evaluate ]]; then
  if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv
  fi
  .venv/bin/pip install -e .
fi

./.venv/bin/packmorph-train all --config "$config_path"

# Compare fixed policy outputs on exactly the rows in balanced test.pkl.
./.venv/bin/packmorph-evaluate all --config "$config_path"
