#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
selection="${1:-vnat}"
case "$selection" in
  vnat|cic-iscx) profile_args=(--dataset "$selection") ;;
  *) profile_args=(--config "$selection") ;;
esac
if [[ ! -x .venv/bin/packmorph-train || ! -x .venv/bin/packmorph-obfuscate || ! -x .venv/bin/packmorph-evaluate ]]; then
  if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv
  fi
  .venv/bin/pip install -e .
fi

./.venv/bin/packmorph-train all "${profile_args[@]}"

# Both datasets support baseline evaluation; packet defenses are separate commands.
./.venv/bin/packmorph-evaluate baseline "${profile_args[@]}"
