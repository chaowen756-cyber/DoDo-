#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible entry point.  The PSF pipeline now runs on physical GPUs
# 2 and 3; keep the historical filename so previously copied commands continue
# to launch the corrected Number5 script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_joint_training_gpu23.sh" "$@"
