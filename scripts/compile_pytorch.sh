#!/usr/bin/env bash
# Export local PyTorch through calibration to S600 HBIR/HBM / 本地浮点模型校准导出。
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
python_path="${BEVFUSION_PYTHON:-$HOME/miniconda3/envs/bevfusion-pytorch/bin/python}"
export GLIBC_TUNABLES="${GLIBC_TUNABLES:+$GLIBC_TUNABLES:}glibc.rtld.execstack=2"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
cd "$project_dir"
exec "$python_path" -u "$project_dir/tools/export_pytorch.py" "$@"
