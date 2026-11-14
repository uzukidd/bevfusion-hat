#!/usr/bin/env bash
# Compile the architecture-independent QAT HBIR for S600 / 为 S600 编译 QAT HBIR。
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
python_path="${BEVFUSION_PYTHON:-$HOME/miniconda3/envs/bevfusion-pytorch/bin/python}"
export GLIBC_TUNABLES="${GLIBC_TUNABLES:+$GLIBC_TUNABLES:}glibc.rtld.execstack=2"
cd "$project_dir"
exec "$python_path" -u "$project_dir/tools/compile_s600.py" "$@"
