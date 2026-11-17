#!/usr/bin/env bash
# Compare reference and test Nash-p models / 使用已有 WSL Conda 环境运行任务。
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
python_path="${BEVFUSION_PYTHON:-$HOME/miniconda3/envs/bevfusion-pytorch/bin/python}"
if [[ ! -x "$python_path" ]]; then
    printf 'Python missing: %s; set BEVFUSION_PYTHON\n' "$python_path" >&2
    exit 1
fi
export GLIBC_TUNABLES="${GLIBC_TUNABLES:+$GLIBC_TUNABLES:}glibc.rtld.execstack=2"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
cd "$project_dir"
exec "$python_path" -u "$project_dir/tools/compare_nash_p.py" "$@"
