#!/usr/bin/env bash
# Run OE helper tools in the existing environment / 使用已有环境运行 OE 辅助工具。
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
python_path="${BEVFUSION_PYTHON:-$HOME/miniconda3/envs/bevfusion-pytorch/bin/python}"
if [[ $# -lt 1 ]]; then
    printf 'Usage: %s {oe_assets|verify_hbir|verify_hbm|verify_float} [arguments...]\n' "$0" >&2
    exit 2
fi
tool="$1"
shift
case "$tool" in
    oe_assets|verify_hbir|verify_hbm|verify_float) ;;
    *) printf 'Unknown tool: %s\n' "$tool" >&2; exit 2 ;;
esac
export GLIBC_TUNABLES="${GLIBC_TUNABLES:+$GLIBC_TUNABLES:}glibc.rtld.execstack=2"
cd "$project_dir"
exec "$python_path" -u "$project_dir/tools/$tool.py" "$@"
