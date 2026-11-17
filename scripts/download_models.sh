#!/usr/bin/env bash
# Download official HAT assets directly with curl / 使用 curl 直接下载官方 HAT 模型。
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
if [[ -z "${OE_FTP_USER_INFO:-}" ]]; then
    oe_root="${OE_ROOT:-$project_dir/../toolchains/oe-package-3.7.0-s100-s600/drobotics_s100_s600_open_explorer_v3.7.0}"
    resolver="$oe_root/samples/ai_toolchain/model_zoo/runtime/nash-p/ai_benchmark/resolve_ai_benchmark_qat.sh"
    if [[ -f "$resolver" ]]; then
        OE_FTP_USER_INFO="$(sed -n "s/^[[:space:]]*export OE_FTP_USER_INFO='\([^']*\)'.*/\1/p" "$resolver")"
    fi
fi
: "${OE_FTP_USER_INFO:?Set OE_FTP_USER_INFO to the FTP user:password supplied with OE / 请设置 OE 配套 FTP 用户和密码}"
base_url="ftp://${OE_FTP_SERVER:-vrftp.horizon.ai}/horizon_torch_samples/3.1.19/py310/modelzoo/qat_origin_modelzoo/bevfusion_pointpillar_henet_multisensor_multitask_nuscenes"
mkdir -p "$project_dir/data/checkpoints"
cd "$project_dir/data/checkpoints"
for name in float-checkpoint-last.pth.tar qat.bc quantized.bc; do
    curl --fail --location --retry 3 --connect-timeout 30 --continue-at - \
        --user "$OE_FTP_USER_INFO" --output "$name" "$base_url/$name"
done
sha256sum float-checkpoint-last.pth.tar qat.bc quantized.bc > SHA256SUMS
printf 'Downloaded assets to %s\n' "$project_dir/data/checkpoints"
