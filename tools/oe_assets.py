"""Fetch the matching official HAT BEVFusion assets / 下载配套官方 HAT 资源。"""
from __future__ import annotations

import argparse
import ftplib
import hashlib
import json
import os
from pathlib import Path
import re
import time
import subprocess

MODEL = "bevfusion_pointpillar_henet_multisensor_multitask_nuscenes"
ROOT = Path(__file__).resolve().parents[1]


def local_path(value):
    """Accept native Windows absolute paths in WSL entrypoints. / 兼容 Windows 绝对路径。"""
    text = str(value)
    if os.name != 'nt' and re.match(r'^[A-Za-z]:[\\/]', text):
        text = subprocess.check_output(['wslpath', '-a', '-u', text], text=True).strip()
    return Path(text).expanduser()


def oe_root(value=None):
    value = value or os.environ.get("OE_ROOT")
    if value:
        root = local_path(value).resolve()
    else:
        candidates = list((ROOT.parent / "toolchains").glob("oe-package-3.7.0-s100-s600/drobotics_s100_s600_open_explorer_v3.7.0"))
        if len(candidates) != 1:
            raise RuntimeError("Set OE_ROOT to the extracted OE 3.7.0 directory / 请设置 OE_ROOT")
        root = candidates[0]
    if not (root / "package/host/ai_toolchain/hat").is_dir():
        raise FileNotFoundError(f"HAT package missing under {root}")
    return root


def connect(root):
    source = root / "samples/ai_toolchain/model_zoo/runtime/nash-p/ai_benchmark/resolve_ai_benchmark_qat.sh"
    account = os.environ.get("OE_FTP_USER_INFO")
    if not account:
        match = re.search(r"OE_FTP_USER_INFO='([^']+)'", source.read_text())
        if not match:
            raise RuntimeError("Set OE_FTP_USER_INFO; no FTP account found in official resolve script")
        account = match[1]
    ftp = ftplib.FTP(os.environ.get("OE_FTP_SERVER", "vrftp.horizon.ai"), timeout=60)
    ftp.login(*account.split(":", 1))
    return ftp


def download(ftp, remote, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ftp.voidcmd("TYPE I")
    size = ftp.size(remote)
    if destination.is_file() and destination.stat().st_size == size:
        print(f"Using {destination.name} ({size:,} bytes)", flush=True)
        return
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise RuntimeError(f"Partial file exceeds remote size: {partial}")
    progress = [offset, time.monotonic()]
    with partial.open("ab") as output:
        def receive(block):
            output.write(block)
            progress[0] += len(block)
            if time.monotonic() - progress[1] > 10:
                print(f"{destination.name}: {progress[0]:,}/{size:,} bytes", flush=True)
                progress[1] = time.monotonic()
        if offset < size:
            ftp.retrbinary("RETR " + remote, receive, blocksize=1024 * 1024, rest=offset or None)
    if partial.stat().st_size != size:
        raise RuntimeError(f"Incomplete download: {partial}")
    partial.replace(destination)
    print(f"Downloaded {destination.name} ({size:,} bytes)", flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oe-root")
    parser.add_argument("--list", metavar="FTP_PATH")
    parser.add_argument("--remote", metavar="FTP_FILE")
    parser.add_argument("--output", type=local_path)
    parser.add_argument("--model-assets", action="store_true", help="Fetch QAT and quantized HBIR / 下载官方 HBIR")
    parser.add_argument("--float-checkpoint", action="store_true", help="Also fetch optional float checkpoint / 同时下载浮点权重")
    args = parser.parse_args()
    with connect(oe_root(args.oe_root)) as ftp:
        if args.model_assets:
            base = f'/horizon_torch_samples/3.1.19/py310/modelzoo/qat_origin_modelzoo/{MODEL}'
            names = ['qat.bc', 'quantized.bc']
            if args.float_checkpoint:
                names.append('float-checkpoint-last.pth.tar')
            manifest = []
            for name in names:
                output = ROOT/'data/checkpoints'/name
                download(ftp, f'{base}/{name}', output)
                manifest.append({'filename':name, 'source':f'ftp://vrftp.horizon.ai{base}/{name}',
                                 'bytes':output.stat().st_size, 'sha256':sha256(output)})
            (ROOT/'data/checkpoints/assets.json').write_text(json.dumps(manifest,indent=2))
            print(json.dumps(manifest,indent=2))
        elif args.list:
            ftp.retrlines("LIST " + args.list)
        elif args.remote and args.output:
            download(ftp, args.remote, args.output)
            print(json.dumps({"source": args.remote, "sha256": sha256(args.output)}))
        else:
            parser.error("Use --model-assets, --list, or --remote with --output")


if __name__ == "__main__":
    main()
