"""Verify S600 HBM metadata and optional host simulation. / 核验元信息与可选主机模拟。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
from oe_assets import local_path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _variable(variable):
    return {"name": variable.name, "shape": list(variable.type.shape),
            "dtype": str(variable.type.np_dtype)}


def verify_hbm(model, inputs=None, output_dir="outputs/verification"):
    """Inspect HBM and optionally run local HBDK simulation, never remote BPU.

    核验 nash-p 架构和张量形状；提供输入时仅执行本地主机模拟，不连接开发板。
    Simulator output files and a JSON report are saved under output_dir.
    模拟输出及 JSON 检查报告写入指定输出目录；不修改或重新编译模型。
    """
    from hbdk4.compiler.hbm import Hbm

    model, output_dir = Path(model).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hbm = Hbm(str(model))
    report = {"model": str(model), "model_bytes": model.stat().st_size,
              "model_sha256": _sha256(model), "march": hbm.march_name,
              "compiler_toolkit_version": str(hbm.toolkit_version),
              "installed_compiler_version": importlib.metadata.version("hbdk4-compiler"),
              "graphs": [{"name": graph.name,
                          "inputs": [_variable(v) for v in graph.flatten_inputs],
                          "outputs": [_variable(v) for v in graph.flatten_outputs]}
                         for graph in hbm.graphs],
              "metadata_verified": hbm.march_name == "nash-p",
              "execution": {"status": "not_requested", "mode": "local_host_simulator",
                            "remote_bpu": False},
              "limitations": "Host simulation checks execution, not board performance or certified preprocessing. / 主机模拟不代表板端性能或原始预处理参数已核验。"}
    report_path = output_dir / "hbm_verification.json"
    failure = None
    try:
        if not report["metadata_verified"]:
            raise ValueError(f"Expected S600 nash-p, got {hbm.march_name}")
        if inputs is not None:
            if len(hbm.graphs) != 1:
                raise ValueError("Single-graph HBM required for this sample verifier")
            graph = hbm.graphs[0]
            feed = {}
            with np.load(inputs, allow_pickle=False) as data:
                for variable in graph.flatten_inputs:
                    if variable.name not in data:
                        raise ValueError(f"Missing input {variable.name}")
                    array = data[variable.name]
                    if tuple(array.shape) != tuple(variable.type.shape):
                        raise ValueError(f"Shape mismatch for {variable.name}: {array.shape}")
                    if array.dtype != variable.type.np_dtype:
                        raise ValueError(f"Dtype mismatch for {variable.name}: {array.dtype}")
                    if not np.isfinite(array).all():
                        raise ValueError(f"Non-finite input {variable.name}")
                    feed[variable.name] = np.ascontiguousarray(array)
            report["execution"].update(status="running", inputs=str(Path(inputs).resolve()))
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            simulator_dir = output_dir / "hbm_host"
            simulator_dir.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            # In HBDK4 4.7.5, remote_ip=None selects hbrt4-run-model-nash locally.
            # HBDK4 4.7.5 此参数选择本地主机模拟；此工具不提供远端连接参数。
            results = graph.feed(feed, output_dir=str(simulator_dir), remote_ip=None)
            report["execution"]["seconds"] = time.perf_counter() - start
            stats = {}
            for variable in graph.flatten_outputs:
                value = np.asarray(results[variable.name])
                shape_ok = tuple(value.shape) == tuple(variable.type.shape)
                dtype_ok = value.dtype == variable.type.np_dtype
                finite = bool(np.isfinite(value).all())
                stats[variable.name] = {
                    "shape": list(value.shape), "dtype": str(value.dtype),
                    "shape_matches": shape_ok, "dtype_matches": dtype_ok, "finite": finite,
                    "minimum": float(np.min(value)), "maximum": float(np.max(value)),
                    "mean": float(np.mean(value, dtype=np.float64)),
                }
                if not (shape_ok and dtype_ok and finite):
                    raise ValueError(f"Invalid output {variable.name}")
            report["execution"].update(status="passed", outputs=stats)
    except Exception as error:
        report["execution"].update(status="failed", error=f"{type(error).__name__}: {error}")
        failure = error
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"Report: {report_path}", flush=True)
    if failure is not None:
        raise failure
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=local_path, default="outputs/s600/bevfusion.hbm")
    parser.add_argument("--inputs", type=local_path, help="Existing NPZ inputs; omit for metadata only / 可选输入")
    parser.add_argument("--output-dir", type=local_path, default="outputs/verification")
    args = parser.parse_args()
    verify_hbm(args.model, args.inputs, args.output_dir)


if __name__ == "__main__":
    main()
