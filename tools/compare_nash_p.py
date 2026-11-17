"""Compare two nash-p BEVFusion 7-input/5-output models on prepared NPZ samples.

Chinese / 中文说明见下一段。 Supports HBIR .bc and HBM .hbm. HBM uses
Graph.feed(remote_ip=None): slow local host simulation, never board execution.
Metrics describe differences from a reference model, not ground-truth accuracy.
Outputs default to graph order; optional names specify the five semantic outputs
in this order: BEV embedding, class logits, reference points, box coordinates,
occupancy logits. Shapes are strictly checked, without guessing by shape.

在同一组预处理 NPZ 样本上比较任意两个 nash-p BEVFusion 七输入、五输出模型。
支持 .bc HBIR 和 .hbm HBM；HBM 使用 remote_ip=None 的慢速本地主机模拟，
并非板端测试。检测与占用统计仅表示相对于参考模型的差异，不代表真实 GT 精度。
输出默认按图顺序；可指定名称依次重排为 BEV 特征、类别、参考点、框坐标、
occupancy logits，严格检查形状，不通过形状猜测语义。保存逐样本与汇总 JSON/CSV；
不设置差异通过阈值。相同形状的类别与框坐标输出必须由调用者保证语义顺序正确。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from oe_assets import ROOT, local_path, oe_root, sha256

SEMANTICS = ("bev_embed", "class_logits", "reference_points", "box_coordinates", "occupancy_logits")
EXPECTED_SHAPES = ((1, 16384, 256), (1, 900, 10), (1, 900, 3),
                   (1, 900, 10), (1, 200, 200, 16, 18))
INPUT_NAMES = {"features", "coors", "img", "queries_rebatch_grid", "restore_bev_grid",
               "reference_points_rebatch", "bev_pillar_counts"}
OCCUPANCY_CLASSES = ("others", "barrier", "bicycle", "bus", "car", "construction_vehicle",
                     "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
                     "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade",
                     "vegetation", "free")
FREE_CLASS = 17


def variable_info(variable):
    """Read graph tensor contracts. / 读取图张量的名称、形状和类型契约。"""
    return {"name": variable.name, "shape": list(variable.type.shape),
            "dtype": str(np.dtype(variable.type.np_dtype))}


class Model:
    """Keep the owning graph alive and copy outputs. / 保持图存活并复制运行输出。"""

    def __init__(self, path, output_names, expected_target="nash-p"):
        if expected_target not in ("nash-p", "nash-m"):
            raise ValueError(f"Unsupported expected target {expected_target!r}")
        self.path = Path(path).resolve()
        suffix = self.path.suffix.lower()
        if suffix == ".bc":
            from hbdk4 import compiler
            self.owner = compiler.load(str(self.path))
            attributes = {item.name: str(item.attr)
                          for item in self.owner.module.operation.attributes}
            # Read only the module root attribute; never infer target from filenames.
            # 仅读取模块根属性，不根据文件名推断目标平台。
            target = attributes.get("hbdk.target")
            if target is None or target.strip('"') != expected_target:
                raise ValueError(f"{self.path}: root hbdk.target must be {expected_target}, got {target!r}")
            if len(self.owner.functions) != 1:
                raise ValueError(f"{self.path}: exactly one HBIR function is required")
            self.graph = self.owner.functions[0]
            self.inputs = list(self.graph.inputs)
            self.outputs = list(self.graph.outputs)
            backend = "local_hbir_interpreter"
        elif suffix == ".hbm":
            from hbdk4.compiler.hbm import Hbm
            self.owner = Hbm(str(self.path))
            target = self.owner.march_name
            if target != expected_target:
                raise ValueError(f"{self.path}: HBM march_name must be {expected_target}, got {target!r}")
            if len(self.owner.graphs) != 1:
                raise ValueError(f"{self.path}: exactly one HBM graph is required")
            self.graph = self.owner.graphs[0]
            self.inputs = list(self.graph.flatten_inputs)
            self.outputs = list(self.graph.flatten_outputs)
            backend = "local_hbm_host_simulator"
            print(f"{self.path.name}: HBM local host simulation may be very slow; "
                  "remote_ip=None, no board test / HBM 本地模拟可能很慢，未板测", flush=True)
        else:
            raise ValueError(f"Expected .bc or .hbm model, got {self.path}")
        input_names = [item.name for item in self.inputs]
        if len(input_names) != 7 or set(input_names) != INPUT_NAMES:
            raise ValueError(f"{self.path}: expected seven semantic input names {sorted(INPUT_NAMES)}, "
                             f"got {input_names}")
        graph_names = [item.name for item in self.outputs]
        if len(graph_names) != 5 or len(set(graph_names)) != 5:
            raise ValueError(f"{self.path}: expected five uniquely named raw outputs, got {graph_names}")
        self.names = list(output_names) if output_names is not None else graph_names
        if len(self.names) != 5 or len(set(self.names)) != 5 or set(self.names) != set(graph_names):
            raise ValueError(f"{self.path}: output-name override must be a permutation of {graph_names}")
        by_name = {item.name: item for item in self.outputs}
        self.ordered_outputs = [by_name[name] for name in self.names]
        for semantic, variable, expected in zip(SEMANTICS, self.ordered_outputs, EXPECTED_SHAPES):
            if tuple(variable.type.shape) != expected:
                raise ValueError(f"{self.path}: {semantic} ({variable.name}) shape "
                                 f"{tuple(variable.type.shape)} != expected {expected}")
        self.info = {"path": str(self.path), "format": suffix[1:], "target": expected_target,
                     "backend": backend, "board_tested": False,
                     "inputs": [variable_info(item) for item in self.inputs],
                     "outputs_in_graph_order": [variable_info(item) for item in self.outputs],
                     "semantic_output_names": dict(zip(SEMANTICS, self.names)),
                     "output_order": "explicit_names" if output_names else "graph_order"}

    def validate_inputs(self, arrays):
        if set(arrays) != INPUT_NAMES:
            raise ValueError(f"NPZ names must match seven inputs; missing={sorted(INPUT_NAMES-set(arrays))}, "
                             f"extra={sorted(set(arrays)-INPUT_NAMES)}")
        for variable in self.inputs:
            value = arrays[variable.name]
            if tuple(value.shape) != tuple(variable.type.shape):
                raise ValueError(f"{self.path.name}: {variable.name} input shape {value.shape} "
                                 f"!= {tuple(variable.type.shape)}")
            if value.dtype != np.dtype(variable.type.np_dtype):
                raise ValueError(f"{self.path.name}: {variable.name} input dtype {value.dtype} "
                                 f"!= {np.dtype(variable.type.np_dtype)}")
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite input {variable.name}")

    def run(self, arrays, simulator_dir):
        # Independent buffers prevent either backend mutating the other model's sample.
        # 独立输入缓冲区，避免某一运行后端修改另一端使用的样本。
        feed = {name: np.array(value, copy=True, order="C") for name, value in arrays.items()}
        started = time.perf_counter()
        if self.path.suffix.lower() == ".hbm":
            simulator_dir.mkdir(parents=True, exist_ok=True)
            values = self.graph.feed(feed, output_dir=str(simulator_dir), remote_ip=None)
        else:
            values = self.graph(*[feed[item.name] for item in self.inputs])
        seconds = time.perf_counter() - started
        if isinstance(values, dict):
            if set(values) != {item.name for item in self.outputs}:
                raise ValueError(f"{self.path.name}: returned output names differ from graph metadata")
            ordered = [values[name] for name in self.names]
        else:
            if len(values) != len(self.outputs):
                raise ValueError(f"{self.path.name}: wrong number of returned outputs")
            named = dict(zip((item.name for item in self.outputs), values))
            ordered = [named[name] for name in self.names]
        # Copy immediately, before another invocation can overwrite runtime caches.
        # 立即复制，防止随后调用覆盖运行时缓存。
        copied = [np.array(value, copy=True, order="C") for value in ordered]
        for semantic, value, variable, expected in zip(
                SEMANTICS, copied, self.ordered_outputs, EXPECTED_SHAPES):
            if value.shape != expected or value.dtype != np.dtype(variable.type.np_dtype):
                raise ValueError(f"{self.path.name}: invalid {semantic} shape/dtype: "
                                 f"{value.shape}/{value.dtype}")
            if not np.isfinite(value).all():
                raise ValueError(f"{self.path.name}: non-finite {semantic}")
        return copied, seconds


def raw_statistics(reference, test):
    """Accumulate in chunks to bound float64 memory. / 分块计算，限制双精度内存占用。"""
    stats = {"elements": reference.size, "absolute_error_sum": 0.0,
             "max_abs_diff": 0.0, "dot": 0.0, "reference_square_sum": 0.0, "test_square_sum": 0.0}
    reference, test = reference.reshape(-1), test.reshape(-1)
    for offset in range(0, reference.size, 262144):
        a = reference[offset:offset+262144].astype(np.float64)
        b = test[offset:offset+262144].astype(np.float64)
        difference = np.abs(a-b)
        stats["absolute_error_sum"] += float(difference.sum())
        stats["max_abs_diff"] = max(stats["max_abs_diff"], float(difference.max()))
        stats["dot"] += float(np.dot(a, b))
        stats["reference_square_sum"] += float(np.dot(a, a))
        stats["test_square_sum"] += float(np.dot(b, b))
    if not all(math.isfinite(value) for value in stats.values()):
        raise ValueError("Raw metric accumulation overflowed; cannot report finite differences")
    return stats


def raw_metrics(stats):
    reference_norm, test_norm = math.sqrt(stats["reference_square_sum"]), math.sqrt(stats["test_square_sum"])
    cosine = max(-1.0, min(1.0, stats["dot"] / reference_norm / test_norm)) if reference_norm and test_norm else None
    return {"elements": stats["elements"], "cosine_similarity": cosine,
            "mae": stats["absolute_error_sum"] / stats["elements"],
            "max_abs_diff": stats["max_abs_diff"]}


def validate_predictions(prediction):
    boxes, scores, labels = (prediction[key] for key in ("boxes_3d", "scores", "labels"))
    if boxes.shape != (len(scores), 9) or labels.shape != scores.shape or scores.ndim != 1:
        raise ValueError("Invalid decoded detection shapes")
    if not all(np.isfinite(value).all() for value in prediction.values()):
        raise ValueError("Non-finite decoded prediction")
    if not np.issubdtype(labels.dtype, np.integer) or ((labels < 0) | (labels >= 10)).any():
        raise ValueError("Decoded detection labels must be integers in [0, 9]")
    occupancy = prediction["occupancy"]
    if occupancy.shape != (200, 200, 16) or not np.issubdtype(occupancy.dtype, np.integer):
        raise ValueError("Invalid decoded occupancy shape/dtype")
    if ((occupancy < 0) | (occupancy >= 18)).any():
        raise ValueError("Decoded occupancy labels must be in [0, 17]")


def detection_metrics(reference, test, distance_limit):
    """Greedy same-class 3D geometric-center matching. / 同类三维几何中心距离贪心匹配。"""
    centers = []
    for prediction in (reference, test):
        boxes = prediction["boxes_3d"].astype(np.float64)
        center = boxes[:, :3].copy()
        center[:, 2] += boxes[:, 5] / 2  # demo.decode exposes bottom-z / 解码输出为底面 z。
        centers.append(center)
    candidates = []
    for i, (center, label) in enumerate(zip(centers[0], reference["labels"])):
        indices = np.flatnonzero(test["labels"] == label)
        distances = np.linalg.norm(centers[1][indices] - center, axis=1)
        candidates.extend((float(distance), i, int(j)) for distance, j in zip(distances, indices)
                          if distance <= distance_limit)
    seen_reference, seen_test, pairs = set(), set(), []
    for distance, i, j in sorted(candidates):
        if i not in seen_reference and j not in seen_test:
            seen_reference.add(i)
            seen_test.add(j)
            pairs.append((distance, i, j))
    n_reference, n_test, count = len(reference["scores"]), len(test["scores"]), len(pairs)
    distances = [distance for distance, _, _ in pairs]
    score_deltas = [float(test["scores"][j])-float(reference["scores"][i]) for _, i, j in pairs]
    center_deltas = np.array([centers[1][j]-centers[0][i] for _, i, j in pairs])
    return {"reference_count": n_reference, "test_count": n_test, "matched_count": count,
            "reference_coverage": count/n_reference if n_reference else None,
            "test_match_rate_precision_like": count/n_test if n_test else None,
            "mean_center_distance_m": float(np.mean(distances)) if count else None,
            "max_center_distance_m": max(distances, default=None),
            "mean_center_delta_xyz_m": center_deltas.mean(axis=0).tolist() if count else None,
            "mean_abs_center_delta_xyz_m": np.abs(center_deltas).mean(axis=0).tolist() if count else None,
            "mean_score_delta": float(np.mean(score_deltas)) if count else None,
            "mean_abs_score_difference": float(np.mean(np.abs(score_deltas))) if count else None,
            "max_abs_score_difference": max(map(abs, score_deltas), default=None)}


def aggregate_detections(samples):
    metrics = [item["detection"] for item in samples
               if item.get("status") == "completed" and "detection" in item]
    count = sum(item["matched_count"] for item in metrics)
    reference = sum(item["reference_count"] for item in metrics)
    test = sum(item["test_count"] for item in metrics)
    result = {"reference_count": reference, "test_count": test, "matched_count": count,
              "reference_coverage": count/reference if reference else None,
              "test_match_rate_precision_like": count/test if test else None}
    for key in ("mean_center_distance_m", "mean_center_delta_xyz_m", "mean_abs_center_delta_xyz_m",
                "mean_score_delta", "mean_abs_score_difference"):
        if not count:
            result[key] = None
        else:
            value = sum(np.asarray(item[key]) * item["matched_count"] for item in metrics
                        if item["matched_count"]) / count
            result[key] = value.tolist()
    for key in ("max_center_distance_m", "max_abs_score_difference"):
        result[key] = max((item[key] for item in metrics if item[key] is not None), default=None)
    return result


def occupancy_metrics(confusion):
    """Reference rows, test columns; no real GT. / 行为参考、列为测试，无真实 GT。"""
    diagonal = np.diag(confusion)
    reference_count, test_count = confusion.sum(axis=1), confusion.sum(axis=0)
    total = int(confusion.sum())
    nonfree = int(reference_count[:FREE_CLASS].sum())
    per_class = []
    for index, name in enumerate(OCCUPANCY_CLASSES):
        intersection = int(diagonal[index])
        reference, test = int(reference_count[index]), int(test_count[index])
        union = reference + test - intersection
        per_class.append({"class_id": index, "class_name": name, "reference_count": reference,
                          "test_count": test, "intersection": intersection, "union": union,
                          "agreement_on_reference_class": intersection/reference if reference else None,
                          "iou_against_reference_pseudo_gt": intersection/union if union else None})
    return {"voxel_count": total, "agreement_all_voxels": int(diagonal.sum())/total if total else None,
            "reference_nonfree_voxel_count": nonfree,
            "agreement_reference_nonfree": int(diagonal[:FREE_CLASS].sum())/nonfree if nonfree else None,
            "confusion_rows_reference_columns_test": confusion.tolist(), "per_class": per_class}


def save_report(report):
    """Save the full JSON and long-form metric CSV, even on failure. / 失败也保存完整报告。"""
    json_path = Path(report["report_paths"]["json"])
    csv_path = Path(report["report_paths"]["csv"])
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scope", "sample", "section", "metric", "semantic", "class_id",
                         "reference_class", "test_class", "value"])
        writer.writerow(["report", "", "execution", "status", "", "", "", "", report["status"]])
        writer.writerow(["report", "", "scope", "limitations", "", "", "", "", report["scope"]])
        if "error" in report:
            writer.writerow(["report", "", "execution", "error", "", "", "", "", report["error"]])
        entries = [("sample", item.get("input", ""), item) for item in report["samples"]]
        if "aggregate" in report:
            entries.append(("aggregate", "", report["aggregate"]))
        for scope, sample, item in entries:
            for key in ("status", "error", "reference_seconds", "test_seconds"):
                if key in item:
                    writer.writerow([scope, sample, "execution", key, "", "", "", "", item[key]])
            for raw in item.get("raw_outputs", []):
                for key in ("elements", "cosine_similarity", "mae", "max_abs_diff"):
                    writer.writerow([scope, sample, "raw", key, raw["semantic"], "", "", "", raw[key]])
            for key, value in item.get("detection", {}).items():
                writer.writerow([scope, sample, "detection", key, "", "", "", "",
                                 json.dumps(value) if isinstance(value, list) else value])
            occupancy = item.get("occupancy", {})
            for key, value in occupancy.items():
                if key not in ("per_class", "confusion_rows_reference_columns_test"):
                    writer.writerow([scope, sample, "occupancy", key, "", "", "", "", value])
            for row in occupancy.get("per_class", []):
                for key, value in row.items():
                    if key != "class_id":
                        writer.writerow([scope, sample, "occupancy_class", key, "", row["class_id"], "", "", value])
            for reference, row in enumerate(occupancy.get("confusion_rows_reference_columns_test", [])):
                for test, value in enumerate(row):
                    writer.writerow([scope, sample, "occupancy_confusion", "count", "", "", reference, test, value])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=local_path, required=True, help="Reference nash-p .bc/.hbm / 参考模型")
    parser.add_argument("--test", type=local_path, required=True, help="Test nash-p .bc/.hbm / 测试模型")
    parser.add_argument("--inputs", type=local_path, nargs="+", required=True, help="Prepared NPZ samples / 多个预处理样本")
    parser.add_argument("--config", type=local_path, default=ROOT / "config/bevfusion_hbir.json")
    parser.add_argument("--output", type=local_path, default=ROOT / "outputs/comparison", help="Report directory / 报告目录")
    parser.add_argument("--match-distance", type=float, default=2.0, help="Same-class center distance in metres / 同类中心距离，默认2米")
    parser.add_argument("--reference-output-names", nargs=5, metavar="NAME", help="Five names in semantic order / 参考输出语义顺序")
    parser.add_argument("--test-output-names", nargs=5, metavar="NAME", help="Five names in semantic order / 测试输出语义顺序")
    parser.add_argument("--oe-root", help="OE root; otherwise use OE_ROOT/oe_assets discovery / 可选OE路径")
    args = parser.parse_args(argv)
    report = compare_models(args.reference, args.test, args.inputs, args.config, args.output,
                            args.match_distance, args.reference_output_names, args.test_output_names,
                            oe_root_value=args.oe_root)
    return 0 if report["status"] == "completed" else 1


def compare_models(reference, test, inputs, config, output_dir, match_distance=2,
                   reference_output_names=None, test_output_names=None,
                   expected_reference_target="nash-p", *, oe_root_value=None):
    """Compare and persist a report dict; inspect status for failure.

    比较并保存报告，调用方须检查 status。CLI 始终要求两端 nash-p；仅内部转换诊断
    可显式指定 expected_reference_target='nash-m'，测试端始终要求 nash-p。
    """
    args = argparse.Namespace(reference=local_path(reference), test=local_path(test),
                              inputs=[local_path(path) for path in inputs], config=local_path(config),
                              output=local_path(output_dir), match_distance=match_distance,
                              reference_output_names=reference_output_names,
                              test_output_names=test_output_names, oe_root=oe_root_value)
    output = args.output.resolve()
    protected = {path.resolve() for path in [args.reference, args.test, args.config, *args.inputs]}
    report_paths = {kind: (output / f"comparison.{kind}").resolve() for kind in ("json", "csv")}
    conflicts = protected.intersection(report_paths.values())
    if conflicts:
        # Keep a failed report without overwriting a model, sample or configuration.
        # 报告路径与来源冲突时，另存失败报告，保护模型、样本与配置。
        prefix = f"comparison.failed.{time.time_ns()}"
        report_paths = {kind: output / f"{prefix}.{kind}" for kind in ("json", "csv")}
        while protected.intersection(path.resolve() for path in report_paths.values()):
            prefix += "_"
            report_paths = {kind: output / f"{prefix}.{kind}" for kind in ("json", "csv")}
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "reference": str(args.reference.resolve()), "test": str(args.test.resolve()),
              "report_paths": {kind: str(path) for kind, path in report_paths.items()},
              "source_sha256": {},
              "inputs": [str(path.resolve()) for path in args.inputs], "config": str(args.config.resolve()),
              "match_distance_m": args.match_distance if math.isfinite(args.match_distance) else str(args.match_distance),
              "scope": "Reference-model difference only, no ground-truth accuracy or real mIoU; "
                       "occupancy reference is pseudo-GT, class IoU is against that pseudo-GT. "
                       "HBM is slow local simulation with remote_ip=None, not board-tested. "
                       "No metric pass/fail thresholds. / 仅参考模型差异，无真实GT精度或真实mIoU；"
                       "occupancy以参考模型为伪GT，逐类IoU仅相对伪GT；HBM为慢速本地模拟，未板测；无差异通过阈值。",
              "definitions": {"raw_cosine": "null if either norm is zero; aggregate concatenates all samples",
                              "detection_matching": "Globally distance-sorted greedy, same class, inclusive distance cutoff",
                              "detection_centers": "XYZ geometric centers in metres (recover center-z from decoded bottom-z)",
                              "deltas": "test minus reference; means use matched pairs only",
                              "reference_coverage": "matched/reference detections; null for zero denominator",
                              "test_match_rate_precision_like": "matched/test detections, reference-model agreement, not GT precision",
                              "occupancy_agreement": "Exact class equality; nonfree restricts to reference class != 17",
                              "class_agreement": "confusion diagonal/reference row count; null if reference class absent",
                              "class_iou": "diagonal/(reference count+test count-diagonal); null if union is zero",
                              "aggregate": "Raw metrics by element, detections by matched pair, occupancy by voxel; completed samples only",
                              "semantic_order": list(SEMANTICS)},
              "samples": []}
    raw_totals = [None] * 5
    confusion_total = np.zeros((18, 18), dtype=np.int64)
    save_report(report)
    try:
        if conflicts:
            raise ValueError(f"Report paths would overwrite source files: {sorted(map(str, conflicts))}; "
                             "failure report saved under separate filenames")
        if not args.inputs:
            raise ValueError("At least one prepared NPZ input is required")
        if not math.isfinite(args.match_distance) or args.match_distance <= 0:
            raise ValueError("--match-distance must be finite and positive")
        for kind, path in (("reference", args.reference), ("test", args.test), ("config", args.config)):
            report["source_sha256"][kind] = sha256(path)
        report["source_sha256"]["inputs"] = []
        for path in args.inputs:
            report["source_sha256"]["inputs"].append({"path": str(path.resolve()), "sha256": sha256(path)})
        sys.path.insert(0, str(oe_root(args.oe_root) / "package/host/ai_toolchain"))
        from demo import decode, load_config
        config = load_config(args.config)
        json.dumps(config, allow_nan=False)  # Reject invalid config before adding it to the saved report.
        report["decode_config"] = config
        reference = Model(args.reference, args.reference_output_names, expected_reference_target)
        report["reference_model"] = reference.info
        test = Model(args.test, args.test_output_names)
        report["test_model"] = test.info
        for left, right in zip(sorted(reference.info["inputs"], key=lambda item: item["name"]),
                               sorted(test.info["inputs"], key=lambda item: item["name"])):
            if left != right:
                raise ValueError(f"Reference/test input contracts differ: {left} vs {right}")
        save_report(report)
        for index, path in enumerate(args.inputs):
            sample = {"index": index, "input": str(path.resolve()), "status": "running"}
            report["samples"].append(sample)
            save_report(report)
            print(f"Sample {index+1}/{len(args.inputs)}: {path}", flush=True)
            try:
                with np.load(path, allow_pickle=False) as stored:
                    arrays = {name: np.array(stored[name], copy=True, order="C") for name in stored.files}
                reference.validate_inputs(arrays)
                test.validate_inputs(arrays)
                sample["inputs"] = {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                    for name, value in arrays.items()}
                left, sample["reference_seconds"] = reference.run(arrays, output / f"hbm_host/sample_{index:04d}/reference")
                right, sample["test_seconds"] = test.run(arrays, output / f"hbm_host/sample_{index:04d}/test")
                del arrays
                statistics = [raw_statistics(a, b) for a, b in zip(left, right)]
                sample["raw_outputs"] = [dict(semantic=semantic, shape=list(shape), **raw_metrics(stats))
                                         for semantic, shape, stats in zip(SEMANTICS, EXPECTED_SHAPES, statistics)]
                left_prediction, right_prediction = decode(left, config), decode(right, config)
                validate_predictions(left_prediction)
                validate_predictions(right_prediction)
                sample["detection"] = detection_metrics(left_prediction, right_prediction, args.match_distance)
                a = left_prediction["occupancy"].astype(np.int64).reshape(-1)
                b = right_prediction["occupancy"].astype(np.int64).reshape(-1)
                confusion = np.bincount(a*18+b, minlength=18*18).reshape(18, 18)
                sample["occupancy"] = occupancy_metrics(confusion)
                sample["status"] = "completed"
                for i, stats in enumerate(statistics):
                    if raw_totals[i] is None:
                        raw_totals[i] = dict(stats)
                    else:
                        for key, value in stats.items():
                            raw_totals[i][key] = max(raw_totals[i][key], value) if key == "max_abs_diff" else raw_totals[i][key]+value
                confusion_total += confusion
                report["aggregate"] = {
                    "completed_samples": sum(item["status"] == "completed" for item in report["samples"]),
                    "raw_outputs": [dict(semantic=semantic, **raw_metrics(stats))
                                    for semantic, stats in zip(SEMANTICS, raw_totals)],
                    "detection": aggregate_detections(report["samples"]),
                    "occupancy": occupancy_metrics(confusion_total)}
                del left, right, left_prediction, right_prediction
                save_report(report)
            except Exception as error:
                sample.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        save_report(report)
    print(f"Status: {report['status']}; JSON: {report['report_paths']['json']}; "
          f"CSV: {report['report_paths']['csv']}", flush=True)
    return report


if __name__ == "__main__":
    raise SystemExit(main())
