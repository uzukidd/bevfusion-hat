"""Benchmark whole models on identical prepared inputs. / 使用同一输入测量各模型整体前向耗时。"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from oe_assets import ROOT, local_path, oe_root, sha256

EXPECTED_SHAPES = ((1, 16384, 256), (1, 900, 10), (1, 900, 3),
                   (1, 900, 10), (1, 200, 200, 16, 18))


def summarize(milliseconds):
    """Report serial frame latency and its reciprocal throughput. / 统计串行帧延迟与倒数吞吐量。"""
    values = np.asarray(milliseconds, dtype=np.float64)
    mean = float(values.mean())
    return {
        'iterations': len(values), 'mean_ms': mean,
        'median_ms': float(np.median(values)),
        'p90_ms': float(np.percentile(values, 90)),
        'p95_ms': float(np.percentile(values, 95)),
        'min_ms': float(values.min()), 'max_ms': float(values.max()),
        'std_ms': float(values.std()), 'serial_frames_per_second': 1000.0 / mean,
        'samples_ms': values.tolist(),
    }


def validate_outputs(outputs):
    """Validate outside the measured interval. / 在计时区间之外检查输出。"""
    if len(outputs) != len(EXPECTED_SHAPES):
        raise ValueError('Expected five raw model outputs')
    for index, (value, shape) in enumerate(zip(outputs, EXPECTED_SHAPES)):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid model output {index}: {value.shape}')


def measure(run, synchronize, warmup, iterations, name):
    """Measure synchronized wall time, excluding setup and validation. / 同步测量墙钟时间，排除准备和检查。"""
    result = None
    warmup_times = []
    for index in range(warmup):
        synchronize()
        start = time.perf_counter()
        result = run()
        synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        warmup_times.append(elapsed)
        print(f'{name}: warmup {index + 1}/{warmup}: {elapsed:.3f} ms', flush=True)
        del result
    samples = []
    for index in range(iterations):
        synchronize()
        start = time.perf_counter()
        result = run()
        synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        samples.append(elapsed)
        print(f'{name}: measured {index + 1}/{iterations}: {elapsed:.3f} ms', flush=True)
        if index == iterations - 1:
            validate_outputs(result)
        del result
    report = summarize(samples)
    report['warmup_samples_ms'] = warmup_times
    report['last_outputs_validated'] = True
    return report


def benchmark_float(args, arrays):
    """Time only the original float HAT forward with device-resident inputs. / 仅测原版 HAT 浮点前向。"""
    import torch
    import horizon_plugin_pytorch as horizon
    from demo import load_config
    from models.bevfusion import build_float_model, raw_forward

    horizon.march.set_march('nash-p')
    start = time.perf_counter()
    model, load_info = build_float_model(args.checkpoint, load_config(args.config), args.float_device)
    tensors = {key: torch.from_numpy(value).to(args.float_device) for key, value in arrays.items()}
    synchronize = (lambda: torch.cuda.synchronize(args.float_device)) if args.float_device == 'cuda' else (lambda: None)
    synchronize()
    setup_seconds = time.perf_counter() - start
    with torch.inference_mode():
        report = measure(lambda: raw_forward(model, tensors), synchronize,
                         args.warmup, args.iterations, 'float')
    report.update(
        model='float', path=str(args.checkpoint.resolve()), sha256=sha256(args.checkpoint),
        backend='PyTorch float32 on CUDA' if args.float_device == 'cuda' else 'PyTorch float32 on CPU',
        device=torch.cuda.get_device_name() if args.float_device == 'cuda' else 'host CPU',
        setup_seconds=setup_seconds, strict_load=load_info['strict_load'],
        input_output_location='CUDA -> CUDA' if args.float_device == 'cuda' else 'CPU -> CPU')
    del tensors, model
    gc.collect()
    if args.float_device == 'cuda':
        torch.cuda.empty_cache()
    return report


def benchmark_hbir(args, arrays, name):
    """Time the synchronous HBDK host interpreter call. / 测量 HBDK 主机解释器同步调用。"""
    from hbdk4 import compiler
    path = args.original if name == 'official' else args.s600
    start = time.perf_counter()
    module = compiler.load(str(path))
    function = module.functions[0]
    feed = []
    for variable in function.inputs:
        value = arrays[variable.name]
        if tuple(value.shape) != tuple(variable.type.shape):
            raise ValueError(f'{variable.name}: input {value.shape} != model {variable.type.shape}')
        feed.append(value)
    setup_seconds = time.perf_counter() - start

    def run():
        values = function(*feed)
        if isinstance(values, dict):
            return [values[variable.name] for variable in function.outputs]
        return values

    report = measure(run, lambda: None, args.warmup, args.iterations, name)
    attrs = {item.name: str(item.attr) for item in module.module.operation.attributes}
    report.update(model=name, path=str(path.resolve()), sha256=sha256(path),
                  backend='HBDK host HBIR interpreter (not BPU)', device='host CPU',
                  target=attrs.get('hbdk.target'), setup_seconds=setup_seconds,
                  input_output_location='CPU NumPy -> CPU NumPy',
                  timing_note='Includes interpreter boundary conversion and host execution / 包含解释器边界转换及主机执行')
    del function, module, feed
    gc.collect()
    return report


def save_results(args, report):
    """Persist JSON and a concise CSV after each model. / 每个模型完成后保存 JSON 和 CSV。"""
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'benchmark.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    columns = ['model', 'backend', 'device', 'iterations', 'mean_ms', 'median_ms',
               'p90_ms', 'p95_ms', 'min_ms', 'max_ms', 'std_ms', 'serial_frames_per_second']
    with (args.output / 'benchmark.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['results'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=['float', 'official', 's600'], default=['float', 'official', 's600'])
    parser.add_argument('--inputs', type=local_path, default=ROOT/'outputs/demo/inputs.npz')
    parser.add_argument('--config', type=local_path, default=ROOT/'config/bevfusion_hbir.json')
    parser.add_argument('--checkpoint', type=local_path, default=ROOT/'data/checkpoints/float-checkpoint-last.pth.tar')
    parser.add_argument('--original', type=local_path, default=ROOT/'data/checkpoints/quantized.bc')
    parser.add_argument('--s600', type=local_path, default=ROOT/'outputs/s600/bevfusion.bc')
    parser.add_argument('--output', type=local_path, default=ROOT/'outputs/benchmark')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--float-device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--oe-root')
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1 or args.threads < 1:
        parser.error('--warmup, --iterations and --threads must be positive')
    if len(args.models) != len(set(args.models)):
        parser.error('--models must not contain duplicates')
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(oe_root(args.oe_root)/'package/host/ai_toolchain'))
    import torch
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    with np.load(args.inputs, allow_pickle=False) as stored:
        arrays = {key: np.ascontiguousarray(stored[key]) for key in stored.files}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Non-finite prepared input')
    report = {
        'status': 'running', 'frame_batch_size': 1, 'camera_count_per_frame': 6,
        'warmup': args.warmup, 'iterations': args.iterations,
        'inputs': str(args.inputs.resolve()), 'inputs_sha256': sha256(args.inputs),
        'input_shapes': {key: list(value.shape) for key, value in arrays.items()},
        'environment': {'platform': platform.platform(), 'machine': platform.machine(),
            'cpu': platform.processor(), 'logical_cpu_count': os.cpu_count(),
            'python': sys.version, 'torch': torch.__version__,
            'horizon_plugin': importlib.metadata.version('horizon-plugin-pytorch'),
            'hbdk4': importlib.metadata.version('hbdk4-compiler'),
            'torch_cpu_threads': torch.get_num_threads(),
            'thread_environment': {key: os.environ.get(key) for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']},
            'cuda_device': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'cudnn_benchmark': False, 'tf32': False},
        'scope': 'Warm serial wall-clock latency for one frame (six cameras + LiDAR), raw five outputs. Excludes loading, preprocessing, input device transfer, decoding, plotting and output disk writes. CUDA synchronized before/after forward. HBIR includes host-interpreter boundary costs. / 单帧六相机与点云预热后串行墙钟延迟，排除加载、预处理、输入设备拷贝、解码、绘图和写盘；CUDA前后同步，HBIR包含主机解释器边界开销。',
        'limitations': 'Repeated identical sample; float uses CUDA by default while HBIR executes on host CPU. Results do not measure BPU speed, board HBM latency, pipelined throughput or dataset accuracy. / 重复同一样本，默认浮点在GPU而HBIR在CPU；不代表BPU、板端HBM、流水线吞吐或数据集精度。',
        'results': [],
    }
    save_results(args, report)
    for name in args.models:
        result = benchmark_float(args, arrays) if name == 'float' else benchmark_hbir(args, arrays, name)
        report['results'].append(result)
        save_results(args, report)
    report['status'] = 'completed'
    save_results(args, report)
    for item in report['results']:
        print(f"{item['model']}: mean={item['mean_ms']:.3f} ms, median={item['median_ms']:.3f} ms, "
              f"p95={item['p95_ms']:.3f} ms, serial FPS={item['serial_frames_per_second']:.3f}", flush=True)
    print('Saved', args.output/'benchmark.json', args.output/'benchmark.csv', flush=True)


if __name__ == '__main__':
    main()
