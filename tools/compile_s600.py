"""Retarget the official HBIR to S600 Nash-p / 转换并编译 S600 模型。"""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import json
import importlib.metadata
from oe_assets import sha256, local_path


def write_manifest(source, output, elapsed, opt, jobs):
    from hbdk4.compiler.hbm import Hbm
    model=Hbm(str(output))
    record={'source':str(source.resolve()),'source_sha256':sha256(source),
            'model':str(output.resolve()),'model_sha256':sha256(output),
            'model_bytes':output.stat().st_size, 'march':model.march_name,
            'compiler_version':importlib.metadata.version('hbdk4-compiler'),
            'integer_conv':True, 'enable_vpu':False, 'opt':opt, 'jobs':jobs,
            'elapsed_seconds':elapsed,
            'validation':'Compiled and metadata checked; board execution not tested / 已编译并核验元信息，未验证开发板执行'}
    output.with_suffix('.compile.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=local_path, default=Path('data/checkpoints/qat.bc'))
    parser.add_argument('--output', type=local_path, default=Path('outputs/s600/bevfusion.hbm'))
    parser.add_argument('--opt', type=int, choices=(0, 1, 2), default=1)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--convert-only', action='store_true')
    args = parser.parse_args()
    from hbdk4 import compiler
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    module = compiler.load(str(args.input))
    print('Original attributes:', [(a.name, str(a.attr)) for a in module.module.operation.attributes], flush=True)
    print('Converting to nash-p / 正在转换为 S600 nash-p', flush=True)
    # HAT 3.1.19 HbirExporter uses integer convolution for march_modify=nash-p.
    module._integer_conv = True
    module = compiler.convert(module, march='nash-p', enable_vpu=False)
    converted = args.output.with_suffix('.bc')
    compiler.save(module, str(converted))
    print('Converted attributes:', [(a.name, str(a.attr)) for a in module.module.operation.attributes], flush=True)
    print('Saved', converted, flush=True)
    if not args.convert_only:
        print('Compiling / 正在编译', args.output, flush=True)
        compiler.compile(module, str(args.output), march='nash-p', opt=args.opt, jobs=args.jobs, progress_bar=True)
        write_manifest(args.input,args.output,time.monotonic()-start,args.opt,args.jobs)
        print('Compiled', args.output, flush=True)
    print(f'Elapsed: {time.monotonic()-start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
