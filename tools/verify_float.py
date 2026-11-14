"""Compare restored float outputs with S600 HBIR. / 比较恢复浮点模型和 S600 HBIR。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from oe_assets import ROOT, oe_root, local_path, sha256
from demo import load_config, decode
from verify_hbir import detection_comparison


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--float-dir',type=local_path,default=ROOT/'outputs/demo_float')
    parser.add_argument('--hbir',type=local_path,default=ROOT/'outputs/s600/bevfusion.bc')
    parser.add_argument('--output',type=local_path,default=ROOT/'outputs/verification/float_vs_hbir.json')
    parser.add_argument('--oe-root')
    args=parser.parse_args()
    sys.path.insert(0,str(oe_root(args.oe_root)/'package/host/ai_toolchain'))
    from hbdk4 import compiler
    with np.load(args.float_dir/'inputs.npz',allow_pickle=False) as data:
        inputs={k:data[k] for k in data.files}
    with np.load(args.float_dir/'raw_outputs.npz',allow_pickle=False) as data:
        floating=[data[f'_output_{i}'] for i in range(5)]
    module=compiler.load(str(args.hbir))
    function=module.functions[0]
    start=time.monotonic()
    values=function(*[inputs[v.name] for v in function.inputs])
    elapsed=time.monotonic()-start
    if isinstance(values,dict):
        values=[values[v.name] for v in function.outputs]
    quantized=[np.asarray(v) for v in values]
    if len(quantized)!=5:
        raise RuntimeError('Expected five HBIR outputs')
    metrics=[]
    for i,(a,b) in enumerate(zip(floating,quantized)):
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError(f'Invalid output {i}')
        a,b=a.astype(np.float64),b.astype(np.float64)
        norm=np.linalg.norm(a)*np.linalg.norm(b)
        metrics.append({'name':f'_output_{i}','shape':list(a.shape),'finite':True,
            'cosine_similarity':float(np.vdot(a,b)/norm) if norm else None,
            'mean_abs_diff':float(np.abs(a-b).mean()),'max_abs_diff':float(np.abs(a-b).max())})
    cfg=load_config(args.float_dir/'config.json')
    report={'float_summary':str((args.float_dir/'summary.json').resolve()),
            'hbir':str(args.hbir.resolve()),'hbir_sha256':sha256(args.hbir),
            'hbir_seconds':elapsed,'same_input_arrays':True,'raw_outputs':metrics,
            'detection_comparison':detection_comparison(decode(floating,cfg),decode(quantized,cfg)),
            'occupancy_argmax_agreement':float((floating[-1].argmax(-1)==quantized[-1].argmax(-1)).mean()),
            'scope':'One sample comparison of separately provided float checkpoint and quantization HBIR; differences include calibration/export parameters and recovered configuration, not isolated quantization error; qat.bc filename does not establish QAT training / 单样本比较分别提供的浮点权重与量化HBIR，差异包含校准导出参数及恢复配置，不代表单独量化误差，qat.bc文件名不证明QAT训练'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
