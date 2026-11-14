"""Compare original and S600 HBIR on identical inputs / 同输入比较两平台 HBIR。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import sys
import numpy as np
from oe_assets import ROOT, oe_root, local_path


def detection_comparison(original, converted):
    """Match same-class boxes within two metres / 按类别和两米距离匹配检测框。"""
    candidates=[]
    for i,(box,label) in enumerate(zip(original['boxes_3d'],original['labels'])):
        for j,(other,other_label) in enumerate(zip(converted['boxes_3d'],converted['labels'])):
            distance=float(np.linalg.norm(box[:3]-other[:3]))
            if label==other_label and distance<=2:
                candidates.append((distance,i,j))
    seen_a,seen_b,pairs=set(),set(),[]
    for distance,i,j in sorted(candidates):
        if i not in seen_a and j not in seen_b:
            seen_a.add(i); seen_b.add(j)
            pairs.append((distance,i,j))
    return {'original_count':len(original['scores']), 'converted_count':len(converted['scores']),
            'same_class_matches_within_2m':len(pairs),
            'mean_center_distance_m':float(np.mean([x[0] for x in pairs])) if pairs else None,
            'max_center_distance_m':max((x[0] for x in pairs),default=None),
            'mean_abs_score_difference':float(np.mean([abs(float(original['scores'][i])-float(converted['scores'][j]))
                                                       for _,i,j in pairs])) if pairs else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=local_path,default=Path('outputs/demo/inputs.npz'))
    parser.add_argument('--original',type=local_path,default=Path('data/checkpoints/quantized.bc'))
    parser.add_argument('--s600',type=local_path,default=Path('outputs/s600/bevfusion.bc'))
    parser.add_argument('--output',type=local_path,default=Path('outputs/verification/hbir_comparison.json'))
    parser.add_argument('--config',type=local_path,default=ROOT/'config/bevfusion_hbir.json')
    args=parser.parse_args()
    from hbdk4 import compiler
    inputs=dict(np.load(args.inputs))
    results=[]
    for model in [args.original,args.s600]:
        module=compiler.load(str(model))
        function=module.functions[0]
        started=time.monotonic()
        values=function(*[inputs[x.name] for x in function.inputs])
        if isinstance(values,dict):
            values=[values[x.name] for x in function.outputs]
        results.append([np.asarray(x) for x in values])
        print(model, f'{time.monotonic()-started:.2f}s', flush=True)
    metrics=[]
    for i,(a,b) in enumerate(zip(*results)):
        a,b=a.astype(np.float64),b.astype(np.float64)
        norm=np.linalg.norm(a)*np.linalg.norm(b)
        metrics.append({'index':i,'shape':list(a.shape), 'finite':bool(np.isfinite(a).all() and np.isfinite(b).all()),
                        'max_abs_diff':float(np.max(np.abs(a-b))), 'mean_abs_diff':float(np.mean(np.abs(a-b))),
                        'cosine_similarity':float(np.vdot(a,b)/norm) if norm else None})
    occupancy_agreement=float(np.mean(results[0][-1].argmax(-1)==results[1][-1].argmax(-1)))
    sys.path.insert(0,str(oe_root()/'package/host/ai_toolchain'))
    from demo import decode, load_config
    cfg=load_config(args.config)
    detections=[decode(values,cfg) for values in results]
    payload={'original':str(args.original),'s600':str(args.s600),'inputs':str(args.inputs),
             'outputs':metrics,'occupancy_argmax_agreement':occupancy_agreement,
             'detection_comparison':detection_comparison(*detections),
             'scope':'Target conversion comparison on one identical sample, not dataset accuracy / 同一样本平台转换对比，非数据集精度评估'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    print(json.dumps(payload,ensure_ascii=False,indent=2),flush=True)
    if not all(x['finite'] for x in metrics):
        raise RuntimeError('Non-finite model output')


if __name__=='__main__':
    main()
