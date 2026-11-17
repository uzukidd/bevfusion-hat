"""Export calibration tensors from sensor frames. / 从传感器样例输出校准张量。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from oe_assets import ROOT, local_path, oe_root, sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=local_path,nargs='+',default=[ROOT/'data/samples'],
                        help='One or more synchronized frame directories / 一个或多个同步帧目录')
    parser.add_argument('--config',type=local_path,default=ROOT/'config/bevfusion_hbir.json')
    parser.add_argument('--output',type=local_path,default=ROOT/'outputs/calibration')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--point-dim',type=int,choices=[4,5],default=5)
    parser.add_argument('--fifth-column',choices=['ring','time-lag'],default='ring')
    parser.add_argument('--oe-root')
    args=parser.parse_args()
    sys.path.insert(0,str(oe_root(args.oe_root)/'package/host/ai_toolchain'))
    import torch
    from demo import prepare,load_config
    torch.set_num_threads(8)
    cfg=load_config(args.config)
    destinations=[args.output/f'frame_{i:04d}' for i in range(len(args.samples))]
    if (args.output/'manifest.json').exists() or any((d/'inputs.npz').exists() for d in destinations):
        parser.error('Output already contains calibration data; choose a new --output')
    args.output.mkdir(parents=True,exist_ok=True)
    report={'status':'running','config':str(args.config.resolve()),'config_sha256':sha256(args.config),
            'frames':[],'scope':'Prepared sensor tensors, not quantization statistics or a calibrated model / 仅预处理张量，不是量化统计或校准后模型'}
    (args.output/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest=args.output/'manifest.json'
    try:
        for source,destination in zip(args.samples,destinations):
            start=time.perf_counter()
            arrays,metadata=prepare(source,cfg,args.device,args.point_dim,args.fifth_column)
            if not all(np.isfinite(a).all() for a in arrays.values()):
                raise ValueError(f'Non-finite inputs for {source}')
            destination.mkdir(parents=True,exist_ok=True)
            output=destination/'inputs.npz'
            np.savez(output,**arrays)
            item={'samples':str(source.resolve()),'inputs':output.relative_to(args.output).as_posix(),
                  'sha256':sha256(output),'preprocess_seconds':time.perf_counter()-start,
                  'shapes':{k:list(v.shape) for k,v in arrays.items()},'metadata':metadata}
            report['frames'].append(item)
            manifest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print('Saved',output,flush=True)
        report['status']='completed'
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        manifest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Calibration manifest:',manifest,flush=True)


if __name__=='__main__':
    main()
