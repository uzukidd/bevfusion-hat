"""Run the strictly restored float PyTorch model. / 运行严格恢复的浮点 PyTorch 模型。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from oe_assets import ROOT, oe_root, local_path, sha256
from demo import prepare, load_config, decode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=local_path,default=ROOT/'data/checkpoints/float-checkpoint-last.pth.tar')
    parser.add_argument('--config',type=local_path,default=ROOT/'config/bevfusion_hbir.json')
    parser.add_argument('--samples',type=local_path,help='Sensor files; with --inputs specify explicitly to enable visualization / 使用历史输入时须显式指定对应样例才绘图')
    parser.add_argument('--inputs',type=local_path,help='Reuse the exact HBIR inputs / 复用 HBIR 输入')
    parser.add_argument('--output',type=local_path,default=ROOT/'outputs/demo_float')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--point-dim',type=int,choices=[4,5],default=5)
    parser.add_argument('--fifth-column',choices=['ring','time-lag'],default='ring')
    parser.add_argument('--oe-root')
    parser.add_argument('--check-load-only',action='store_true')
    parser.add_argument('--no-visualization',action='store_true')
    args=parser.parse_args()
    visualize_enabled = not args.no_visualization and (args.inputs is None or args.samples is not None)
    samples = args.samples or ROOT/'data/samples'
    sys.path.insert(0,str(ROOT))
    sys.path.insert(0,str(oe_root(args.oe_root)/'package/host/ai_toolchain'))
    import torch
    import horizon_plugin_pytorch as horizon
    from models.bevfusion import build_float_model, raw_forward
    torch.set_num_threads(8)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    horizon.march.set_march('nash-p')
    cfg=load_config(args.config)
    args.output.mkdir(parents=True,exist_ok=True)
    model,info=build_float_model(args.checkpoint,cfg,args.device)
    info.update(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=sha256(args.checkpoint),
                device=args.device, torch_version=torch.__version__,
                execution='float32 PyTorch original HAT modules; no HBIR execution / 原版 HAT 浮点 PyTorch 模块，未执行 HBIR')
    (args.output/'load_report.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    print('LOAD',json.dumps(info,ensure_ascii=False),flush=True)
    if args.check_load_only:
        return
    if args.inputs:
        with np.load(args.inputs,allow_pickle=False) as stored:
            arrays={k:stored[k] for k in stored.files}
        metadata={'input_source':str(args.inputs.resolve()), 'parameter_status':cfg['parameter_status']}
    else:
        arrays,metadata=prepare(samples,cfg,args.device,args.point_dim,args.fifth_column)
    np.savez(args.output/'inputs.npz',**arrays)
    tensors={k:torch.from_numpy(v).to(args.device) for k,v in arrays.items()}
    start=time.monotonic()
    print('Running FLOAT PyTorch / 正在运行浮点 PyTorch ...',flush=True)
    with torch.inference_mode():
        outputs=raw_forward(model,tensors)
    if args.device=='cuda':
        torch.cuda.synchronize()
    elapsed=time.monotonic()-start
    values=[v.detach().cpu().numpy() for v in outputs]
    expected=[(1,16384,256),(1,900,10),(1,900,3),(1,900,10),(1,200,200,16,18)]
    if len(values)!=5 or any(v.shape!=shape or not np.isfinite(v).all() for v,shape in zip(values,expected)):
        raise RuntimeError('Invalid raw floating point model output')
    np.savez_compressed(args.output/'raw_outputs.npz',**{f'_output_{i}':v for i,v in enumerate(values)})
    predictions=decode(values,cfg)
    if not all(np.isfinite(v).all() for v in predictions.values()):
        raise RuntimeError('Non-finite decoded predictions')
    np.savez_compressed(args.output/'predictions.npz',**predictions)
    metadata.update(info)
    metadata.update(inference_seconds=elapsed,output_shapes=[list(x.shape) for x in values],
        detections=len(predictions['scores']),
        score_max=float(predictions['scores'].max()) if len(predictions['scores']) else None,
        occupancy_shape=list(predictions['occupancy'].shape),
        occupancy_classes=np.unique(predictions['occupancy']).tolist(),
        box_columns=['x','y','z_bottom','dx','dy','dz','yaw','vx','vy'],
        validation='Strict state load and finite outputs verified; reconstruction/external parameters require original-config confirmation / 已验证严格权重加载与输出有限值；恢复配置和图外参数仍需官方原配置核验')
    if visualize_enabled:
        if args.inputs:
            check_arrays,_=prepare(samples,cfg,args.device,args.point_dim,args.fifth_column)
            if set(check_arrays)!=set(arrays) or any(not np.array_equal(arrays[k],check_arrays[k]) for k in check_arrays):
                raise ValueError('--inputs does not match --samples/config/point format; cannot visualize')
        from visualize import visualize
        metadata['visualization']=str(visualize(samples,predictions['boxes_3d'],predictions['scores'],
            predictions['labels'],predictions['occupancy'],args.output,cfg['score_threshold'],args.point_dim))
    else:
        metadata['visualization_status']='disabled; historical inputs require matching explicit --samples / 已关闭；历史输入绘图须显式指定匹配样例'
    (args.output/'summary.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    (args.output/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    print(json.dumps(metadata,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
