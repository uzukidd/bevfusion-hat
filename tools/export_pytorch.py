"""PyTorch -> calibration -> HBIR -> S600 HBM. / 从本地 PyTorch 权重校准、导出并编译。

English: calibration accepts prepared NPZ frames. A single demo frame only verifies
conversion, not dataset accuracy. 中文：校准支持多个预处理 NPZ；单帧仅验证链路。
"""
from __future__ import annotations
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from oe_assets import ROOT, local_path, oe_root, sha256

INPUT_NAMES = ('features', 'coors', 'img', 'queries_rebatch_grid', 'restore_bev_grid',
               'reference_points_rebatch', 'bev_pillar_counts')
OUTPUT_NAMES = ('bev_embed', 'outputs_classes', 'reference_out', 'outputs_coords', 'occ_preds')


def arrays_from_outputs(value):
    """Flatten tensor containers in deterministic order. / 按固定顺序展开张量容器。"""
    if isinstance(value, dict):
        return [x for v in value.values() for x in arrays_from_outputs(v)]
    if isinstance(value, (tuple,list)):
        return [x for v in value for x in arrays_from_outputs(v)]
    if hasattr(value,'detach'):
        value=value.detach().cpu().numpy()
    return [np.asarray(value).copy()]


def compare(reference, actual):
    """Check every raw output before reporting numerical differences. / 检查原始输出并报告误差。"""
    a,b=arrays_from_outputs(reference),arrays_from_outputs(actual)
    if len(a)!=5 or len(b)!=5:
        raise RuntimeError('Expected five raw outputs')
    result=[]
    for name,x,y in zip(OUTPUT_NAMES,a,b):
        if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise RuntimeError(f'Invalid output {name}: {x.shape} / {y.shape}')
        x,y=x.astype(np.float64),y.astype(np.float64)
        norm=np.linalg.norm(x)*np.linalg.norm(y)
        result.append({'name':name,'shape':list(x.shape),'finite':True,
            'cosine_similarity':float(np.vdot(x,y)/norm) if norm else None,
            'max_abs_diff':float(np.abs(x-y).max()),'mean_abs_diff':float(np.abs(x-y).mean())})
    return result


def read_inputs(path):
    """Load one complete fixed-shape input frame. / 读取一组完整固定形状输入。"""
    with np.load(path,allow_pickle=False) as stored:
        if set(stored.files)!=set(INPUT_NAMES):
            raise ValueError(f'{path}: expected exactly {INPUT_NAMES}')
        arrays={name:np.ascontiguousarray(stored[name]) for name in INPUT_NAMES}
    for name,array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(f'Non-finite {name} in {path}')
        if array.dtype != (np.int32 if name=='coors' else np.float32):
            raise ValueError(f'Wrong dtype {name}: {array.dtype}')
    return arrays


def configure_quantization(output):
    """Use the vendor template API available in Plugin 3.1.5. / 使用旧版插件支持的量化模板。"""
    from horizon_plugin_pytorch.dtype import qint8,qint16
    from horizon_plugin_pytorch.quantization import get_qconfig,observer_v2
    from horizon_plugin_pytorch.quantization.qconfig_setter import (
        QconfigSetter,ModuleNameTemplate,ConvDtypeTemplate,MatmulDtypeTemplate)
    return QconfigSetter(
        reference_qconfig=get_qconfig(observer=observer_v2.MSEObserver),
        templates=[ModuleNameTemplate({'':qint8}),
            MatmulDtypeTemplate(input_dtypes=[qint8,qint8]),
            ConvDtypeTemplate(input_dtype=qint8,weight_dtype=qint8),
            ModuleNameTemplate({
                'bev_decoders.0.quant_object_query_embed':{'output':qint16},
                'bev_decoders.0.reference_points':{'dtype':qint16},
                'bev_decoders.0.decoder.layers.5.sa.matmul':{'output':qint16},
            })],
        save_dir=str(output/'qconfig'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=local_path,default=ROOT/'data/checkpoints/float-checkpoint-last.pth.tar')
    parser.add_argument('--config',type=local_path,default=ROOT/'config/bevfusion_hbir.json')
    calibration_group=parser.add_mutually_exclusive_group()
    calibration_group.add_argument('--calibration-inputs',type=local_path,nargs='+')
    calibration_group.add_argument('--calibration-manifest',type=local_path,
                                   help='Manifest produced by export_calibration / 校准数据导出清单')
    parser.add_argument('--output',type=local_path,default=ROOT/'outputs/from_pytorch')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--visualize-samples',type=local_path,
                        help='Optional matching sensor directory; custom NPZ export does not require samples / 可选匹配样例，仅用于绘图')
    parser.add_argument('--calibration-passes',type=int,default=1)
    parser.add_argument('--format',choices=['hbir','hbm'],default=None,
                        help='hbir: export/convert only; hbm: also compile / 输出模式')
    parser.add_argument('--compile',action='store_true',help='Alias for --format hbm / 兼容旧参数')
    parser.add_argument('--opt',type=int,choices=[0,1,2],default=1)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--min-export-cosine',type=float,default=0.90,
                        help='Minimum fake-quant/HBIR raw-output cosine; not a task accuracy threshold / 导出一致性下限，非任务精度指标')
    parser.add_argument('--oe-root')
    args=parser.parse_args()
    if args.compile and args.format=='hbir':
        parser.error('--compile conflicts with --format hbir')
    args.compile=args.compile or args.format=='hbm'
    args.format='hbm' if args.compile else 'hbir'
    if args.calibration_manifest:
        manifest=json.loads(args.calibration_manifest.read_text(encoding='utf-8'))
        if manifest.get('status')!='completed' or not manifest.get('frames'):
            parser.error('Calibration manifest must be completed and contain frames')
        args.calibration_inputs=[]
        for frame in manifest['frames']:
            path=local_path(frame['inputs'])
            if not path.is_absolute():
                path=args.calibration_manifest.parent/path
            if sha256(path)!=frame['sha256']:
                parser.error(f'Calibration manifest checksum mismatch: {path}')
            args.calibration_inputs.append(path)
        if manifest.get('config_sha256')!=sha256(args.config):
            parser.error('Calibration manifest config differs from --config')
    elif args.calibration_inputs is None:
        args.calibration_inputs=[ROOT/'outputs/demo/inputs.npz']
    if args.calibration_passes<1 or args.jobs<1 or not 0<=args.min_export_cosine<=1:
        parser.error('Positive calibration passes/jobs and --min-export-cosine in [0,1] required')
    args.output.mkdir(parents=True,exist_ok=True)
    if any((args.output/name).exists() for name in ['qat.bc','bevfusion.bc','bevfusion.hbm']):
        parser.error('Output already contains model artifacts; choose a new --output directory')
    sys.path.insert(0,str(ROOT))
    sys.path.insert(0,str(oe_root(args.oe_root)/'package/host/ai_toolchain'))
    source_paths=[ROOT/'tools/export_pytorch.py',*sorted((ROOT/'models').glob('*.py'))]
    source_hashes={str(path.relative_to(ROOT)):sha256(path) for path in source_paths}
    (args.output/'executed_export_pytorch.py').write_bytes((ROOT/'tools/export_pytorch.py').read_bytes())
    report={'status':'running','stage':'environment','source':'local PyTorch model and float checkpoint',
        'source_files_sha256':source_hashes,'config_sha256':sha256(args.config),
        'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':sha256(args.checkpoint),
        'march':'nash-p','output_format':args.format,
        'calibration_manifest':str(args.calibration_manifest.resolve()) if args.calibration_manifest else None,
        'downloaded_hbir_used':False,'calibration_only':True,'qat_training':False,
        'calibration_inputs':[{'path':str(p.resolve()),'sha256':sha256(p)} for p in args.calibration_inputs],
        'calibration_passes':args.calibration_passes,
        'limitations':'Demo calibration is not representative dataset calibration or an accuracy claim / 样例校准不代表完整数据集校准及精度认证'}
    started=time.monotonic()
    def record(stage):
        report['stage']=stage
        report['elapsed_seconds']=time.monotonic()-started
        (args.output/'export_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('[stage]',stage,flush=True)
    try:
        import torch
        import horizon_plugin_pytorch as horizon
        from horizon_plugin_pytorch.quantization import prepare,PrepareMethod,FakeQuantState,set_fake_quantize
        from horizon_plugin_pytorch.quantization.hbdk4 import export
        from horizon_plugin_pytorch.utils.global_quant_round_mode import QuantRoundMode
        from hbdk4 import compiler
        from models.bevfusion import build_float_model
        from hat.models.structures.bevfusion import BevFusion
        from demo import load_config
        torch.set_num_threads(8)
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        horizon.set_march('nash-p')
        horizon.qat_mode.set_qat_mode('fuse_bn')
        QuantRoundMode.set(QuantRoundMode.HALF_TO_EVEN)
        report['versions']={p:importlib.metadata.version(p) for p in ['torch','horizon-plugin-pytorch','hbdk4-compiler']}
        cfg=load_config(args.config)
        (args.output/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
        arrays=read_inputs(args.calibration_inputs[0])
        np.savez(args.output/'verification_inputs.npz',**arrays)
        sample={k:torch.from_numpy(v).to(args.device) for k,v in arrays.items()}
        example=(sample,)
        record('load_float')
        float_model,load_report=build_float_model(args.checkpoint,cfg,args.device)
        report['float_load']=load_report
        class ExportBevFusion(BevFusion):
            def forward(self,data):
                det,occ=super().forward(data)
                return tuple(det)+tuple(occ)
        model=ExportBevFusion(lidar_network=float_model.lidar_net,camera_network=float_model.camera_net,
            bev_decoders=float_model.bev_decoders,fuse_module=float_model.fuse_module,
            bev_h=cfg['bev_h'],bev_w=cfg['bev_w']).eval()
        del float_model
        with torch.inference_mode():
            float_outputs=arrays_from_outputs(model(sample))
        np.savez_compressed(args.output/'float_outputs.npz',**dict(zip(OUTPUT_NAMES,float_outputs)))
        record('switch_to_deploy')
        switched=[]
        for name,module in list(model.named_modules()):
            if hasattr(module,'switch_to_deploy'):
                module.switch_to_deploy()
                switched.append(name)
        report['deploy_modules']=switched
        with torch.inference_mode():
            deploy_outputs=arrays_from_outputs(model(sample))
        report['float_vs_deploy']=compare(float_outputs,deploy_outputs)
        for a,b in zip(float_outputs,deploy_outputs):
            np.testing.assert_allclose(a,b,rtol=1e-3,atol=1e-3)
        # Training-only criterion is unused in the fixed raw inference graph.
        model.bev_decoders[0].criterion=None
        record('prepare_calibration')
        prepared=prepare(model,example_inputs=example,qconfig_setter=configure_quantization(args.output),
                         method=PrepareMethod.JIT_STRIP,check_result_dir=str(args.output/'model_check')).eval()
        del model
        gc.collect()
        record('calibration')
        set_fake_quantize(prepared,FakeQuantState.CALIBRATION)
        fixed=[]
        for name,child in prepared.named_modules():
            if hasattr(child,'fix_weight_qscale'):
                child.fix_weight_qscale()
                fixed.append(name)
        report['fixed_weight_qscale_modules']=fixed
        report['quantization_policy']={'activation_observer':'MSEObserver','default_dtype':'qint8',
            'int16_modules':['bev_decoders.0.quant_object_query_embed','bev_decoders.0.reference_points',
                             'bev_decoders.0.decoder.layers.5.sa.matmul'],
            'automatic_precision_templates':True,'fixed_sum_weight_scale':'1/128',
            'round_mode':'HALF_TO_EVEN','prepare':'JIT_STRIP'}
        with torch.no_grad():
            for repeat in range(args.calibration_passes):
                for path in args.calibration_inputs:
                    current=read_inputs(path)
                    if any(current[k].shape!=arrays[k].shape for k in INPUT_NAMES):
                        raise ValueError(f'Calibration shape mismatch in {path}')
                    tensors={k:torch.from_numpy(v).to(args.device) for k,v in current.items()}
                    prepared(tensors)
                    print('[calibrated]',repeat+1,path,flush=True)
        prepared.eval()
        set_fake_quantize(prepared,FakeQuantState.VALIDATION)
        with torch.no_grad():
            fake_outputs=arrays_from_outputs(prepared(sample))
            changed_sample={k:v.clone() for k,v in sample.items()}
            changed_sample['img']=changed_sample['img']*0.7
            changed_fake_outputs=arrays_from_outputs(prepared(changed_sample))
        changed_arrays={k:v.detach().cpu().numpy() for k,v in changed_sample.items()}
        np.savez_compressed(args.output/'changed_fake_quant_outputs.npz',**dict(zip(OUTPUT_NAMES,changed_fake_outputs)))
        report['float_vs_fake_quant']=compare(float_outputs,fake_outputs)
        torch.save({'state_dict':prepared.state_dict(),'source_checkpoint_sha256':report['checkpoint_sha256'],
                    'calibration_inputs':report['calibration_inputs'],'march':'nash-p'},args.output/'calibration-checkpoint.pth.tar')
        np.savez_compressed(args.output/'fake_quant_outputs.npz',**dict(zip(OUTPUT_NAMES,fake_outputs)))
        record('export_hbir')
        module=export(prepared,example,name='bevfusion_from_pytorch',input_names=list(INPUT_NAMES),
                      output_names=list(OUTPUT_NAMES),native_pytree=False)
        compiler.save(module,str(args.output/'qat.bc'))
        record('verify_export')
        hbir_outputs=module.functions[0](*[arrays[k] for k in INPUT_NAMES])
        if isinstance(hbir_outputs,dict):
            hbir_outputs=[hbir_outputs[k] for k in OUTPUT_NAMES]
        report['fake_quant_vs_export_hbir']=compare(fake_outputs,hbir_outputs)
        changed_export_outputs=module.functions[0](*[changed_arrays[k] for k in INPUT_NAMES])
        if isinstance(changed_export_outputs,dict):
            changed_export_outputs=[changed_export_outputs[k] for k in OUTPUT_NAMES]
        report['changed_images_fake_quant_vs_export']=compare(changed_fake_outputs,changed_export_outputs)
        changed_delta=compare(hbir_outputs,changed_export_outputs)
        if not any(x['max_abs_diff']>0 for x in changed_delta):
            raise RuntimeError('Exported outputs did not respond to changed images')
        report['changed_images_output_response']=changed_delta
        np.savez_compressed(args.output/'export_hbir_outputs.npz',**dict(zip(OUTPUT_NAMES,arrays_from_outputs(hbir_outputs))))
        del prepared
        gc.collect()
        if args.device=='cuda': torch.cuda.empty_cache()
        record('convert_nash_p')
        module._integer_conv=True
        converted=compiler.convert(module,march='nash-p',enable_vpu=False)
        compiler.save(converted,str(args.output/'bevfusion.bc'))
        record('verify_quantized_hbir')
        report['export_comparison_floor']=args.min_export_cosine
        for group in ['fake_quant_vs_export_hbir','changed_images_fake_quant_vs_export']:
            for metric in report[group]:
                cosine=metric['cosine_similarity']
                if (cosine is None and metric['max_abs_diff']>1e-4) or (cosine is not None and cosine<args.min_export_cosine):
                    raise RuntimeError(f'Export comparison below threshold: {group}/{metric["name"]}: {cosine}')
        quantized_outputs=converted.functions[0](*[arrays[k] for k in INPUT_NAMES])
        report['fake_quant_vs_quantized_hbir']=compare(fake_outputs,quantized_outputs)
        report['float_vs_quantized_hbir']=compare(float_outputs,quantized_outputs)
        if isinstance(quantized_outputs,dict):
            quantized_outputs=[quantized_outputs[k] for k in OUTPUT_NAMES]
        changed_quantized_outputs=converted.functions[0](*[changed_arrays[k] for k in INPUT_NAMES])
        report['changed_images_fake_quant_vs_quantized']=compare(changed_fake_outputs,changed_quantized_outputs)
        for group in ['fake_quant_vs_quantized_hbir','changed_images_fake_quant_vs_quantized']:
            for metric in report[group]:
                cosine=metric['cosine_similarity']
                if (cosine is None and metric['max_abs_diff']>1e-4) or (cosine is not None and cosine<args.min_export_cosine):
                    raise RuntimeError(f'Converted comparison below threshold: {group}/{metric["name"]}: {cosine}')
        if not any(x['max_abs_diff']>0 for x in compare(quantized_outputs,changed_quantized_outputs)):
            raise RuntimeError('Converted outputs did not respond to changed images')
        from demo import decode
        from verify_hbir import detection_comparison
        float_predictions=decode(float_outputs,cfg)
        quant_predictions=decode(arrays_from_outputs(quantized_outputs),cfg)
        report['task_comparison']={
            'detections':detection_comparison(float_predictions,quant_predictions),
            'occupancy_argmax_agreement':float((float_predictions['occupancy']==quant_predictions['occupancy']).mean()),
            'scope':'Single calibration frame; execution check, not accuracy validation / 单个校准帧执行检查，非精度认证'}
        np.savez_compressed(args.output/'predictions.npz',**quant_predictions)
        from visualize import visualize
        # Rendering is independent of custom NPZ calibration and is opt-in.
        # 绘图为独立可选步骤，自定义 NPZ 不依赖内置样例。
        if args.visualize_samples:
            from demo import prepare
            sensor_inputs,_=prepare(args.visualize_samples,cfg,args.device)
            if not all(np.array_equal(sensor_inputs[k],arrays[k]) for k in INPUT_NAMES):
                raise ValueError('--visualize-samples does not match first calibration input')
            report['visualization']=str(visualize(args.visualize_samples,quant_predictions['boxes_3d'],
                quant_predictions['scores'],quant_predictions['labels'],quant_predictions['occupancy'],
                args.output,threshold=cfg['score_threshold']))
        np.savez_compressed(args.output/'quantized_hbir_outputs.npz',**dict(zip(OUTPUT_NAMES,arrays_from_outputs(quantized_outputs))))
        if args.compile:
            record('compile_hbm')
            compiler.compile(converted,str(args.output/'bevfusion.hbm'),march='nash-p',opt=args.opt,jobs=args.jobs,progress_bar=True)
            from hbdk4.compiler.hbm import Hbm
            hbm=Hbm(str(args.output/'bevfusion.hbm'))
            report['hbm']={'march':hbm.march_name,'bytes':(args.output/'bevfusion.hbm').stat().st_size,
                           'sha256':sha256(args.output/'bevfusion.hbm'),'execution_verified':False}
            if hbm.march_name!='nash-p': raise RuntimeError('Wrong HBM target')
        report['artifacts']={name:sha256(args.output/name) for name in ['qat.bc','bevfusion.bc','calibration-checkpoint.pth.tar']}
        report['checkpoint_unchanged']=sha256(args.checkpoint)==report['checkpoint_sha256']
        report['source_files_unchanged']={str(path.relative_to(ROOT)):source_hashes[str(path.relative_to(ROOT))]==sha256(path)
            for path in source_paths}
        report['calibration_inputs_unchanged']=all(sha256(item['path'])==item['sha256'] for item in report['calibration_inputs'])
        report['config_unchanged']=sha256(args.config)==report['config_sha256']
        report['status']='completed'
        record('completed')
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        (args.output/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        record(report['stage'])
        raise
    print(json.dumps({'status':report['status'],'output':str(args.output.resolve()),
        'downloaded_hbir_used':False,'task_comparison':report.get('task_comparison'),
        'hbm':report.get('hbm'),'report':str(args.output/'export_report.json')},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
