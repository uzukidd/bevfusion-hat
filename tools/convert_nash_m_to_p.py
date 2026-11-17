"""Retarget a Nash-m model using its pre-lowering HBIR. / 使用 Nash-m 模型的定点转换前 HBIR 转为 Nash-p。"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import traceback
from oe_assets import ROOT, local_path, sha256


def attributes(module):
    """Read portable root attributes. / 读取模型根属性。"""
    return {x.name:str(x.attr) for x in module.module.operation.attributes}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=local_path,required=True,help='Nash-m .bc or .hbm for identity/metadata / 原Nash-m模型')
    parser.add_argument('--source-hbir',type=local_path,required=True,
        help='Corresponding pre-lowering HBIR (e.g. qat.bc); backend .bc/.hbm cannot be directly retargeted / 对应的定点转换前HBIR')
    parser.add_argument('--format',choices=['hbir','hbm'],default='hbir')
    parser.add_argument('--output',type=local_path,default=ROOT/'outputs/nash_m_to_p')
    parser.add_argument('--opt',type=int,choices=[0,1,2],default=1)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--inputs',type=local_path,nargs='+',help='Optional prepared NPZ cases to compare source Nash-m vs converted Nash-p / 可选同输入比较')
    parser.add_argument('--config',type=local_path,default=ROOT/'config/bevfusion_hbir.json')
    args=parser.parse_args()
    if args.jobs<1:
        parser.error('--jobs must be positive')
    if args.source_hbir.suffix.lower()!='.bc':
        parser.error('--source-hbir must be pre-lowering .bc')
    if args.input.resolve()==args.source_hbir.resolve():
        parser.error('Nash-m backend and pre-lowering source must be distinct files')
    if any((args.output/name).exists() for name in ['bevfusion.bc','bevfusion.hbm']):
        parser.error('Output contains model artifacts; choose a new --output')
    from hbdk4 import compiler
    original=None
    if args.input.suffix.lower()=='.bc':
        original=compiler.load(str(args.input))
        original_info=attributes(original)
        march=original_info.get('hbdk.target','').strip('"')
    elif args.input.suffix.lower()=='.hbm':
        from hbdk4.compiler.hbm import Hbm
        hbm=Hbm(str(args.input)); march=hbm.march_name
        original_info={'march':march,'graphs':[g.name for g in hbm.graphs]}
    else:
        parser.error('--input must be .bc or .hbm')
    if march!='nash-m':
        parser.error(f'--input target must be nash-m, got {march!r}')
    source=compiler.load(str(args.source_hbir))
    source_attrs=attributes(source)
    if 'hbdk.target' in source_attrs:
        parser.error('--source-hbir is already lowered; use its pre-lowering qat.bc, not quantized.bc')
    # Reject backend dialects even when a manually modified file lacks its target attribute.
    dialects=set()
    def visit(operation):
        name=operation.name
        if name.startswith(('b30.','b25.','bpu.')):
            dialects.add(name.split('.')[0])
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    visit(child.operation)
    visit(source.module.operation)
    if dialects:
        parser.error(f'--source-hbir contains backend dialects: {sorted(dialects)}')
    if original is not None:
        a,b=original.functions[0],source.functions[0]
        if len(original.functions)!=1 or len(source.functions)!=1:
            parser.error('This task requires one BEVFusion graph')
        if [(x.name,tuple(x.type.shape)) for x in a.inputs]!=[(x.name,tuple(x.type.shape)) for x in b.inputs]:
            parser.error('Nash-m model and source HBIR input interfaces differ')
        if [tuple(x.type.shape) for x in a.outputs]!=[tuple(x.type.shape) for x in b.outputs]:
            parser.error('Nash-m model and source HBIR output shapes differ')
    args.output.mkdir(parents=True,exist_ok=True)
    report={'status':'running','input':str(args.input.resolve()),'input_sha256':sha256(args.input),
        'input_metadata':original_info,'source_hbir':str(args.source_hbir.resolve()),
        'source_sha256':sha256(args.source_hbir),'actual_conversion_input':str(args.source_hbir.resolve()),
        'source_correspondence':'User supplied paired source; interface checked where possible, weight identity not proven / 用户提供配对源，已按可用接口检查，未证明权重完全同源',
        'target':'nash-p','format':args.format,'compiler_version':importlib.metadata.version('hbdk4-compiler'),
        'integer_conv':True,'enable_vpu':False,'opt':args.opt,'jobs':args.jobs,
        'note':'Backend Nash-m IR is not directly rewritten; conversion restarts from its source HBIR / 不直接改写Nash-m后端图，使用其源HBIR重新转换'}
    started=time.monotonic()
    try:
        print('Converting supplied source HBIR -> nash-p',flush=True)
        source._integer_conv=True
        converted=compiler.convert(source,march='nash-p',enable_vpu=False)
        output=args.output/'bevfusion.bc'
        compiler.save(converted,str(output))
        report['output_hbir']={'path':str(output.resolve()),'sha256':sha256(output),'attributes':attributes(converted)}
        if args.format=='hbm':
            print('Compiling nash-p HBM',flush=True)
            compiler.compile(converted,str(args.output/'bevfusion.hbm'),march='nash-p',opt=args.opt,jobs=args.jobs,progress_bar=True)
            from hbdk4.compiler.hbm import Hbm
            model=Hbm(str(args.output/'bevfusion.hbm'))
            if model.march_name!='nash-p':
                raise RuntimeError('Compiled HBM target mismatch')
            report['output_hbm']={'path':str((args.output/'bevfusion.hbm').resolve()),'march':model.march_name,
                'bytes':(args.output/'bevfusion.hbm').stat().st_size,'sha256':sha256(args.output/'bevfusion.hbm'),
                'execution_verified':False}
        if args.inputs:
            # Use the common runner without claiming Nash-m is itself a Nash-p reference.
            from compare_nash_p import compare_models
            report['comparison']=compare_models(args.input,output,args.inputs,args.config,
                args.output/'comparison',expected_reference_target='nash-m')
            if report['comparison'].get('status')!='completed':
                raise RuntimeError('Converted model comparison failed; inspect comparison report')
        report['status']='completed'
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        (args.output/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        (args.output/'conversion.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
