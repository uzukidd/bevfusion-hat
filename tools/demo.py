"""HAT sensor adapter and HBIR demo / HAT 六相机、点云适配与 HBIR 推理。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from oe_assets import ROOT, oe_root, local_path


def load_config(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare(samples, cfg, device='cpu', point_dim=5, fifth_column='ring'):
    import numpy as np
    import torch
    from PIL import Image
    from hat.models.task_modules.centerpoint.pre_process import CenterPointPreProcess
    from hat.models.task_modules.bevformer.view_transformer import BevFormerViewTransformer
    from hat.data.transforms.common import BgrToYuv444

    samples = Path(samples)
    calibration = json.loads((samples/'calibration.json').read_text())
    names = calibration['camera_names']
    order = cfg.get('camera_order', names)
    if len(order) != 6 or set(order) != set(names):
        raise ValueError('camera_order must be a permutation of the calibration camera names')
    matrices, images = [], []
    for name in order:
        i = names.index(name)
        path = samples / calibration.get('image_files', [n+'.png' for n in names])[i]
        with Image.open(path) as source:
            image = source.convert('RGB')
            width, height = image.size
            resize_h, resize_w = cfg['image_resize']
            crop_h, crop_w = cfg['image_crop']
            if resize_h < crop_h or resize_w < crop_w:
                raise ValueError('Image resize must cover crop size')
            left, top = (resize_w-crop_w)//2, resize_h-crop_h
            if cfg.get('image_resize_backend') == 'torchvision_tensor':
                # Match Heal infer.process_img: uint8 CHW tensor resize.
                # 对齐 Heal 推理入口的 uint8 CHW 张量缩放。
                from torchvision.transforms._functional_tensor import resize
                tensor = torch.from_numpy(np.array(image, copy=True)).permute(2,0,1)
                tensor = resize(tensor, [resize_h, resize_w])
                tensor = tensor[:, top:top+crop_h, left:left+crop_w]
            else:
                image = image.resize((resize_w, resize_h), Image.Resampling.BILINEAR)
                image = image.crop((left, top, left+crop_w, top+crop_h))
                tensor = torch.from_numpy(np.array(image, copy=True)).permute(2,0,1)
        if cfg['image_color'] == 'yuv':
            tensor = BgrToYuv444(rgb_input=True)({'img':tensor, 'layout':'chw', 'color_space':'rgb'})['img']
        tensor = (tensor.float()-cfg['image_mean'])/cfg['image_std']
        images.append(tensor)
        k = np.eye(4)
        k[:3,:3] = np.asarray(calibration['camera_intrinsics'][i])
        projection = k @ np.linalg.inv(np.asarray(calibration['camera2lidar'][i]))
        augmentation = np.eye(4)
        augmentation[0,0], augmentation[1,1] = resize_w/width, resize_h/height
        augmentation[0,2], augmentation[1,2] = -left, -top
        matrices.append(augmentation @ projection)
    point_file = samples/'points.bin'
    raw = np.fromfile(point_file, dtype='<f4')
    if raw.size % point_dim:
        raise ValueError('Point cloud size does not match point_dim')
    points = raw.reshape(-1, point_dim).copy()
    if not np.isfinite(points).all():
        raise ValueError('Non-finite point cloud values')
    if point_dim == 4:
        points = np.column_stack((points, np.zeros(len(points), dtype=np.float32)))
    elif fifth_column == 'ring':
        points[:,4] = 0
    tensor = torch.from_numpy(points).to(device)
    preprocessor = CenterPointPreProcess(
        pc_range=cfg['point_cloud_range'], voxel_size=cfg['voxel_size'],
        max_voxels_num=cfg['max_voxels'], max_points_in_voxel=cfg['max_points_in_voxel'],
        norm_range=cfg['norm_range'], norm_dims=cfg['norm_dims']).eval().to(device)
    with torch.inference_mode():
        features, coordinates = preprocessor([tensor], is_deploy=True)
        image_batch = torch.stack(images).to(device)
        transformer = BevFormerViewTransformer(
            pc_range=cfg['point_cloud_range'], num_points_in_pillar=cfg['num_points_in_pillar'],
            bev_h=cfg['bev_h'], bev_w=cfg['bev_w'], embed_dims=cfg['embed_dims'],
            queue_length=1, max_camoverlap_num=cfg['max_camoverlap_num'],
            virtual_bev_h=cfg['virtual_bev_h'], virtual_bev_w=cfg['virtual_bev_w'],
            encoder=None, positional_encoding=None, single_bev=True, use_lidar2img=True).eval().to(device)
        grids = transformer.get_reference_points_cam({
            'img': image_batch, 'seq_meta':[{'lidar2img':np.array(matrices)[None]}]})
    keys = ['queries_rebatch_grid','restore_bev_grid','reference_points_rebatch','bev_pillar_counts']
    result = {'features': features, 'coors': coordinates.to(torch.int32), 'img':image_batch}
    result.update(zip(keys, grids))
    arrays = {k:v.detach().cpu().numpy() for k,v in result.items()}
    metadata = {'camera_order':order, 'point_count':len(points), 'point_dim':point_dim,
                'fifth_column':fifth_column, 'coordinate_frame':'reference_lidar',
                'valid_voxels':int((arrays['coors'][:,1]>=0).sum()),
                'parameter_status':cfg['parameter_status']}
    return arrays, metadata


def decode(outputs, cfg):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from hat.models.task_modules.bevformer.decoder import BEVFormerDetDecoder
    from hat.models.task_modules.bevformer.postprocess import BevFormerProcess
    embed, cls, reference, coords, occupancy = [torch.from_numpy(np.asarray(x)) for x in outputs]
    with torch.inference_mode():
        predictions = BEVFormerDetDecoder.get_outputs(
            SimpleNamespace(pc_range=cfg['point_cloud_range']), [cls], [reference], [coords.clone()], embed)
        # Preserve the actual requested cutoff; HAT's optional threshold adaptively lowers itself.
        post = BevFormerProcess(pc_range=cfg['point_cloud_range'],
            post_center_range=cfg['post_center_range'], max_num=cfg['max_detections'], num_classes=10)
        det = post(predictions)[0]
        det = det[det[:,-2] >= cfg['score_threshold']]
        det[:,2] -= det[:,5] / 2  # HAT center-z -> explicit bottom-z for visualization.
        occ = occupancy.argmax(-1).to(torch.uint8).numpy()[0]
    return {'boxes_3d':det[:,:9].numpy(), 'scores':det[:,9].numpy(), 'labels':det[:,10].numpy().astype(np.int64),
            'occupancy':occ}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=local_path,default=ROOT/'data/samples')
    parser.add_argument('--config',type=local_path,default=ROOT/'config/bevfusion_hbir.json')
    parser.add_argument('--model',type=local_path,default=ROOT/'outputs/s600/bevfusion.bc')
    parser.add_argument('--output',type=local_path,default=ROOT/'outputs/demo')
    parser.add_argument('--oe-root')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--point-dim',type=int,choices=[4,5],default=5)
    parser.add_argument('--fifth-column',choices=['ring','time-lag'],default='ring')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--no-visualization',action='store_true')
    args = parser.parse_args()
    sys.path.insert(0,str(oe_root(args.oe_root)/'package/host/ai_toolchain'))
    import torch
    torch.set_num_threads(8)
    cfg = load_config(args.config)
    args.output.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    arrays, meta = prepare(args.samples, cfg, args.device, args.point_dim, args.fifth_column)
    import numpy as np
    np.savez(args.output/'inputs.npz',**arrays)
    meta['preprocess_seconds'] = time.monotonic()-start
    print('Inputs:',{k:(v.shape,str(v.dtype)) for k,v in arrays.items()},flush=True)
    print('Metadata:',meta,flush=True)
    (args.output/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    if args.prepare_only:
        (args.output/'summary.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
        return
    from hbdk4 import compiler
    module = compiler.load(str(args.model))
    function = module.functions[0]
    for item in function.inputs:
        if tuple(arrays[item.name].shape) != tuple(item.type.shape):
            raise ValueError(f'{item.name}: input shape {arrays[item.name].shape} != model {item.type.shape}')
    print('Running HBIR / 正在执行 HBIR ...',flush=True)
    start = time.monotonic()
    outputs = function(*[arrays[i.name] for i in function.inputs])
    elapsed = time.monotonic()-start
    if isinstance(outputs,dict):
        outputs = [outputs[o.name] for o in function.outputs]
    if len(outputs) != 5 or not all(np.isfinite(np.asarray(x)).all() for x in outputs):
        raise RuntimeError('Invalid or non-finite raw HBIR outputs')
    predictions = decode(outputs,cfg)
    if not all(np.isfinite(x).all() for x in predictions.values()):
        raise RuntimeError('Model returned non-finite predictions')
    np.savez_compressed(args.output/'predictions.npz',**predictions)
    meta.update(model=str(args.model.resolve()), inference_seconds=elapsed,
                detections=len(predictions['scores']),
                score_max=float(predictions['scores'].max()) if len(predictions['scores']) else None,
                occupancy_shape=list(predictions['occupancy'].shape),
                occupancy_classes=np.unique(predictions['occupancy']).tolist(),
                output_shapes=[list(np.asarray(x).shape) for x in outputs],
                validation='Execution/finite-output check only; original external preprocessing parameters not certified / 已验证执行与输出有限值，原始图外预处理参数尚未确证')
    if not args.no_visualization:
        from visualize import visualize
        meta['visualization']=str(visualize(args.samples,predictions['boxes_3d'],predictions['scores'],
             predictions['labels'],predictions['occupancy'],args.output,threshold=cfg['score_threshold'],point_dim=args.point_dim))
    (args.output/'summary.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
