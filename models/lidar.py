"""Assemble the original HAT LiDAR branch and fusion block. / 装配原版 HAT 点云与融合模块。"""
from __future__ import annotations


def build_lidar(state_dict, cfg):
    """Recover layer sizes from checkpoint and strides from HBIR. / 按权重和 HBIR 恢复层参数。"""
    from hat.models.structures.detectors.pointpillars import PointPillarsDetector
    from hat.models.task_modules.lidar.pillar_encoder import PillarFeatureNet, PointPillarScatter
    from hat.models.necks.second_neck import SECONDNeck

    prefix = 'lidar_net.'
    reader_channels = state_dict[prefix+'reader.pfn_layers.0.linear.weight'].shape[0]
    reader = PillarFeatureNet(
        num_input_features=5, num_filters=(reader_channels,),
        with_distance=False, voxel_size=cfg['voxel_size'], pc_range=cfg['point_cloud_range'],
        bn_kwargs={'eps':1e-3, 'momentum':0.01}, quantize=True,
        use_4dim=True, use_conv=True, pool_size=(cfg['max_points_in_voxel'],1), hw_reverse=True)
    grid = [round((cfg['point_cloud_range'][i+3]-cfg['point_cloud_range'][i])/cfg['voxel_size'][i])
            for i in range(3)]
    if grid != [512,512,1]:
        raise ValueError(f'Checkpoint/HBIR expects grid [512,512,1], got {grid}')
    scatter = PointPillarScatter(num_input_features=reader_channels,
                                use_horizon_pillar_scatter=True, quantize=True)
    neck = SECONDNeck(in_feature_channel=reader_channels, down_layer_nums=[3,5,5],
        down_layer_strides=[2,2,2], down_layer_channels=[64,128,256],
        up_layer_strides=[0.5,1,2], up_layer_channels=[128,128,128],
        bn_kwargs={'eps':1e-3,'momentum':0.01}, use_relu6=False, quantize=False)
    # Preprocessed features/coors are shared with HBIR to isolate network differences.
    # 与 HBIR 共享已生成的体素输入，避免预处理差异影响网络对比。
    model = PointPillarsDetector(feature_map_shape=grid, pre_process=None,
                                 reader=reader, backbone=scatter, neck=neck)
    expected = {k[len(prefix):]:v for k,v in state_dict.items() if k.startswith(prefix)}
    model.load_state_dict(expected, strict=True)
    return model


def build_fusion(state_dict):
    """Reuse the original convolution and SE fusion. / 复用原版卷积与 SE 融合。"""
    from hat.models.task_modules.lidar_fusion.fusion_module import BevFuseModule
    weight = state_dict['fuse_module.reduce_conv.0.weight']
    model = BevFuseModule(input_c=weight.shape[1],fuse_c=weight.shape[0])
    prefix='fuse_module.'
    model.load_state_dict({k[len(prefix):]:v for k,v in state_dict.items() if k.startswith(prefix)},strict=True)
    return model
