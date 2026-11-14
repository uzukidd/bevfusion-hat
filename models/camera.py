"""Restore the camera branch with the original HAT modules.

使用原版 HAT 组件恢复相机分支；HAT/OE 路径及运行环境由调用方配置。
"""

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

__all__ = ["build_camera"]


def _camera_state(state_dict: Mapping[str, Any]) -> OrderedDict[str, torch.Tensor]:
    """Select a complete camera state, accepting checkpoint or branch keys.

    提取完整相机权重，接受 checkpoint、全模型键或已去前缀的分支键。
    """
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], Mapping):
        state_dict = state_dict["state_dict"]
    state = OrderedDict(
        (key.removeprefix("module."), value)
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    )
    full_model = any(key.startswith("camera_net.") for key in state)
    if full_model:
        state = OrderedDict(
            (key.removeprefix("camera_net."), value)
            for key, value in state.items()
            if key.startswith("camera_net.")
        )
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        state._metadata = OrderedDict()
        for key, value in metadata.items():
            key = key.removeprefix("module.")
            if full_model:
                if key == "camera_net":
                    key = ""
                elif key.startswith("camera_net."):
                    key = key.removeprefix("camera_net.")
                else:
                    continue
            state._metadata[key] = value
    if not state or "backbone.patch_embed.0.0.weight" not in state:
        raise ValueError("No complete camera state found / 未找到完整相机权重")
    return state


def build_camera(state_dict: Mapping[str, Any], cfg: Mapping[str, Any]) -> nn.Module:
    """Build and strictly load the original HAT camera branch.

    构建并严格加载原版 HAT 相机分支。返回 BevFormer，供 BevFusion 调用
    extract_feat 和 view_transformer；该容器自身的 forward 返回 decoder 结果，
    因本分支无 decoder，调用方应沿用原版 BevFusion.forward_camera_feature。

    ``cfg`` uses the existing BEV/image fields and optional ``camera`` overrides:
    ``dropout``, ``act_layer``, ``bn_eps``, ``bn_momentum``, ``use_lidar2img``.
    ``is_compile`` selects the original four-reference-tensor input path.
    Evidence gaps are exposed in ``recovery_unknowns`` without adding state keys.

    cfg 沿用现有 BEV/图像字段，可在 camera 子字典覆盖 dropout、act_layer、
    bn_eps、bn_momentum、use_lidar2img；is_compile 选择原版四参考张量路径。
    recovery_unknowns 显式列出恢复证据不足的项，不增加 state_dict 键。
    """
    from hat.models.backbones.henet import HENet
    from hat.models.base_modules.position_encoding import LearnedPositionalEncoding
    from hat.models.necks.fpn import FPN
    from hat.models.structures.detectors.bevformer import BevFormer
    from hat.models.task_modules.bevformer.attention import (
        HorizonMultiScaleDeformableAttention,
        HorizonMultiScaleDeformableAttention3D,
        HorizonSpatialCrossAttention,
    )
    from hat.models.task_modules.bevformer.encoder import (
        SingleBEVFormerEncoder,
        SingleBEVFormerEncoderLayer,
    )
    from hat.models.task_modules.bevformer.view_transformer import (
        SingleBevFormerViewTransformer,
    )

    state = _camera_state(state_dict)
    options = dict(cfg.get("camera", {}))
    bev_h, bev_w = int(cfg["bev_h"]), int(cfg["bev_w"])
    embed_dims = int(cfg["embed_dims"])
    num_cams = int(state["view_transformer.encoder.cams_embeds"].shape[0])
    num_levels = int(state["view_transformer.encoder.level_embeds"].shape[0])
    if num_levels != 1:
        raise ValueError("This camera checkpoint requires one feature level / 相机仅支持一层特征")
    overlap = int(cfg["max_camoverlap_num"])
    dropout = float(options.get("dropout", cfg.get("dropout", 0.1)))
    act_layer = tuple(options.get("act_layer", ("nn.GELU",) * 4))
    if len(act_layer) != 4:
        raise ValueError("camera.act_layer must contain four entries / 激活配置须为四项")
    bn_eps = float(options.get("bn_eps", cfg.get("bn_eps", 1e-5)))
    bn_momentum = float(options.get("bn_momentum", cfg.get("bn_momentum", 0.1)))
    image_h, image_w = cfg.get("image_crop", (512, 960))
    feature_size = ((int(image_w) + 31) // 32, (int(image_h) + 31) // 32)

    backbone = HENet(
        in_channels=3,
        block_nums=(4, 3, 8, 6),
        embed_dims=(64, 128, 192, 384),
        attention_block_num=(0, 0, 0, 0),
        mlp_ratios=(2, 2, 2, 3),
        act_layer=act_layer,
        use_layer_scale=(True, True, True, True),
        include_top=False,
        extra_act=(False, False, False, False),
        final_expand_channel=0,
        feature_mix_channel=1024,
        block_cls=("GroupDWCB", "GroupDWCB", "AltDWCB", "DWCB"),
        down_cls=("S2DDown", "S2DDown", "S2DDown", "None"),
        patch_embed="origin",
        quant_input=True,
        dequant_output=True,
        stage_out_norm=True,
    )
    neck = FPN(
        in_strides=[32],
        in_channels=[384],
        out_strides=[32],
        out_channels=[embed_dims],
        bn_kwargs={"eps": bn_eps, "momentum": bn_momentum},
    )
    # Shapes fix grid alignment and points; HBIR confirms eight spatial heads.
    # 权重固定网格对齐和采样点数；HBIR 确认空间注意力为八头。
    self_prefix = "view_transformer.encoder.layers.0.sa."
    cross_prefix = "view_transformer.encoder.layers.0.ca.deformable_attention."
    self_grid = int(state[self_prefix + "reduce_sum.weight"].shape[0])
    cross_grid = int(state[cross_prefix + "reduce_sum.weight"].shape[0])
    self_points = int(state[self_prefix + "reduce_sum.weight"].shape[1]) // self_grid
    cross_points = int(state[cross_prefix + "reduce_sum.weight"].shape[1]) // cross_grid
    self_heads = int(state[self_prefix + "attention_weights.weight"].shape[0]) // self_points
    cross_heads = int(state[cross_prefix + "attention_weights.weight"].shape[0]) // cross_points
    self_attention = HorizonMultiScaleDeformableAttention(
        embed_dims=embed_dims,
        num_heads=self_heads,
        num_levels=num_levels,
        num_points=self_points,
        dropout=dropout,
        batch_first=True,
        grid_align_num=self_grid,
        feats_size=((bev_w, bev_h),),
    )
    spatial_attention = HorizonMultiScaleDeformableAttention3D(
        embed_dims=embed_dims,
        num_heads=cross_heads,
        num_levels=num_levels,
        num_points=cross_points,
        grid_align_num=cross_grid,
        feats_size=(feature_size,),
    )
    cross_attention = HorizonSpatialCrossAttention(
        bev_h=bev_h,
        bev_w=bev_w,
        deformable_attention=spatial_attention,
        embed_dims=embed_dims,
        num_cams=num_cams,
        dropout=dropout,
        max_camoverlap_num=overlap,
    )
    encoder_layer = SingleBEVFormerEncoderLayer(
        selfattention=self_attention,
        crossattention=cross_attention,
        embed_dims=embed_dims,
        dropout=dropout,
    )
    encoder = SingleBEVFormerEncoder(
        bev_h=bev_h,
        bev_w=bev_w,
        embed_dims=embed_dims,
        encoder_layer=encoder_layer,
        use_cams_embeds=True,
        num_feature_levels=num_levels,
        num_cams=num_cams,
        num_layers=1,
        return_intermediate=False,
    )
    position_prefix = "view_transformer.positional_encoding."
    row_shape = state[position_prefix + "row_embed.weight"].shape
    col_shape = state[position_prefix + "col_embed.weight"].shape
    view_transformer = SingleBevFormerViewTransformer(
        pc_range=list(cfg["point_cloud_range"]),
        num_points_in_pillar=int(cfg["num_points_in_pillar"]),
        bev_h=bev_h,
        bev_w=bev_w,
        embed_dims=embed_dims,
        queue_length=1,
        max_camoverlap_num=overlap,
        virtual_bev_h=int(cfg["virtual_bev_h"]),
        virtual_bev_w=int(cfg["virtual_bev_w"]),
        encoder=encoder,
        positional_encoding=LearnedPositionalEncoding(
            num_feats=int(row_shape[1]),
            row_num_embed=int(row_shape[0]),
            col_num_embed=int(col_shape[0]),
        ),
        numcam=num_cams,
        in_indices=(0,),
        is_compile=bool(cfg.get("is_compile", False)),
        single_bev=True,
        use_lidar2img=bool(options.get("use_lidar2img", cfg.get("use_lidar2img", True))),
    )
    camera = BevFormer(
        backbone=backbone,
        neck=neck,
        view_transformer=view_transformer,
        out_indices=(4,),
        bev_decoders=[],
    )
    for module in camera.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eps = bn_eps
            module.momentum = bn_momentum

    expected = camera.state_dict()
    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    mismatched = {
        key: (tuple(expected[key].shape), tuple(state[key].shape))
        for key in expected.keys() & state.keys()
        if expected[key].shape != state[key].shape
    }
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Camera checkpoint mismatch / 相机权重不匹配: "
            f"missing={missing}, unexpected={unexpected}, shapes={mismatched}"
        )
    camera.load_state_dict(state, strict=True)
    camera.recovery_unknowns = {
        "point_cloud_range": "confirmed by Heal target common.py; old checkpoint training config is not archived / 已由Heal目标配置确认，旧权重训练配置未存档",
        "projection": "use_lidar2img=True confirmed by Heal; sample calibration is camera-to-reference-LiDAR / Heal确认使用lidar2img，样例标定为相机到参考雷达",
        "preprocessing": "Heal resize/YUV/normalization are used; single-frame sample lacks the configured nine historical LiDAR sweeps / 已采用Heal缩放与YUV归一化，单帧样例缺少配置的九帧历史点云",
        "bn_eps_momentum": "stage_norm.3 eps=1e-5 is explicit in QAT HBIR; remaining BN use HAT defaults and training momentum is not recoverable / QAT图明确stage_norm.3 eps=1e-5，其余采用原版默认值，训练momentum无法确定",
        "dropout": "training probability is absent from inference graph / 推理图无法确定训练丢弃率",
        "activation": "GELU confirmed by Heal target model.py; old fused LUT provenance remains separate / Heal目标配置确认GELU，旧版融合LUT来源另行保留",
        "feature_selection": "final stride-32 feature is verified; equivalent unused input-index lists are not recoverable / 已核实最终32倍特征，等效的未使用输入索引列表无法区分",
    }
    camera.recovery_graph_unused_state_keys = tuple(
        key for key in state
        if key.startswith(("backbone.norm.", "backbone.feature_mix_layer.", "backbone.stage_norm.0.", "backbone.stage_norm.1.", "backbone.stage_norm.2."))
    )
    return camera
