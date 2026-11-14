"""Restore the original HAT BEV detection/occupancy decoders.

中文：恢复原版 HAT BEV 检测与占用解码器。
"""
from __future__ import annotations


def build_decoders(state_dict, cfg):
    """Build and strictly load the two original HAT decoder modules.

    English (中文说明见下文): ``state_dict`` contains full-model keys starting
    with ``bev_decoders.``. Return an ``nn.ModuleList`` in detection, occupancy
    order. Call ``eval()`` for inference. ``cfg['is_compile']`` defaults to True;
    this selects the original raw-output path without quantizing the model or
    folding attention weights into deploy form.

    Raw usage::

        decoders = build_decoders(state_dict, cfg).eval()
        det = decoders[0](bev_feature, data)
        occ = decoders[1](bev_feature, data)
        raw_outputs = tuple(det) + tuple(occ)

    ``bev_feature`` is [B,256,128,128]. The five raw tensors have shapes
    [B,16384,256], [B,900,10], [B,900,3], [B,900,10],
    [B,200,200,16,18]. Detection tensors are BEV features, class logits,
    reference logits and box deltas. All six transformer layers execute; only
    their final predictions are returned. ``data=None`` works in raw mode.

    With ``is_compile=False``, detection returns the original prediction dict
    (all_cls_scores/all_bbox_preds contain six layers; BEV is sequence-first).
    Occupancy returns a one-element list of argmax labels [B,200,200,16] and
    requires ``data={'seq_meta': [{}]}`` even in eval mode. No training losses or
    assignment configuration can be recovered from tensor shapes; criterion
    is retained solely to preserve its checkpoint parameter. Dropout defaults
    to HAT's 0.1 (configurable via decoder_dropout) and has no effect in eval.
    The physical point-cloud range is supplied by cfg and remains provisional.

    中文：state_dict 使用完整模型的 bev_decoders. 前缀；返回检测、占用顺序的
    ModuleList，并严格加载全部参数。推理前调用 eval()。is_compile 默认 True，
    仅选择原版 raw 输出分支，不进行量化，也不调用会改写注意力权重的
    switch_to_deploy。输入为 [B,256,128,128]，按上述示例拼接即 HBIR 五个输出。
    检测的六层均执行，仅返回最后层的分类 logits、参考点 logits 和框增量。
    raw 模式 data 可为 None。is_compile=False 时检测返回六层预测字典，占用
    返回单元素标签列表，且 eval 仍须传入 {'seq_meta': [{}]}。
    损失与匹配器配置无法从权重推断；保留原版 criterion 以完整加载参数，
    不提供训练恢复保证。dropout 默认原版 0.1，可用 decoder_dropout 覆盖，
    不影响 eval；物理点云范围沿用 cfg，仍待原始配置核实。
    """
    from collections import OrderedDict

    from torch import nn

    from hat.models.base_modules.roi_feat_extractors import RoiResize
    from hat.models.task_modules.bevformer.attention import (
        HorizonMultiScaleDeformableAttention,
    )
    from hat.models.task_modules.bevformer.criterion import BevFormerCriterion
    from hat.models.task_modules.bevformer.decoder import (
        BEVFormerDetDecoder,
        DetectionTransformerDecoder,
        DetrTransformerDecoderLayer,
    )
    from hat.models.task_modules.flashocc.bev_occ_head import BEVOCCHead2D
    from hat.models.task_modules.flashocc.decoder import BevformerOccDetDecoder

    prefix = 'bev_decoders.'
    weights = OrderedDict(
        (key[len(prefix):], value)
        for key, value in state_dict.items()
        if key.startswith(prefix)
    )
    metadata = getattr(state_dict, '_metadata', None)
    if metadata is not None:
        weights._metadata = {
            key[len(prefix):]: value
            for key, value in metadata.items()
            if key.startswith(prefix)
        }
        if 'bev_decoders' in metadata:
            weights._metadata[''] = metadata['bev_decoders']
    embed_dims = weights['0.reference_points.weight'].shape[1]
    num_queries = weights['0.query_embedding.weight'].shape[0]
    num_classes = weights['0.cls_branches.0.6.weight'].shape[0]
    code_size = weights['0.reg_branches.0.4.weight'].shape[0]
    layer_indices = {
        int(key.split('.')[3])
        for key in weights
        if key.startswith('0.decoder.layers.')
    }
    num_layers = len(layer_indices)
    if layer_indices != set(range(num_layers)) or num_layers != 6:
        raise ValueError(f'Expected six contiguous decoder layers, got {layer_indices}')
    bev_h, bev_w = cfg['bev_h'], cfg['bev_w']
    if (bev_h, bev_w, embed_dims, num_queries, num_classes, code_size) != (
        128, 128, 256, 900, 10, 10
    ):
        raise ValueError('Decoder dimensions do not match the target checkpoint/HBIR')
    dropout = cfg.get('decoder_dropout', 0.1)
    is_compile = cfg.get('is_compile', True)
    pc_range = cfg['point_cloud_range']

    # HBIR verifies 8 heads, 1 BEV level, 4 sample points and grid alignment 4.
    # HBIR 已核实 8 个头、1 层 BEV 特征、4 个采样点、网格对齐数 4。
    cross_attention = HorizonMultiScaleDeformableAttention(
        embed_dims=embed_dims,
        num_heads=8,
        num_levels=1,
        num_points=4,
        dropout=dropout,
        batch_first=False,
        grid_align_num=4,
        feats_size=((bev_w, bev_h),),
    )
    layer = DetrTransformerDecoderLayer(
        crossattention=cross_attention,
        embed_dims=embed_dims,
        dropout=dropout,
        num_heads=8,
    )
    transformer = DetectionTransformerDecoder(
        num_layers=num_layers,
        decoder_layer=layer,
        return_intermediate=True,
    )
    criterion = BevFormerCriterion(
        num_classes=num_classes,
        pc_range=pc_range,
        code_weights=weights['0.criterion.code_weights'].detach().cpu().tolist(),
    )
    detection = BEVFormerDetDecoder(
        decoder=transformer,
        embed_dims=embed_dims,
        num_reg_fcs=2,
        num_pred=num_layers,
        bev_h=bev_h,
        bev_w=bev_w,
        code_size=code_size,
        num_query=num_queries,
        num_classes=num_classes,
        pc_range=pc_range,
        is_compile=is_compile,
        criterion=criterion,
        group_detr=1,
    )

    # HBIR crops indices [14:114,14:114], then upsamples the head to 200x200.
    # HBIR 先裁剪 [14:114,14:114]，再由占用头上采样到 200x200。
    roi_resizer = RoiResize(
        in_strides=[1],
        roi_resize_cfgs={
            'in_stride': 1,
            'roi_box': [14, 14, 114, 114],
            'roi_feat_name': 'roi_feat0',
        },
    )
    occupancy = BevformerOccDetDecoder(
        occ_head=BEVOCCHead2D(
            in_dim=embed_dims,
            out_dim=weights['1.occ_head.final_conv.0.weight'].shape[0],
            Dz=16,
            num_classes=18,
            use_predicter=True,
            use_upsample=True,
        ),
        roi_resizer=roi_resizer,
        num_classes=18,
        is_compile=is_compile,
    )
    decoders = nn.ModuleList([detection, occupancy])
    decoders.load_state_dict(weights, strict=True)
    return decoders
