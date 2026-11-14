"""Build the float BEVFusion from original HAT components. / 用原版 HAT 组件装配浮点模型。"""
from __future__ import annotations
from pathlib import Path


def build_float_model(checkpoint, cfg, device='cuda'):
    """Strictly load every tensor; never fall back to partial loading. / 严格完整加载全部张量。"""
    import torch
    from hat.models.structures.bevfusion import BevFusion
    from .camera import build_camera
    from .lidar import build_lidar, build_fusion
    from .decoders import build_decoders
    payload = torch.load(Path(checkpoint), map_location='cpu', weights_only=False)
    state = payload['state_dict']
    options = dict(cfg, is_compile=True)
    model = BevFusion(lidar_network=build_lidar(state, options),
        camera_network=build_camera(state, options),
        bev_decoders=build_decoders(state, options),
        fuse_module=build_fusion(state), bev_h=cfg['bev_h'], bev_w=cfg['bev_w'])
    model.load_state_dict(state, strict=True)
    exact = all(torch.equal(value, model.state_dict()[key]) for key,value in state.items())
    if not exact:
        raise RuntimeError('Loaded model tensors differ from checkpoint')
    fake_quant_modules = [name for name,module in model.named_modules()
                          if 'FakeQuant' in type(module).__name__ or 'Observer' in type(module).__name__]
    if fake_quant_modules:
        raise RuntimeError(f'Float model unexpectedly contains quantizers: {fake_quant_modules}')
    info = {'checkpoint_tensors':len(state), 'model_state_tensors':len(model.state_dict()),
            'strict_load':True, 'all_tensors_equal_checkpoint':exact,
            'fake_quant_modules':fake_quant_modules, 'missing_keys':[], 'unexpected_keys':[],
            'parameters':sum(x.numel() for x in model.parameters()),
            'checkpoint_epoch':payload.get('epoch'),
            'checkpoint_plugin_version':payload.get('horizon-plugin-version'),
            'recovery_unknowns':getattr(model.camera_net, 'recovery_unknowns', []),
            'config_status':cfg['parameter_status']}
    return model.eval().to(device), info


def raw_forward(model, tensors):
    """Use the original BevFusion forward and expose its five raw outputs. / 调用原版 forward 并展开五个输出。"""
    det, occ = model(tensors)
    return tuple(det) + tuple(occ)
