"""Render BEVFusion camera / LiDAR / occupancy predictions. / 绘制多传感器预测结果。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


CLASS_NAMES = (
    "car", "truck", "trailer", "bus", "construction_vehicle", "bicycle",
    "motorcycle", "pedestrian", "traffic_cone", "barrier",
)
# Occ3D-nuScenes order, independent of the detection class IDs.
# Occ3D-nuScenes 类别顺序，与检测任务的类别 ID 独立。
# https://github.com/Yzichen/FlashOCC/blob/master/projects/mmdet3d_plugin/core/evaluation/occ_metrics.py
OCCUPANCY_CLASS_NAMES = (
    "others", "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade",
    "vegetation", "free",
)
_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
          (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def _corners(box):
    """Return corners for [x,y,bottom_z,dx,dy,dz,yaw,vx,vy]. / 从底面框生成顶点。"""
    x, y, z, dx, dy, dz, yaw = box[:7]
    local = np.array([
        [-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0],
        [-.5, -.5, 1], [.5, -.5, 1], [.5, .5, 1], [-.5, .5, 1],
    ]) * np.array([dx, dy, dz])
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return local @ rotation.T + np.array([x, y, z])


def _project_segment(a, b, intrinsic, near=.1):
    """Clip a 3D camera segment to positive depth before projection. / 投影前裁剪近面。"""
    a, b = a.copy(), b.copy()
    if a[2] < near and b[2] < near:
        return None
    if a[2] < near:
        a += (b - a) * ((near - a[2]) / (b[2] - a[2]))
    if b[2] < near:
        b += (a - b) * ((near - b[2]) / (a[2] - b[2]))
    pixels = np.stack([a, b]) @ intrinsic.T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    return pixels if np.isfinite(pixels).all() else None


def _load_points(samples_dir, point_dim=5):
    """Load float32 points with an explicit record width. / 按明确列数读取点云。"""
    raw = np.fromfile(samples_dir / "points.bin", dtype=np.float32)
    if point_dim not in (4, 5) or raw.size % point_dim:
        raise ValueError('points.bin size does not match explicit point_dim=4 or 5')
    points = raw.reshape(-1, point_dim)
    return points[np.isfinite(points[:, :3]).all(axis=1), :3]


def visualize(samples_dir, boxes, scores, labels, occupancy, output_dir, threshold=.3, point_dim=5):
    """Save camera boxes, LiDAR BEV and occupancy as demo.png. / 保存预测汇总图。

    boxes: Nx9 LiDAR boxes [x,y,bottom_z,dx,dy,dz,yaw,vx,vy], metres/radians.
        输入检测框为 LiDAR 坐标，z 表示底面，尺寸沿局部 x/y/z 轴。
    Calibration uses six camera_intrinsics (3x3), camera2lidar (4x4),
        and camera_names in the same order as the image files. / 标定与图片顺序须一致。
    occupancy: integer [200,200,16], axes (x,y,z); class 17 is free.
        顶视图显示每个 xy 位置最高的非空体素；坐标为网格索引，不假设物理范围。
    Predictions are model output, no ground truth. / 图中仅为模型预测，无真值。
    Returns the absolute PNG Path. / 返回 PNG 的绝对路径。
    """
    samples_dir, output_dir = Path(samples_dir), Path(output_dir)
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        boxes = np.empty((0, 9), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 9:
        raise ValueError("boxes must have shape (N,9)")
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    raw_labels = np.asarray(labels).reshape(-1)
    if len(boxes) != len(scores) or len(boxes) != len(raw_labels):
        raise ValueError("boxes, scores and labels must have equal lengths")
    if not np.isfinite(raw_labels).all() or np.any(raw_labels != np.floor(raw_labels)):
        raise ValueError("labels must be finite integer NuScenes class ids")
    labels = raw_labels.astype(np.int64)
    if np.any((labels < 0) | (labels >= len(CLASS_NAMES))):
        raise ValueError("detection labels must be in [0,9]")
    occ = np.asarray(occupancy)
    if occ.shape != (200, 200, 16):
        raise ValueError("occupancy must have shape (200,200,16)")
    if not np.isfinite(occ).all() or np.any(occ != np.floor(occ)) or np.any((occ < 0) | (occ > 17)):
        raise ValueError("occupancy must contain integer class ids in [0,17]")
    occ = occ.astype(np.int16)
    with (samples_dir / "calibration.json").open(encoding="utf-8") as source:
        calibration = json.load(source)
    names = calibration["camera_names"]
    intrinsics = np.asarray(calibration["camera_intrinsics"], dtype=np.float64)
    camera2lidar = np.asarray(calibration["camera2lidar"], dtype=np.float64)
    if len(names) != 6 or intrinsics.shape != (6, 3, 3) or camera2lidar.shape != (6, 4, 4):
        raise ValueError("calibration must describe six cameras, with 3x3 K and 4x4 camera2lidar")
    keep = (scores >= threshold) & np.isfinite(scores) & np.isfinite(boxes).all(axis=1)
    keep &= (boxes[:, 3:6] > 0).all(axis=1)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    corners = [_corners(box) for box in boxes]
    colors = colormaps["tab10"](np.arange(10))

    figure = Figure(figsize=(18, 15), dpi=150, facecolor="#111827")
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(4, 3, height_ratios=[1, 1, 1.30, .65],
                             left=.035, right=.985, bottom=.055, top=.925,
                             hspace=.20, wspace=.10)
    figure.suptitle("BEVFusion | six cameras + LiDAR + occupancy", color="white", fontsize=21, y=.978)
    figure.text(.5, .947, f"Predictions are model output, no GT  |  score >= {threshold:g}  |  {len(boxes)} boxes",
                color="#cbd5e1", ha="center", fontsize=12)
    for index, name in enumerate(names):
        axis = figure.add_subplot(grid[index // 3, index % 3])
        image_path = samples_dir / f"{name}.png"
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"))
        height, width = image.shape[:2]
        axis.imshow(image)
        lidar2camera = np.linalg.inv(camera2lidar[index])
        for box_corners, score, label in zip(corners, scores, labels):
            camera_points = box_corners @ lidar2camera[:3, :3].T + lidar2camera[:3, 3]
            color = colors[label]
            for start, end in _EDGES:
                pixels = _project_segment(camera_points[start], camera_points[end], intrinsics[index])
                if pixels is not None:
                    axis.plot(pixels[:, 0], pixels[:, 1], color=color, linewidth=1.4, clip_on=True)
            positive = camera_points[:, 2] > .1
            if positive.any():
                pixels = camera_points[positive] @ intrinsics[index].T
                pixels = pixels[:, :2] / pixels[:, 2:3]
                visible = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width)
                           & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
                if visible.any():
                    anchor = pixels[visible][np.argmin(pixels[visible, 1])]
                    axis.text(*anchor, f"{CLASS_NAMES[label]} {score:.2f}", color="white", fontsize=6,
                              bbox={"facecolor": color, "alpha": .8, "pad": 1, "edgecolor": "none"}, clip_on=True)
        axis.set(xlim=(0, width), ylim=(height, 0))
        axis.set_title(name, color="white", fontsize=11, pad=5)
        axis.set_axis_off()

    bev = figure.add_subplot(grid[2, 0], facecolor="#0b1220")
    points = _load_points(samples_dir, point_dim)
    if len(points):
        points = points[::max(1, int(np.ceil(len(points) / 80000)))]
        bev.scatter(points[:, 0], points[:, 1], s=.3, c="#7c93aa", alpha=.55, rasterized=True)
    for box, box_corners, label in zip(boxes, corners, labels):
        outline = box_corners[[0, 1, 2, 3, 0], :2]
        bev.plot(outline[:, 0], outline[:, 1], color=colors[label], linewidth=1.2)
        # Local +x marks heading; velocities are not used to infer orientation.
        bev.plot([box[0], box[0] + np.cos(box[6]) * box[3] * .6],
                 [box[1], box[1] + np.sin(box[6]) * box[3] * .6], color=colors[label], linewidth=1)
    bev.scatter([0], [0], marker="^", s=55, color="white", label="LiDAR origin")
    extent = 55.
    if len(points):
        extent = max(extent, min(80., float(np.percentile(np.abs(points[:, :2]), 99))))
    bev.set(xlim=(-extent, extent), ylim=(-extent, extent), xlabel="LiDAR x (m)", ylabel="LiDAR y (m)")
    bev.set_aspect("equal")
    bev.set_title("LiDAR BEV + detection footprints", color="white", fontsize=12)
    bev.tick_params(colors="#cbd5e1", labelsize=8)
    bev.xaxis.label.set_color("#cbd5e1")
    bev.yaxis.label.set_color("#cbd5e1")
    bev.grid(alpha=.15)
    for spine in bev.spines.values():
        spine.set_color("#475569")

    occupied = occ != 17
    top_index = np.where(occupied, np.arange(16)[None, None, :], -1).max(axis=2)
    top_class = np.take_along_axis(occ, np.maximum(top_index, 0)[..., None], axis=2)[..., 0]
    top_class[top_index < 0] = 17
    palette = colormaps["tab20"](np.arange(18))
    palette[17] = [.055, .08, .13, 1.]
    occupancy_axis = figure.add_subplot(grid[2, 1], facecolor="#0b1220")
    occupancy_axis.imshow(palette[top_class.T], origin="lower", extent=(-.5, 199.5, -.5, 199.5), interpolation="nearest")
    occupancy_axis.set(xlabel="Occupancy x cell", ylabel="Occupancy y cell")
    occupancy_axis.set_title("Occupancy: highest occupied voxel", color="white", fontsize=12)
    occupancy_axis.tick_params(colors="#cbd5e1", labelsize=8)
    occupancy_axis.xaxis.label.set_color("#cbd5e1")
    occupancy_axis.yaxis.label.set_color("#cbd5e1")

    legend_axis = figure.add_subplot(grid[2, 2])
    legend_axis.set_axis_off()
    detection_handles = [Line2D([0], [0], color=colors[i], lw=3,
                         label=f"{CLASS_NAMES[i]} ({int(np.sum(labels == i))})") for i in range(10)]
    legend_axis.legend(handles=detection_handles, loc="upper left", frameon=False,
                       labelcolor="white", fontsize=10, title="Detection classes / count", title_fontsize=12)
    legend_axis.get_legend().get_title().set_color("white")
    legend_axis.text(.02, .05,
                     "Occupancy: topmost non-free voxel shown\n"
                     "Grid indices shown; no assumed metric calibration\n"
                     "Camera boxes use the original image calibration.\n"
                     "Predictions are model output, no ground truth.",
                     transform=legend_axis.transAxes, color="#cbd5e1", fontsize=8, va="bottom", linespacing=1.5)

    # Use the exact image palette for every semantic legend swatch.
    # 图例色块直接使用图像调色板，确保类别颜色一一对应。
    occupancy_legend_axis = figure.add_subplot(grid[3, :])
    occupancy_legend_axis.set_axis_off()
    occupancy_handles = [
        Patch(facecolor=palette[i], edgecolor="#64748b", linewidth=.6,
              label=f"{i:02d}  {name}")
        for i, name in enumerate(OCCUPANCY_CLASS_NAMES)
    ]
    occupancy_legend = occupancy_legend_axis.legend(
        handles=occupancy_handles, loc="upper center", ncol=6, frameon=False,
        labelcolor="white", fontsize=10, title="Occupancy classes (Occ3D-nuScenes) / class ID",
        title_fontsize=12, columnspacing=2.5, handlelength=1.5, handleheight=1.0,
        labelspacing=.9, borderaxespad=0.)
    occupancy_legend.get_title().set_color("white")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = (output_dir / "demo.png").resolve()
    figure.savefig(destination, dpi=150, facecolor=figure.get_facecolor())
    figure.clear()
    return destination
