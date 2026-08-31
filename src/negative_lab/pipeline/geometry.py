"""统一几何变换（整改 P0-C；I-1/I-2/I-3 的共同根因是各入口各搞一套坐标）。

坐标语义（唯一权威定义）：

- SOURCE 空间：原始未旋转文件的 (row, col) 像素坐标。标注、sidecar、
  GUI/CLI/批处理传入的 ROI 一律用 SOURCE 空间声明。
- ORIENTED 空间：按 rotation_degrees **顺时针**旋转后的坐标。
- 处理顺序固定：rotate(source) -> crop(oriented)。裁切 ROI 在 ORIENTED
  空间给出，或由 SOURCE 空间变换得到。

`apply_geometry` 是 GUI、CLI、批处理唯一允许的几何入口。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

VALID_ROT = (0, 90, 180, 270)


def rotate_cw(arr: np.ndarray, deg: int) -> np.ndarray:
    """顺时针旋转 90 的倍数。np.rot90 是逆时针，故 k=(4-deg//90)%4。"""
    if deg not in VALID_ROT:
        raise ValueError(f"rotation_degrees 必须是 {VALID_ROT}，得到 {deg!r}")
    k = (4 - deg // 90) % 4
    return np.rot90(arr, k, axes=(0, 1)).copy() if k else arr


def oriented_to_source(roi: tuple[int, int, int, int], src_h: int, src_w: int,
                       deg: int) -> tuple[int, int, int, int]:
    """roi_source_to_oriented 的逆变换（ORIENTED -> SOURCE）。"""
    y0, x0, y1, x1 = roi
    if deg == 0:
        return (y0, x0, y1, x1)
    if deg == 90:      # 正向: (y0,x0,y1,x1)->(x0, sh-y1, x1, sh-y0)
        # 反解: source = (sh-x1o, y0o, sh-x0o, y1o)
        return (src_h - x1, y0, src_h - x0, y1)
    if deg == 180:
        return (src_h - y1, src_w - x1, src_h - y0, src_w - x0)
    return (x0, src_w - y1, x1, src_w - y0)   # deg==270 反解


def roi_source_to_oriented(roi: tuple[int, int, int, int], src_h: int, src_w: int,
                           deg: int) -> tuple[int, int, int, int]:
    """SOURCE (y0,x0,y1,x1) 半开区间 -> ORIENTED 空间（顺时针 deg 后）。

    推导（CW90：源 (y,x) -> 新 (x, src_h-1-y)）：
      90:  (x0, src_h-y1, x1, src_h-y0)
      180: (src_h-y1, src_w-x1, src_h-y0, src_w-x0)
      270: (src_w-x1, y0, src_w-x0, y1)
    """
    y0, x0, y1, x1 = roi
    if deg == 0:
        return (max(0, y0), max(0, x0), min(src_h, y1), min(src_w, x1))
    if deg == 90:
        out = (x0, src_h - y1, x1, src_h - y0)
    elif deg == 180:
        out = (src_h - y1, src_w - x1, src_h - y0, src_w - x0)
    else:
        out = (src_w - x1, y0, src_w - x0, y1)
    # 钳制到有效范围（标注/检测坐标可能因降采样步长越过边缘，
    # 负索引会静默产生空切片 -> Dmin 回退 -> 全图冲淡，实测教训）
    nh = src_w if deg in (90, 270) else src_h
    nw = src_h if deg in (90, 270) else src_w
    oy0, ox0, oy1, ox1 = out
    return (max(0, oy0), max(0, ox0), min(nh, oy1), min(nw, ox1))


@dataclass
class GeometryRecord:
    """进入 sidecar 的几何事实（整改 I-3：旋转/裁切必须可复现）。"""

    source_size: tuple[int, int]          # (h, w)
    rotation_degrees: int
    crop_oriented: tuple[int, int, int, int] | None   # (y0,x0,y1,x1) ORIENTED 空间
    output_size: tuple[int, int]
    keep_rebate: bool = False
    roi_source_space: dict = field(default_factory=dict)  # {name: [y0,x0,y1,x1]}

    def to_dict(self) -> dict:
        return {
            "source_size": list(self.source_size),
            "rotation_degrees": self.rotation_degrees,
            "crop_oriented": list(self.crop_oriented) if self.crop_oriented else None,
            "output_size": list(self.output_size),
            "keep_rebate": self.keep_rebate,
            "roi_source_space": self.roi_source_space,
        }


def apply_geometry(t_source: np.ndarray, *, rotation_degrees: int = 0,
                   crop_oriented: tuple[int, int, int, int] | None = None,
                   keep_rebate: bool = False,
                   roi_source_space: dict | None = None) -> tuple[np.ndarray, GeometryRecord]:
    """旋转（顺时针）+ 可选裁切（ORIENTED 空间）。所有入口共用。

    crop_oriented=None 时不裁切（keep_rebate=True 的语义就是保留片边不裁）。
    裁切窗口过小（<16px）视为无效并忽略裁切（不静默产生退化输出）。
    """
    if t_source.ndim != 3 or t_source.shape[2] != 3:
        raise ValueError(f"需要 (H,W,3) 数组，得到 {t_source.shape}")
    src_h, src_w = t_source.shape[:2]
    out = rotate_cw(t_source, rotation_degrees)
    crop = crop_oriented
    if crop is not None and not keep_rebate:
        y0, x0, y1, x1 = (int(v) for v in crop)
        y0 = max(0, y0); x0 = max(0, x0)
        y1 = min(out.shape[0], y1); x1 = min(out.shape[1], x1)
        if y1 - y0 >= 16 and x1 - x0 >= 16:
            out = out[y0:y1, x0:x1, :]
            crop = (y0, x0, y1, x1)
        else:
            crop = None
    elif keep_rebate:
        crop = None
    rec = GeometryRecord(
        source_size=(src_h, src_w),
        rotation_degrees=rotation_degrees,
        crop_oriented=crop,
        output_size=out.shape[:2],
        keep_rebate=keep_rebate,
        roi_source_space=roi_source_space or {},
    )
    return out, rec
