"""统一几何变换（v5 §3.2 / §5.4）。

坐标语义（唯一权威定义）
------------------------
- **SOURCE**：原始未旋转文件的 (row, col) 像素坐标。标注、sidecar、会话、
  GUI/CLI/批处理传入的 ROI 一律以 SOURCE 空间、**半开区间** (y0,x0,y1,x1) 保存。
- **ORIENTED**：依次施加 `rotation_degrees`（顺时针 90 的倍数）→
  `mirror_horizontal` → `fine_rotation_degrees`（绕中心，画布尺寸不变）之后
  的坐标。处理顺序固定：orient(source) -> crop(oriented)。

三类 ROI（v5 §3.2，语义互不合并）
---------------------------------
`output_crop_roi`（最终构图）/ `image_area_roi`（算法与统计）/
`base_sample_rois[]`（未曝光片基取样，可多个）。

细微旋转下矩形映射为四边形：正向映射取四角变换后的多边形，反向映射取其
逆变换四角的轴对齐外接框。因此 SOURCE 矩形 -> ORIENTED -> SOURCE 的
round-trip 误差为 0（浮点内），满足 v5 §11.1 的 <=1 像素要求。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

VALID_ROT = (0, 90, 180, 270)
MAX_FINE_ROTATION = 10.0          # P0 只做小范围拉直
MIN_CROP_PX = 16                  # 小于此边长的裁切视为无效（不静默退化）


# ------------------------------------------------------------------ 整数旋转
def rotate_cw(arr: np.ndarray, deg: int) -> np.ndarray:
    """顺时针旋转 90 的倍数。np.rot90 是逆时针，故 k=(4-deg//90)%4。"""
    if deg not in VALID_ROT:
        raise ValueError(f"rotation_degrees 必须是 {VALID_ROT}，得到 {deg!r}")
    k = (4 - deg // 90) % 4
    return np.ascontiguousarray(np.rot90(arr, k, axes=(0, 1))) if k else arr


def mirror_h(arr: np.ndarray) -> np.ndarray:
    """水平镜像（左右翻转）。"""
    return np.ascontiguousarray(arr[:, ::-1, ...])


#: **35mm 单格画框必然宽>高**（齿孔在长边上，画框 36×24）。翻拍时片带可以横躺在
#: 相机画布里，这时量出来的画框是**高>宽**——那不是构图，是没回正。
#: 实测 `1299_G200` 用户亲手框出的 11 帧画框全是 5940~6012 高 × 4800~4854 宽
#: （解码画布 8280 高 × 5520 宽，竖幅），而他一侧的手工成品是 6144×5295 一类横幅。
#: 补转的方向取**逆时针 270°（= 逆时针 90°）**：几何只断定"必须转 90/270"，
#: 断定不了往哪边转，所以拿他那三张 1299 参考图当目标实测——
#: 5 个网格 × 3 个中心裁切共 15 个窗口，三帧**每一格都指向逆时针**
#: （lead(CW−CCW) 的 min −1.210 / −0.566 / −0.192，中位 −1.630 / −0.825 / −0.254；
#: 探针 `tmp_orientation_sign2.py`）。
#:
#: **这个数上一版是 90（顺时针），被看图当场推翻。** 错在量法不在几何：
#: 旧探针的 `layout()` 无条件做 `255−g`（源数组是负片，确实该反相），
#: 可他手工做的参考图是**正片**，一起反相等于拿"我方正片 vs 他的负片"比亮度布局，
#: 相关符号整体翻转 ⇒ 15 个窗口"高度一致"地指向一个把画面转成**上下颠倒**的角度，
#: 而 `outputs/ref_look_probe/all11_annotated.jpg` 上 1299 三帧一眼就是倒的。
#: 教训：**窗口自检只回答"换个取样窗口结论会不会翻"，它查不出"两边的极性对不对"**；
#: 极性这类错只能靠看图，或靠一个已知正立的参照物来抓。
#: 方向仍然逐帧可推翻：规则改过的帧一定带复核提示，他一键就能翻 180°。
SIDEWAYS_FRAME_ROTATION = 270


def frame_output_is_portrait(image_area_source, rotation_degrees: int) -> bool | None:
    """按 SOURCE 空间画框 + 整数旋转，判断**输出**里的画框是不是高>宽。

    0/180 保持画框纵横，90/270 交换它。画框缺失或正方时返回 None（无从判定）。
    """
    if not image_area_source:
        return None
    y0, x0, y1, x1 = (int(v) for v in image_area_source)
    h, w = y1 - y0, x1 - x0
    if h <= 0 or w <= 0 or h == w:
        return None
    if rotation_degrees in (90, 270):
        h, w = w, h
    return h > w


def sideways_correction(image_area_source, rotation_degrees: int) -> int:
    """画框在输出里仍是竖幅时返回**还需要补的**角度，否则 0。

    只回答"要不要转 90°"这一件事（几何事实），不回答"往哪边转"（约定，
    见 `SIDEWAYS_FRAME_ROTATION` 的实测出处与被看图推翻的经过）。
    """
    return SIDEWAYS_FRAME_ROTATION if frame_output_is_portrait(
        image_area_source, rotation_degrees) else 0


def rotate_step(image_area_source, delta: int) -> int:
    """旋转按钮该转多少度：片带横躺的帧，一次按键翻 **180°** 而不是 90°。

    还是那条几何事实：源画框高>宽（片带横躺）时，从横幅再转 90° 必定得到竖幅，
    而 35mm 单格不存在竖幅 ⇒ 那一按要么被规则立刻补回去（看起来什么都没发生），
    要么把他带进物理上不可能的状态。所以这种帧上 ±90 的按键执行
    "在两个合法横幅之间翻面"的 180°；画框本来就是横的（正常扫描）时按键语义不变。
    """
    if delta in (90, 270) and frame_output_is_portrait(image_area_source, 0):
        return (2 * delta) % 360
    return delta


def oriented_shape(src_h: int, src_w: int, deg: int,
                   mirror: bool = False) -> tuple[int, int]:
    """整数旋转/镜像后的画布尺寸（细微旋转不改变画布尺寸）。"""
    if deg in (90, 270):
        return src_w, src_h
    return src_h, src_w


def _int_source_to_oriented(yx: tuple[float, float], src_h: int, src_w: int,
                            deg: int, mirror: bool) -> tuple[float, float]:
    """SOURCE 像素点 -> 整数旋转/镜像后的 ORIENTED 点（半开区间角点语义）。"""
    y, x = float(yx[0]), float(yx[1])
    if deg == 0:
        oy, ox = y, x
    elif deg == 90:
        oy, ox = x, src_h - y
    elif deg == 180:
        oy, ox = src_h - y, src_w - x
    else:
        oy, ox = src_w - x, y
    if mirror:
        ow = src_w if deg in (0, 180) else src_h
        ox = ow - ox
    return oy, ox


def _int_oriented_to_source(yx: tuple[float, float], src_h: int, src_w: int,
                            deg: int, mirror: bool) -> tuple[float, float]:
    oy, ox = float(yx[0]), float(yx[1])
    if mirror:
        ow = src_w if deg in (0, 180) else src_h
        ox = ow - ox
    if deg == 0:
        return oy, ox
    if deg == 90:
        return src_h - ox, oy
    if deg == 180:
        return src_h - oy, src_w - ox
    return ox, src_w - oy


def roi_source_to_oriented(roi, src_h: int, src_w: int, deg: int,
                           mirror: bool = False) -> tuple[int, int, int, int]:
    """SOURCE (y0,x0,y1,x1) -> ORIENTED 轴对齐外接框（整数旋转/镜像）。"""
    y0, x0, y1, x1 = roi
    pts = [_int_source_to_oriented((yy, xx), src_h, src_w, deg, mirror)
           for yy in (y0, y1) for xx in (x0, x1)]
    ys = [p[0] for p in pts]
    xs = [p[1] for p in pts]
    nh, nw = oriented_shape(src_h, src_w, deg, mirror)
    return (max(0, int(math.floor(min(ys)))), max(0, int(math.floor(min(xs)))),
            min(nh, int(math.ceil(max(ys)))), min(nw, int(math.ceil(max(xs)))))


def oriented_to_source(roi, src_h: int, src_w: int, deg: int,
                       mirror: bool = False) -> tuple[int, int, int, int]:
    """ORIENTED 轴对齐框 -> SOURCE（整数旋转/镜像的精确逆变换）。"""
    y0, x0, y1, x1 = roi
    pts = [_int_oriented_to_source((yy, xx), src_h, src_w, deg, mirror)
           for yy in (y0, y1) for xx in (x0, x1)]
    ys = [p[0] for p in pts]
    xs = [p[1] for p in pts]
    return (int(round(min(ys))), int(round(min(xs))),
            int(round(max(ys))), int(round(max(xs))))


# ------------------------------------------------------------------ 细微旋转
def fine_matrix(h: int, w: int, deg: float) -> np.ndarray:
    """绕画布中心旋转 deg（顺时针为正）的 3x3 仿射（oriented 像素坐标）。"""
    a = math.radians(float(deg))
    cy, cx = h / 2.0, w / 2.0
    cos, sin = math.cos(a), math.sin(a)
    # 顺时针旋转：(y,x) 平面内 [y';x'] = R(-a) 的组合，这里显式推导：
    # x' = cx + (x-cx)cos a + (y-cy) sin a
    # y' = cy - (x-cx)sin a + (y-cy) cos a
    m = np.array([[cos, sin, cx - cos * cx - sin * cy],
                  [-sin, cos, cy + sin * cx - cos * cy],
                  [0.0, 0.0, 1.0]], np.float64)
    return m


def _apply3(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """pts: (N,2) 的 (y,x) -> 仿射映射后的 (N,2)。"""
    hom = np.column_stack([pts[:, 1], pts[:, 0], np.ones(len(pts))])  # x,y,1
    out = hom @ m.T
    return np.column_stack([out[:, 1], out[:, 0]])


@dataclass
class OrientTransform:
    """SOURCE <-> ORIENTED 的完整可逆映射（含细微旋转）。"""

    source_size: tuple[int, int]
    rotation_degrees: int = 0
    mirror_horizontal: bool = False
    fine_rotation_degrees: float = 0.0

    def __post_init__(self):
        if self.rotation_degrees not in VALID_ROT:
            raise ValueError(f"rotation_degrees 非法：{self.rotation_degrees!r}")
        if abs(float(self.fine_rotation_degrees)) > MAX_FINE_ROTATION:
            raise ValueError(f"fine_rotation_degrees 超出 P0 范围 ±{MAX_FINE_ROTATION}："
                             f"{self.fine_rotation_degrees!r}")

    @property
    def oriented_size(self) -> tuple[int, int]:
        h, w = self.source_size
        return oriented_shape(h, w, self.rotation_degrees, self.mirror_horizontal)

    # ---- 点
    def source_to_oriented(self, yx) -> tuple[float, float]:
        h, w = self.source_size
        oy, ox = _int_source_to_oriented(yx, h, w, self.rotation_degrees,
                                         self.mirror_horizontal)
        if abs(self.fine_rotation_degrees) < 1e-9:
            return float(oy), float(ox)
        oh, ow = self.oriented_size
        p = _apply3(fine_matrix(oh, ow, self.fine_rotation_degrees),
                    np.array([[oy, ox]], np.float64))[0]
        return float(p[0]), float(p[1])

    def oriented_to_source_point(self, yx) -> tuple[float, float]:
        h, w = self.source_size
        oy, ox = float(yx[0]), float(yx[1])
        if abs(self.fine_rotation_degrees) > 1e-9:
            oh, ow = self.oriented_size
            inv = np.linalg.inv(fine_matrix(oh, ow, self.fine_rotation_degrees))
            p = _apply3(inv, np.array([[oy, ox]], np.float64))[0]
            oy, ox = float(p[0]), float(p[1])
        return _int_oriented_to_source((oy, ox), h, w, self.rotation_degrees,
                                       self.mirror_horizontal)

    # ---- 矩形 <-> 四边形
    def rect_corners(self, roi) -> np.ndarray:
        y0, x0, y1, x1 = roi
        return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], np.float64)

    def quad_source_to_oriented(self, roi) -> np.ndarray:
        return np.array([self.source_to_oriented(p) for p in self.rect_corners(roi)])

    def bbox_source_to_oriented(self, roi) -> tuple[int, int, int, int]:
        q = self.quad_source_to_oriented(roi)
        oh, ow = self.oriented_size
        return (max(0, int(math.floor(q[:, 0].min()))),
                max(0, int(math.floor(q[:, 1].min()))),
                min(oh, int(math.ceil(q[:, 0].max()))),
                min(ow, int(math.ceil(q[:, 1].max()))))

    def bbox_oriented_to_source(self, roi) -> tuple[int, int, int, int]:
        q = np.array(self.rect_corners(roi), np.float64)
        back = np.array([self.oriented_to_source_point(p) for p in q])
        h, w = self.source_size
        return (int(np.clip(round(back[:, 0].min()), 0, h)),
                int(np.clip(round(back[:, 1].min()), 0, w)),
                int(np.clip(round(back[:, 0].max()), 0, h)),
                int(np.clip(round(back[:, 1].max()), 0, w)))

    # ---- 图像
    def apply(self, t_source: np.ndarray) -> np.ndarray:
        """SOURCE 图像 -> ORIENTED 图像（旋转/镜像/细微旋转）。"""
        out = rotate_cw(np.asarray(t_source), self.rotation_degrees)
        if self.mirror_horizontal:
            out = mirror_h(out)
        if abs(self.fine_rotation_degrees) > 1e-9:
            out = _rotate_fine(out, self.fine_rotation_degrees)
        return np.ascontiguousarray(out)


def _rotate_fine(arr: np.ndarray, deg: float) -> np.ndarray:
    """双线性小角度旋转（画布尺寸不变，越界用边缘值填充）。

    不引入新依赖：用 scipy 之外的纯 NumPy 实现，P0 角度 <=10 度，
    采样误差可忽略且严格保持有限值。
    """
    h, w = arr.shape[:2]
    m = np.linalg.inv(fine_matrix(h, w, deg))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    hom = np.column_stack([xx.ravel(), yy.ravel(), np.ones(h * w)])
    src = hom @ m.T                       # x', y'
    sx, sy = src[:, 0], src[:, 1]
    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    fx = (sx - x0).astype(np.float32)
    fy = (sy - y0).astype(np.float32)
    x0c = np.clip(x0, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)
    a = arr[y0c, x0c]
    b = arr[y0c, x1c]
    c = arr[y1c, x0c]
    d = arr[y1c, x1c]
    top = a + (b - a) * fx[:, None] if arr.ndim == 3 else a + (b - a) * fx
    bot = c + (d - c) * fx[:, None] if arr.ndim == 3 else c + (d - c) * fx
    out = top + (bot - top) * fy[:, None] if arr.ndim == 3 else top + (bot - top) * fy
    return out.reshape(arr.shape).astype(np.float32)


# ------------------------------------------------------------------ 记录
@dataclass
class GeometryRecord:
    """进入 sidecar 的几何事实（旋转/镜像/细微旋转/三类 ROI 全部可复现）。"""

    source_size: tuple[int, int]          # (h, w)
    rotation_degrees: int
    crop_oriented: tuple[int, int, int, int] | None   # (y0,x0,y1,x1) ORIENTED
    output_size: tuple[int, int]
    keep_rebate: bool = False
    mirror_horizontal: bool = False
    fine_rotation_degrees: float = 0.0
    roi_source_space: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_size": list(self.source_size),
            "rotation_degrees": self.rotation_degrees,
            "mirror_horizontal": bool(self.mirror_horizontal),
            "fine_rotation_degrees": float(self.fine_rotation_degrees),
            "crop_oriented": list(self.crop_oriented) if self.crop_oriented else None,
            "output_size": list(self.output_size),
            "keep_rebate": self.keep_rebate,
            "roi_source_space": self.roi_source_space,
        }

    def transform(self) -> OrientTransform:
        return OrientTransform(self.source_size, self.rotation_degrees,
                               self.mirror_horizontal, self.fine_rotation_degrees)


def apply_geometry(t_source: np.ndarray, *, rotation_degrees: int = 0,
                   crop_oriented: tuple[int, int, int, int] | None = None,
                   keep_rebate: bool = False,
                   mirror_horizontal: bool = False,
                   fine_rotation_degrees: float = 0.0,
                   roi_source_space: dict | None = None
                   ) -> tuple[np.ndarray, GeometryRecord]:
    """orient(旋转+镜像+细微旋转) + 可选裁切。所有入口共用的唯一几何出口。

    crop_oriented=None 或过小（<16px）时不裁切（不静默产生退化输出）。
    """
    if t_source.ndim != 3 or t_source.shape[2] != 3:
        raise ValueError(f"需要 (H,W,3) 数组，得到 {t_source.shape}")
    src_h, src_w = t_source.shape[:2]
    tr = OrientTransform((src_h, src_w), rotation_degrees, mirror_horizontal,
                         fine_rotation_degrees)
    out = tr.apply(t_source)
    crop = crop_oriented
    if crop is not None and not keep_rebate:
        y0, x0, y1, x1 = (int(v) for v in crop)
        y0, x0 = max(0, y0), max(0, x0)
        y1, x1 = min(out.shape[0], y1), min(out.shape[1], x1)
        if y1 - y0 >= MIN_CROP_PX and x1 - x0 >= MIN_CROP_PX:
            out = np.ascontiguousarray(out[y0:y1, x0:x1, :])
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
        mirror_horizontal=mirror_horizontal,
        fine_rotation_degrees=float(fine_rotation_degrees),
        roi_source_space=roi_source_space or {},
    )
    return out, rec


# ------------------------------------------------------------------ mask
def quad_mask(shape: tuple[int, int], quad: np.ndarray) -> np.ndarray:
    """四边形（ORIENTED 坐标 (N,2) 的 y,x）-> bool 掩码（扫描线填充）。"""
    h, w = shape
    mask = np.zeros((h, w), bool)
    if len(quad) == 0:
        return mask
    ys, _xs = quad[:, 0], quad[:, 1]
    y0 = max(0, int(math.floor(ys.min())))
    y1 = min(h, int(math.ceil(ys.max())))
    if y1 <= y0:
        return mask
    n = len(quad)
    for y in range(y0, y1):
        yc = y + 0.5
        xints: list[float] = []
        for i in range(n):
            (ya, xa), (yb, xb) = quad[i], quad[(i + 1) % n]
            if ya == yb:
                continue
            if min(ya, yb) <= yc < max(ya, yb):
                t = (yc - ya) / (yb - ya)
                xints.append(xa + t * (xb - xa))
        xints.sort()
        for a, b in zip(xints[0::2], xints[1::2]):
            xa = max(0, int(math.floor(a)))
            xb = min(w, int(math.ceil(b)))
            if xb > xa:
                mask[y, xa:xb] = True
    return mask


def rect_mask(shape: tuple[int, int], roi) -> np.ndarray:
    h, w = shape
    y0 = max(0, int(roi[0])); x0 = max(0, int(roi[1]))
    y1 = min(h, int(roi[2])); x1 = min(w, int(roi[3]))
    m = np.zeros((h, w), bool)
    if y1 > y0 and x1 > x0:
        m[y0:y1, x0:x1] = True
    return m


# ------------------------------------------------------------------ 同步
def clamp_roi(roi, size: tuple[int, int], min_side: int = MIN_CROP_PX
              ) -> tuple[int, int, int, int] | None:
    """把 ROI 钳制进画布；退化到小于 min_side 时返回 None（不静默越界）。"""
    h, w = size
    y0, x0, y1, x1 = (int(v) for v in roi)
    y0 = max(0, min(y0, h - min_side)); x0 = max(0, min(x0, w - min_side))
    y1 = max(min_side, min(y1, h)); x1 = max(min_side, min(x1, w))
    if y1 <= y0 or x1 <= x0:
        return None
    return (y0, x0, y1, x1)
