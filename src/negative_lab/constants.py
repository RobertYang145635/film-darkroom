"""叶子常量（无任何包内依赖）。

为什么单独一个模块：io/sidecar.py 需要通用片基先验 GENERIC_PRIOR 给旧 sidecar
填默认字段，而这个字典原先定义在 pipeline/roll_analysis.py —— 构成 io(下层)
-> pipeline(上层) 的反向依赖。值是纯数据，下沉到叶子后两条依赖都指向叶子，
方向正确且不可能成环。pipeline.roll_analysis.GENERIC_PRIOR 仍以别名可用。
"""

from __future__ import annotations

#: "未知彩色负片"片基色度形状（按 R 归一）。来源：仓库内 C-41 与 ECN-2
#: 卷实测片基通道比的保守平均（fixtures/measurements/*.measure.json）。
GENERIC_PRIOR = {
    "asset_version": "generic-color-negative-base-v1",
    "ratios": (1.0, 0.45, 0.12),
    "note": ("非物理测量：未知彩色负片的保守回退。只允许从画面统计估计共同"
             "曝光尺度，不得从场景内容重新拟合片基色度（v5 §6.2.5）。"),
}
