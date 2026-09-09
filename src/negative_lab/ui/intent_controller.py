"""IntentController：微调意图与中性点救援（2026-09-08 MainWindow 拆分）。

只搬不改（交接 §12-(6)）：候选选择、RenderIntent 滑杆编辑/提交/全部重置、
帧步进、中性点拾点救援与撤销从 MainWindow 原文搬入本 mixin；
共享状态（_intent/_candidate/_pick_undo_stack/_canvas_rec/...）仍在
MainWindow 实例上。

坐标契约（v6.2 F5）：拾点画布坐标先加 _view_origin 才是
_positive_full 索引；「按住看原片」时拒绝拾点（C8 前提已实测：
tests/unit/test_c8_hold_orig_guard.py）。
"""

from __future__ import annotations

import numpy as np

from ..pipeline.render import RenderIntent
from .state_store import _CANVAS_STALE


class IntentControllerMixin:
    """RenderIntent 编辑链与中性点救援（8 个方法，原文搬移）。"""

    def _step(self, direction: int):
        paths = self.library.current_frames()
        if not paths:
            return
        try:
            i = paths.index(self.current.source_path) if self.current else 0
        except ValueError:
            i = 0
        self.load_frame(paths[max(0, min(len(paths) - 1, i + direction))])

    def _choose_candidate(self, name: str):
        self._candidate = name
        if self.current is not None:
            self.current.chosen_candidate = name
        if self.pr is not None and self.pr.candidates is not None:
            cand = self.pr.candidates.get(name)
            self._intent = RenderIntent.from_dict(cand.intent)
            # v6.2 F3：v6 单工作区已没有 convert_panel，滑杆在右栏
            self.right.panel.set_intent(self._intent, silent=True)
        self._compose_view()

    def _intent_changed(self, field: str, value: float):
        """微调滑杆：即时合成（RenderIntent 应用在缓存的正片上，毫秒级），
        不重跑管线；技术正片按契约不变。"""
        setattr(self._intent, field, float(value))
        self._compose_view()

    def _intent_committed(self, field: str, value: float):
        setattr(self._intent, field, float(value))
        if self.current is not None:
            self.current.render_intent_override = self._intent.to_dict()
        self._compose_view()

    def _reset_all_intent(self):
        """「全部重置」：只清 RenderIntent（风格契约所在层）。

        候选与技术正片一概不动：reset_all() 清的是十个微调控件。
        此前这条链断了 —— 按钮 emit 出去没有任何接收者。
        """
        self._intent.reset_all()
        if self.current is not None:
            self.current.render_intent_override = self._intent.to_dict()
        self.right.panel.set_intent(self._intent, silent=True)
        self._compose_view()
        self._set_banner("已重置全部微调项（技术正片与候选未变）", "info")

    # ------------------------------------------------ R5b 中性点救援
    def _on_pick_neutral(self, on: bool):
        """拾点救援开关（调整阶段）：进入/退出 view 的 neutral 拾点模式。"""
        # 先判数据再切阶段：此前顺序反了，空文档也会被打到「调整」页
        if on and self.current is None:
            self.right.btn_pick_neutral.setChecked(False)
            self.right.set_pick_hint("还没有帧可拾点：请先导入并选择一帧")
            return
        if on and self._stage != 1:
            self._set_stage(1)
        self.view.set_pick_mode("neutral" if on else None)
        if on:
            self._set_banner("中性点救援：在画面上点击应该呈现中性灰的位置"
                             "（Esc 退出）", "info")

    def _on_point_picked(self, kind: str, x: float, y: float):
        """画布拾点：中性点 -> 精确救援 -> 即时合成 + 撤销栈。"""
        if kind != "neutral":
            return
        rec = self.current
        if rec is None or self._positive_full is None:
            return
        # 审查修复（GUI 审查 重要-6）：帧加载期间画布仍显示上一帧，而
        # self.current 已指向新帧 —— 此时拾点会把「上一帧像素解出的 intent」
        # 写进新帧记录。校验画布数据归属，不一致直接拒绝。
        if getattr(self, "_canvas_rec", None) is _CANVAS_STALE:
            self.right.set_pick_hint("帧数据尚未就绪（正在加载/刚切换），请稍候再拾点")
            return
        # v6.2 §7.6 / 审查 F9：拾点求解取的是**技术正片**像素，只能发生在
        # 调整阶段；分析阶段中央显示的是负片，此时点击必须拒绝而不是按正片求解。
        if self._stage != 1:
            self.right.set_pick_hint("中性点救援只在「调整」阶段可用")
            return
        if self._orig_held or self.btn_hold_orig.isDown():
            # 冷读审查 L4-①：「按住看原片」时 _compose_view 提前 return，既不刷新
            # _view_origin 也不换采样源：此刻底图是负片，采样却按正片 + 上一次
            # 裁切偏移取像素，解出来的调色与所见画面毫无关系。
            self.right.set_pick_hint("请先松开「按住看原片」再拾点")
            return
        from ..pipeline.rescue import neutral_pick_intent_exact

        # v6.2 F5（修正 v6.1 的"零换算"断言）：调整阶段且未勾选保留片边时，
        # 画布显示的是整幅技术正片的整体裁切子矩形，_compose_view 已把该子矩形
        # 原点记入 _view_origin；画布坐标必须先加偏移才是 _positive_full 索引。
        # 仅当显示整幅（裁切为空 / 勾选保留片边）时偏移为 (0, 0)。
        ox, oy = self._view_origin
        x_full, y_full = float(x) + ox, float(y) + oy
        H, W = self._positive_full.shape[:2]
        if not (np.isfinite(x_full) and np.isfinite(y_full)) \
                or not (0 <= int(round(x_full)) < W
                        and 0 <= int(round(y_full)) < H):
            self.right.set_pick_hint("拾点失败：请点在画面内")
            self._set_banner("拾点位置在画面外，未应用救援", "warning")
            return
        px, py = int(round(x_full)), int(round(y_full))
        try:
            intent, warns = neutral_pick_intent_exact(
                self._positive_full, py, px, base_intent=self._intent,
                return_warnings=True)
        except ValueError as e:
            self.right.set_pick_hint(f"拾点失败：{e}")
            return
        self._pick_undo_stack.append(self._intent.copy())
        self._intent = intent
        if rec is not None:
            rec.render_intent_override = intent.to_dict()
        self.right.panel.set_intent(intent, silent=True)
        self._compose_view()
        self.right.set_undo_pick_enabled(True)
        gain_note = (f"temp {intent.temperature:+.1f} / tint {intent.tint:+.1f}")
        if warns:
            self.right.set_pick_hint("⚠ " + "；".join(warns[:2]) +
                                     f"（{gain_note}）")
            self._set_banner("中性点救援已应用，但拾点可信度存疑（见右栏提示）",
                             "warning")
        else:
            self.right.set_pick_hint(f"已把该点拉向中性灰（{gain_note}）。"
                                     "不满意可再点别处或「撤销拾点」。")
            self._set_banner("中性点救援已应用", "success")

    def _undo_pick_rescue(self):
        """撤销上一次拾点救援（恢复栈里的上一个 intent）。

        栈只在 _frame_done 成功路径清空；加载失败时会留着上一帧的 intent，
        此时撤销会把 A 帧的调色写进 B 帧（冷读审查 L4-②）。
        """
        if not self._pick_undo_stack:
            return
        if getattr(self, "_canvas_rec", None) is _CANVAS_STALE:
            self._set_banner("本帧尚未加载完成，撤销拾点暂不可用（栈里是上一帧的结果）",
                             "warning")
            return
        prev = self._pick_undo_stack.pop()
        self._intent = prev
        if self.current is not None:
            self.current.render_intent_override = prev.to_dict()
        self.right.panel.set_intent(prev, silent=True)
        self._compose_view()
        if not self._pick_undo_stack:
            self.right.set_undo_pick_enabled(False)
            self.right.set_pick_hint("")
        self._set_banner("已撤销上一次拾点救援", "info")
