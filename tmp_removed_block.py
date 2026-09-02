    def _build_toolbars(self):
        """中央查看区下方的工具条：随模式切换显隐（图像不动）。"""
        from PySide6.QtWidgets import QFrame

        t = theme
        # 裁切工具条
        self.bar_crop = QWidget()
        lay = QHBoxLayout(self.bar_crop)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(t.SPACING["xs"])
        self.btn_crop = self._tool_button(icons.crop(), "绘制：输出裁切")
        self.btn_area = self._tool_button(icons.target(), "绘制：有效画面")
        self.btn_base = self._tool_button(icons.plus(), "绘制：片基取样")
        for b, kind in ((self.btn_crop, RoiKind.OUTPUT_CROP),
                        (self.btn_area, RoiKind.IMAGE_AREA),
                        (self.btn_base, RoiKind.BASE)):
            b.setCheckable(True)
            b.clicked.connect(lambda _c=False, k=kind: self._toggle_draw(k))
            lay.addWidget(b)
        self.btn_del_roi = self._tool_button(icons.trash(), "删除当前选中对象")
        self.btn_del_roi.clicked.connect(self._delete_active_roi)
        lay.addWidget(self.btn_del_roi)
        self._vsep(lay)
        for sig, icon, tip in ((self.crop_panel.rotate_requested,
                                icons.rotate_ccw(), "逆时针 90 度"),
                               (self.crop_panel.rotate_requested,
                                icons.rotate_cw(), "顺时针 90 度")):
            pass
        self.btn_ccw = self._tool_button(icons.rotate_ccw(), "逆时针旋转 90 度")
        self.btn_ccw.clicked.connect(lambda: self._rotate(-90))
        self.btn_cw = self._tool_button(icons.rotate_cw(), "顺时针旋转 90 度")
        self.btn_cw.clicked.connect(lambda: self._rotate(90))
        self.btn_mirror = self._tool_button(icons.mirror_h(), "水平镜像")
        self.btn_mirror.clicked.connect(self._mirror)
        for b in (self.btn_ccw, self.btn_cw, self.btn_mirror):
            lay.addWidget(b)
        self._vsep(lay)
        self.btn_fit = self._tool_button(icons.zoom_fit(), "适合窗口")
        self.btn_fit.clicked.connect(self.view.fit)
        self.btn_100 = self._tool_button(icons.zoom_100(), "100% 检查")
        self.btn_100.clicked.connect(self.view.zoom_100)
        lay.addWidget(self.btn_fit)
        lay.addWidget(self.btn_100)
        self._vsep(lay)
        self.btn_autocrop = self._tool_button(icons.target(), "重新检测三类区域")
        self.btn_autocrop.clicked.connect(self._auto_crop)
        lay.addWidget(self.btn_autocrop)
        lay.addStretch(1)

        # 转换工具条
        self.bar_convert = QWidget()
        lay2 = QHBoxLayout(self.bar_convert)
        lay2.setContentsMargins(0, 0, 0, 0)
        lay2.setSpacing(t.SPACING["xs"])
        self.btn_original = self._tool_button(icons.compare(), "按住看原片（负片）")
        self.btn_original.pressed.connect(lambda: self._hold_original(True))
        self.btn_original.released.connect(lambda: self._hold_original(False))
        lay2.addWidget(self.btn_original)
        self.btn_compare = self._tool_button(icons.compare(), "并排前后对比")
        self.btn_compare.setCheckable(True)
        self.btn_compare.toggled.connect(self._toggle_compare)
        lay2.addWidget(self.btn_compare)
        self.btn_fit2 = self._tool_button(icons.zoom_fit(), "适合窗口")
        self.btn_fit2.clicked.connect(self.view.fit)
        self.btn_1002 = self._tool_button(icons.zoom_100(), "100% 检查")
        self.btn_1002.clicked.connect(self.view.zoom_100)
        lay2.addWidget(self.btn_fit2)
        lay2.addWidget(self.btn_1002)
        lay2.addStretch(1)
        lbl_hint = QLabel("自动结果仅为起点，可继续微调")
        lbl_hint.setProperty("caption", True)
        lay2.addWidget(lbl_hint)

        self.viewbar.layout().addWidget(self.bar_crop)
        self.viewbar.layout().addWidget(self.bar_convert)
        self.bar_convert.setVisible(False)

    def _tool_button(self, icon, tooltip: str) -> QToolButton:
        b = QToolButton()
        b.setIcon(icon)
        b.setToolTip(tooltip)
        b.setMinimumSize(theme.SIZES["min_target"], theme.SIZES["min_target"])
        return b

    @staticmethod
    def _vsep(lay):
        line = QWidget()
        line.setFixedWidth(1)
        line.setStyleSheet(f"background: {theme.COLORS['separator']};")
        lay.addWidget(line)

    def _toggle_draw(self, kind: RoiKind):
        self.view.set_draw_mode(
            None if self.view.draw_mode() is kind else kind)
        self._sync_draw_buttons()

    def _sync_draw_buttons(self):
        cur = self.view.draw_mode()
        for b, kind in ((self.btn_crop, RoiKind.OUTPUT_CROP),
                        (self.btn_area, RoiKind.IMAGE_AREA),
                        (self.btn_base, RoiKind.BASE)):
            b.blockSignals(True)
            b.setChecked(cur is kind)
            b.blockSignals(False)
        self.crop_panel.set_draw_mode(cur)

    # ---------------------------------------------------------------- 模式
    def page_count(self) -> int:
        return 2

    def page_title(self, index: int) -> str:
        return ("裁切", "转换")[index] if 0 <= index < 2 else ""

    def current_page(self) -> int:
        return self.right_stack.currentIndex()

    def _set_mode(self, i: int):
        self.right_stack.setCurrentIndex(i)
        (self.btn_mode_crop, self.btn_mode_convert)[i].setChecked(True)
        self.bar_crop.setVisible(i == 0)
        self.bar_convert.setVisible(i == 1)
        # ROI 叠加层只在裁切模式显示（转换图上叠裁切框没有意义）
        self.view.set_rois_visible(i == 0)
        self._compose_view(fit=True)

    def _hold_original(self, on: bool):
        """按住看原片（负片）；松开回到合成正片。"""
        if self._small is None:
            return
        if on:
            self.view.set_image(self._small)
        else:
            self._compose_view()

    def _toggle_compare(self, on: bool):
        self.convert_panel.btn_compare.blockSignals(True)
        self.convert_panel.btn_compare.setChecked(on)
        self.convert_panel.btn_compare.blockSignals(False)
        self._compose_view(fit=on)

    def _show_preview_image(self, arr01):
        """转换面板的显示回调：中央常驻 viewer 直接显示渲染结果。

        裁切模式下 viewer 显示原负片+ROI；转换模式显示渲染正片。
        同一个 QGraphicsView，切换的是内容而不是控件（Lightroom 语义）。
        """
        self.view.set_image(arr01)
        self.view.fit()

    # ---------------------------------------------------------------- 导入