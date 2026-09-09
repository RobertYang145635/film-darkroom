# 负片裁切与去色罩 AI 执行规格 v6.1（质量修复）

> 适用仓库：/src/film-darkroom  
> 执行者：DSH 或其他能直接修改仓库、运行 PowerShell 和测试的 AI agent  
> 基线：执行开始时以 git rev-parse HEAD 输出为准；旧报告不能替代本轮证据  
> 目标：修复 v6 当前的确定性错误、状态闭环和验收证据，不进入新的去色罩算法实验  
> 色卡：不是普通预览、转换、导出或发布的前置条件

## 0. 当前结论

当前 v6 不能标记为完成。

本轮复核得到的事实：

~~~text
HEAD：3ab39a0f98a806ccc8f104f8749b76887be0ec20
全量测试：297 passed, 1 failed
失败：tests/regression/test_reference_quality.py::test_auto_crop_iou[G200-4.NEF]
失败值：IoU = 0.7752937022605593，门槛 >= 0.95
~~~

已复现的用户可见问题：

1. 已有完整 ROI 的帧重新载入时，把 (H,W,3) 数组直接传给只接受 (H,W) 的 GeometryRegions.normalized()，导致 ValueError: too many values to unpack。
2. 高清放大调用不存在的 QGraphicsView.m11()，并且把 worker 返回的 {"ok": True, "value": ndarray} 当作数组传给 set_image()；调整阶段还可能把原始负片显示成正片。
3. “保留片边”只写入不存在的动态属性 FrameRecord.keep_rebate，没有完整传入导出链路；勾选后可能与未勾选得到相同文件。

旧的 outputs/v6_ui/handover-report.md、旧提交号和旧测试数字不能作为当前证据。

## 1. 执行纪律

执行前运行并保存输出：

~~~powershell
Set-Location /src/film-darkroom
git status --short
git rev-parse HEAD
.venv\Scripts\python.exe -m pytest -q
~~~

硬约束：

- 不得 git reset --hard、git checkout --、批量删除未知文件或覆盖用户样片。
- 原始 NEF、DNG、TIFF 和参考图只读；新截图、报告和临时文件只写 outputs\v6_ui\ 或测试临时目录。
- GUI、CLI、批处理和测试继续调用同一个 src\negative_lab\pipeline\service.py::process_image()；不得在 UI 复制反相或去色罩算法。
- 不得删除或跳过 G200-4.NEF，不得把 IoU 门槛改低，不得把人工框直接改成自动框，不得修改 golden 图来掩盖失败。
- 每个阶段先补一个能复现问题的测试，再改生产代码，再运行该阶段门禁。
- 本轮不修改 Dmin/Dmax、密度符号、颜色矩阵、候选排序、默认渲染参数或 generic prior 算法。

## 2. P0-A：修复 RGB 尺寸解包崩溃

### 目标

任何以全分辨率 RGB 数组作为 source 的路径都必须使用 source.shape[:2]；任何 (H,W) 尺寸元组仍可正常工作。ann_source is not None，不得对 ndarray 使用真假判断。

### 责任文件

- src\negative_lab\ui\main_window.py：_geometry_for_job()，当前约第 49-62 行。
- tests\unit\test_v6_ui_contracts.py 或新增 tests\unit\test_v6_geometry_reload.py。

### 实现要求

增加统一的 _shape_hw(value)：

~~~python
def _shape_hw(value):
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    if len(value) >= 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"invalid image/size value: {value!r}")
~~~

_geometry_for_job() 先得到 small_hw = _shape_hw(small)、source_hw = _shape_hw(source)，然后只把尺寸元组传给 normalized() 和 from_normalized()：

~~~text
geo.normalized(source_hw)
GeometryRegions.from_normalized(..., small_hw)
g.normalized(_shape_hw(ann_source) if ann_source is not None else small_hw)
~~~

不要把完整 ndarray 继续传给 normalized()，不要使用 if ann_source。

### 必须新增的测试

1. 完整三类 ROI + RGB source 输入不会抛异常，输出 ROI 在 small.shape[:2] 内。
2. 构造完整 FrameRecord，走“重新打开已有完整 ROI”的路径，断言无 ValueError，三类 ROI 数量和 source 坐标语义不变。
3. 片基 ROI 修改后触发 _refresh_preview()，worker 完成后无错误 banner，草稿仍存在；必须等待 worker 并处理 Qt 事件。

### P0-A 门禁

~~~powershell
.venv\Scripts\python.exe -m pytest -q tests\unit\test_v6_ui_contracts.py tests\unit\test_v6_geometry_reload.py
.venv\Scripts\python.exe -m pytest -q
~~~

全量测试若仍为 1 failed，只能把剩余失败记录为 P1，不能报告为全绿。

## 2.1 P0-A2：修复旋转/镜像后底图与 ROI 错位

当前 _show_frame() 已用 OrientTransform 把 source ROI 映射到 oriented 坐标，但分析阶段的 _set_stage()、_frame_done()、_show_frame() 和“按住看原片”仍可能直接显示未经方向变换的 self._small。rotation_degrees 非 0 或 mirror_horizontal=True 时，底图和框会使用不同坐标系。

实现要求：

- 提供唯一的“分析阶段显示图”函数，对 small 预览应用与当前 FrameRecord.geometry 完全相同的 OrientTransform。
- _set_stage()、_frame_done()、_show_frame()、_hold_original() 和缩放恢复路径都调用这一函数，不得各自直接 set_image(self._small)。
- 不复制旋转/镜像公式，继续使用 pipeline.geometry.OrientTransform。
- 变换后的图像尺寸、ROI 映射和 sceneRect 必须一致；90/270 度时宽高交换不能靠硬编码修补。

新增测试：0/90/180/270 度和水平镜像下，使用带明显方向标记的非正方形合成图；断言底图方向正确、ROI 覆盖同一源区域、保存再重开误差不超过 1 个源像素。

## 3. P0-B：修复高清预览 API、结果解包和竞态

### 首选交付策略

v6.1 首要要求是不能显示错误内容。在真正实现同场景高清 tile 前，允许并推荐关闭后台高清替换，只保留稳定的缩放、平移和 100% 检查。不能把全分辨率图直接塞进使用降采样 ROI 坐标的场景。

可选安全开关：

~~~python
ENABLE_HD_PREVIEW = False
~~~

关闭时 _on_zoom_changed() 只更新缩放状态，不启动 _hd_tile_job()；不要删除普通缩放。

### 无论是否关闭，都必须修复

责任文件：

- src\negative_lab\ui\preview_view.py：当前约第 141-146、655-663 行。
- src\negative_lab\ui\main_window.py：当前约第 987-1028 行。
- src\negative_lab\ui\workers.py：保持统一 worker 返回契约。

修复项：

1. wheelEvent() 和 current_zoom() 使用 self.transform().m11()，不能调用 self.m11()。
2. fit()、zoom_100() 和滚轮缩放都发出同一个 zoom_changed 信号，标签、恢复小图和内部缩放值一致。
3. _hd_tile_done() 先处理 {"ok": ..., "value": ...}：失败显示可读错误并保留当前小图；成功只传 result["value"] 给 set_image()。
4. 使用递增 task_id/generation，并同时检查当前帧、阶段和当前缩放。缩回后再次放大时，旧任务不得覆盖新状态。
5. 全分辨率数组的 sceneRect 与 small ROI 坐标不能混用。要么按 small 坐标生成同尺寸 tile，要么同步映射所有 ROI、滚动中心和场景变换；只替换一张全帧图不算修好。

### 调整阶段硬要求

如果重新打开高清替换，调整阶段 worker 必须调用同一 process_image(loaded.rgb, geometry=..., render_intent=..., candidate=..., keep_rebate=True)，取正确正片结果后再编码；不能用 tr.apply(loaded.rgb) 冒充正片。若处理时间或内存不满足，保持高清替换关闭。

### P0-B 门禁

新增测试至少覆盖：

- current_zoom() 不抛 AttributeError；
- worker 成功结果只传 result["value"]，失败结果不替换当前图；
- fit()/zoom_100() 后缩放信号和标签同步；
- 帧切换、缩回或再次放大后旧任务结果被丢弃；
- 调整阶段不会把负片数组当正片上屏。

~~~powershell
.venv\Scripts\python.exe -m pytest -q tests\unit\test_v6_ui_contracts.py
.venv\Scripts\python.exe -m pytest -q
~~~

## 4. P0-C：打通“保留片边”状态和导出

### 规范化数据落点

当前 UI 有意写入 self.current.keep_rebate，但 FrameRecord 没有该字段。v6.1 以 FrameRecord.keep_rebate: bool = False 作为当前帧的用户选择；GeometryRegions 继续只保存 ROI 与方向，不新增第二个同名输入状态。

### 实现要求

1. src\negative_lab\library\models.py
   - 在 FrameRecord 增加 keep_rebate: bool = False。
   - to_dict() 显式写入布尔值，from_dict() 对旧文件默认 False。
   - 非 bool 值不能静默当真。
2. src\negative_lab\ui\main_window.py
   - _on_keep_rebate(on) 显式检查 self.current is not None 后写入。
   - 当前帧加载/切换后，用右栏公开 setter 反映 rec.keep_rebate，避免串用上一帧状态。
   - 技术缓存可以恒用 keep_rebate=True，因为它是整幅技术正片；删除 _refresh_preview() 中无效的 keep 变量。
   - 显示层按当前帧状态决定是否裁切，不得让片边进入 image-area 统计。
3. src\negative_lab\pipeline\roll_export.py 当前约第 121-124 行必须传 keep_rebate=bool(rec.keep_rebate) 给 process_image()。不得在导出层自己裁切数组。
4. 重开持久化必须真实覆盖应用使用的来源。当前 UI 草稿来自 Annotation，而不是自动保存 ImportSession。推荐给 Annotation 增加可选 keep_rebate: bool = False，to_dict/from_dict 兼容旧 schema；保存草稿和用户确认标注时写入，加载时恢复到 FrameRecord.keep_rebate。若改走 session 保存，必须证明应用重启后确实从 session 恢复，不能只测内存对象。

### 必须新增的测试

- FrameRecord.to_dict()/from_dict() true/false round-trip。
- 勾选与未勾选分别导出，输出尺寸不同且与 GeometryRecord.keep_rebate 一致。
- 两次导出的 image-area、片基和门禁统计一致，片边不能污染统计。
- 保存草稿/确认标注，关闭并重新加载后勾选状态恢复。
- 未勾选使用 output_crop_roi；没有 output crop 时按现有规则进入 needs_crop_review，不得偷偷用 image area 代替用户构图。

### P0-C 门禁

~~~powershell
.venv\Scripts\python.exe -m pytest -q tests\unit\test_service.py tests\unit\test_export_transaction.py tests\unit\test_v6_ui_contracts.py
.venv\Scripts\python.exe -m pytest -q
~~~

## 5. P1：诊断并修复自动裁切失败，不得先改阈值

当前失败：

~~~text
样片：G200-4.NEF
自动 image_area：(1166, 1089, 7172, 5522)
人工 image_area：(1192, 344, 7128, 5144)
IoU：0.775
~~~

### 第一步：生成证据

新增一次性诊断脚本或测试辅助，所有输出写入 outputs\v6_ui\auto_crop_diagnostics\：

- 输入哈希、原图尺寸、降采样 step。
- frame、base、image_area、edge、confidence、warnings。
- 人工框与 IoU。
- 亮度行/列 profile、平滑 profile、阈值、候选端点和最终端点。
- 带自动框和人工框的缩略图 overlay。

不要把完整原片复制到报告；overlay 只写输出目录。

### 第二步：确认目标定义

改算法前必须在诊断报告回答：

1. 自动 image_area 与人工框是否都是排除片边后的有效画面，还是其中一个实际代表胶片 frame？
2. G200-4 的 x 方向是否被齿孔/边码亮带误判为片边，还是人工框包含了应排除区域？
3. base_edge、旋转方向和 source/oriented 坐标是否一致？

没有证据回答前，不得改 detect_image_area() 阈值。

### 第三步：最小修复与回归

只根据已确认原因修一个问题，优先保证：

- 只从边缘连续结构裁切，不能因内部亮场景裁掉有效画面。
- 片基/齿孔亮带与画面端点必须分离。
- 自动结果低置信时返回 needs_crop_review，不能静默宣称通过。
- source/oriented 转换使用现有 OrientTransform，不复制坐标公式。

每次算法修改必须运行全部真实样片和合成边界测试，覆盖暗场、亮场、黑边、无片基带和四种方向。达不到 G200-4 门槛时保留失败并报告原因，不得伪造通过。

## 6. P2：同步脚本、文档和证据

必须检查并按当前 v6 API 更新：

- scripts\gui_smoke.py：删除 crop_page、convert_page、_show_page() 旧调用；若读取整卷报告，使用真实返回结构，不能读取不存在的 rep["items"]。
- scripts\diag_capture_ui.py：删除 _show_page() 旧调用，使用当前 stage API。
- docs\user-quickstart.md：删除 v5 两页面描述，说明当前是同一工作区的 stage。
- docs\v6-ui-redesign-plan.md：区分已实施与待实施。
- outputs\v6_ui\baseline.txt、ui_interaction.json、handover-report.md：用当前 HEAD、当前测试结果、失败项、截图路径和未完成项重生成。

真实 Windows 桌面至少确认中文标题、ROI 标签、错误 banner、保存状态和按钮可读；离屏方框不能被写成 Windows 已验证。

## 7. 停止条件

任一条件触发就停止扩展功能并报告：

- 完整 ROI 重开仍异常。
- 高清显示内容与当前 stage 不一致，旧任务覆盖新状态，或 worker 错误被吞掉。
- 勾选“保留片边”后导出文件未改变尺寸/sidecar，或统计被片边污染。
- G200-4.NEF 仍低于 IoU 门槛且原因没有证据。
- 全量测试不是全绿，但报告写成“通过”。
- 自动脚本仍依赖 v5 页面 API。
- 普通转换被要求先提供色卡、胶片型号、相机 profile 或扫描仪 profile。

停止后不要进入新的 NLP、去色罩矩阵、LUT、色卡或神经网络实验。

## 8. 最终报告格式

每次执行结束必须填写：

~~~text
阶段：P0-A / P0-B / P0-C / P1 / P2
改动文件：...
新增或修改测试：...
通过命令：...
失败命令：...
当前 HEAD：...
证据文件：...
未完成项：...
是否触发停止条件：是/否（若是，引用第 7 节具体条目）
~~~

只有确定性错误修复、保留片边导出闭环、脚本/报告同步完成，并且全量测试与真实 GUI 证据都如实通过后，才可把 v6.1 标记为完成。自动裁切仍失败时，版本必须保持“未通过”。
