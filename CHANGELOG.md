# 变更日志

> 说明：v6 / v6.1 / v6.2 此前只存在于规格文档与工作区，未登记进本日志，
> 导致"从 README/CHANGELOG 入口进来会拿到错误的当前范围"（2026-09-07 审查 §4.1）。
> 以下条目按已落地代码补记，不重写历史。

## 冷读审查第二轮（开发版，未发布）— 2026-09-09

依据：`质量审查报告_20260909.md`。原计划四路独立冷读，但本会话 subagent 的
spawn/fork **两条路径都返回 `subagent run failed`**（与交接 §7 的故障同形态），
故为单观察者审查——覆盖面不如四路，不应当作"四路都过了"。
基线：审查前 433 passed → 审查后 **440 passed**（+7 回归锁）· ruff 无豁免全绿。

- **导入不再冻住窗口**：`load_files`（「导入文件…」与双击打开）此前在 **GUI
  线程**逐文件 SHA-256——实测真实样片 NEF 55 MB ≈ 80 ms/张、16 位 TIFF
  274 MB ≈ 400 ms/张，选 20 个 TIFF 约 8 秒「未响应」且横幅不重绘。
  现默认经 `_start_import` 走后台线程，`background=False` 保留同步语义给脚本。
- **导入忙碌守卫生效**：`load_folder` 丢弃 `_busy_enter("import")` 返回值，
  连点两次会并发跑两遍导入、后完成的会话**无声覆盖**前一个；现在忙则拒绝并
  横幅说明（不给不成立的承诺，与 L1-② 同口径）。
- **CI 修正（此前从未跑过：`git remote -v` 为空）**：workflow 装了 Qt 的
  系统库却从不装 PySide6（`-e .` 不含 extras）。实测用 `sys.meta_path`
  屏蔽 PySide6 复现该环境：收集阶段 4 个 error，配合旧 `--maxfail=1`
  → **一个测试都没跑**。改为 `-e ".[gui,dev]"` + 新增「GUI 测试层自检」一步
  （extra 再被删时大声失败，而不是 12 个 Qt 测试模块静默 skip 换假绿）+
  去掉 `--maxfail=1` + `timeout-minutes` + pip 缓存。
- **清除区域后分析必须失效（v6.2 F2 补漏）**：F2 的规则只落在键盘删除路径
  （`_on_roi_deleted` 的 `_changes_statistics` 判据），右栏「清除」与
  「重置几何」同样清掉片基取样区/有效画面区却**不重算**，界面继续显示按
  已删除区域算出的 Dmin/来源/置信度。两处统一到同一条判据。
- **光标态**：退出「忙碌」时只认绘制模式，中性点救援的十字准星被吞成箭头
  （拾点模式仍在，用户以为已退出）；改为同时认绘制/拾点/涂抹。
- **死代码与假签名**（逐个 grep 证明 0 引用才动手）：
  `RightPanel.set_fine_rotation`（只给一个从不被读取的字段赋值、从不动滑杆——
  真正生效的是 `AnalysisSection.set_fine_rotation`，属误导性陷阱）、
  `RightPanel.set_tone_intent`、`PreviewView.set_image(flip_y=)` 死形参、
  `process_image` 里 `if ...: pass` 脚手架残留；`load_folder` 的假注解
  `-> ImportSession`（实为 Thread）与 `_import_done` 未用形参改诚实。
- **登记不修**：`docs/blockers.md` 新增 **B-4** —— 可写状态根
  `storage.REPO_ROOT = parents[2]` 只在仓库布局下成立（`pip install .` 算到
  `<venv>/Lib`，PyInstaller onedir 算到安装目录本身，`Program Files` 下
  用户确认标注会写失败）。改道属产品决定，不在审查轮里静默改。
- **自我推翻两条**（沿用交接 §5 的传统）：`slider_fine` 是 `sliderReleased`
  才 emit，不存在"拖动期间几十次 fsync"；`neutral_pick_gain/delta/intent`
  三兄弟是研究面 API（UI 走 `neutral_pick_intent_exact`），不是死代码。

## 结构整改清账（开发版，未发布）— 2026-09-08 Round 6

依据：`docs/交接_20260908.md` §12 未完成清单全部清账；13 笔提交（门禁卫生 →
原子落盘 → 表示归属 → 算法 → UI 线程纪律 → 死模块 → 测试文档 → 上述五项）。

- **导出输入快照冻结（交接 §4-1，最高危）**：整卷导出启动前在主线程一次性
  冻结全部逐帧输入（几何深拷贝、微调、意图查表、批次编码假设、卷统计），
  后台 worker 全程零读零写活对象；状态与采集指纹由主线程按 items 回写
  （`pipeline/roll_export.snapshot_frame / freeze_export_inputs /
  apply_status_writeback`）。回归锁：`test_export_snapshot_20260908.py`
  —— 导出中途改活对象几何/勾选片边，成品仍用启动前输入。
- **process_image 死参数清理（§4-2）**：`film_type` / `calibration`
  （34 形参中 AST 核实从未被读取的两个）随 15 处仪式性调用点一并移除；
  sidecar/RollCalibration/FilmProfile 的同名真形参不受影响。
- **反向依赖清零（§4-3 前半）**：`RenderIntent` 及色温/影调数学整体下沉
  `library/render.py`（`pipeline/render.py` 变兼容重导出壳，io/sidecar
  不再上行）；监督解混拟合下沉叶子 `metrics.py`；`calibration/neutral_stats`
  移入 pipeline（唯一 src 调用方是研发脚本）；冷读新发现 `GENERIC_PRIOR`
  在 roll_analysis 是**独立复制的字面量**而非别名，已改真别名。
  分层门禁 ALLOW 只剩登记在案的 `profiles/profile → pipeline.params`
  （放错目录而非倒置，决定不搬家）。
- **报告 schema 统一（§4-3 后半）**：batch 与 roll 报告共用核心键集
  （schema_version=2 / kind 判别 / summary / summary_complete / items
  最小契约）；roll-report 1→2。
- **MainWindow 拆分（§4-4）**：93 方法 / 1996 行 → 25 方法 / 541 行；
  StateStore（模块级持久状态辅助）、FrameLoader（导入/会话/帧加载/卷分析
  装配）、GeometryBridge（预览管线/即时合成/草稿读写/ROI 事件）、
  IntentController（候选/微调/拾点救援）、ExportCoordinator（整卷导出）
  各自成模块，方法原文搬移；`mw.process_image` monkeypatch 契约经
  `ui_process_image` 保持有效。顺手统一 `_abs` 最后一处 `parents[3]`。
- **C8 实测收口**：`test_c8_hold_orig_guard.py` 证明按住按钮期间
  `isDown()` 为真（守卫前提成立），且按住拒绝、松手恢复。
- **门禁**：分层测试 ALLOW 收紧；ruff `select=["E9","F"]` 无豁免维持；
  全量基线 427 passed（真实样片回归层含内）。

## v6.2 交互与标定闭环修复（开发版，未发布）— 2026-09-08

规格：`负片裁切与去色罩_AI执行规格_v6.2_交互与标定闭环修复.md`；
审查依据：`docs/审查_20260907_v6.2与救援实现核查.md`。

- **写入物理隔离（阶段 A）**：新增 `negative_lab/storage.py`，标注目录与 GUI
  草稿目录统一解析（显式改道 > 环境变量 > 仓库默认）；新增 `tests/conftest.py`
  会话级隔离，测试结束核对真实目录哈希未变，被污染即判失败。
  冒烟脚本此前只改标注目录、把草稿写进真实 `outputs/drafts/`（审计 F6）已封堵。
- **F1 冷启动**：帧加载先在后台解析持久状态，用户标定真正进入算法输入
  （此前 `ann=None` + 空几何，算法跑的是自动检测，界面显示的却是用户标定）。
- **F2 分析失效**：有效画面区编辑与片基取样区一样触发重算；输出裁切保持
  零管线快路径。
- **F3 三个坏按钮**：重新检测 / 候选 / 同步作用域不再调用不存在的
  `_set_banner_status` / `convert_panel`。
- **F4 状态选择**：新增 `calibration/state.py`（确认基线 vs 工作草稿六种组合，
  含内容指纹与草稿修订号）；草稿落盘携带 `base_annotation_sha` +
  `draft_revision`；旧草稿无来源指纹 -> 判冲突、确认基线生效、草稿原地保留。
- **F5 拾点坐标**：`_compose_view` 显式记录显示图在整幅技术正片中的原点
  （`_view_origin` / `_view_space`），拾点先换算再采样；更正研究笔记里
  "场景坐标零换算"的错误断言。
- **F9 阶段门控**：离开调整阶段清除拾点态；分析阶段点击拒绝按正片像素求解。
- **导出计数完整性**：`export_roll_frames` 增加 `cancelled` 计数并返回 `items`
  与 `summary_complete`；取消的帧不再静默消失（gui_smoke 曾因此 KeyError）。
- **一致性度量指向输出**：`roll_consistency.py` 落盘逐帧**实际施加**的
  dmax/temperature/tint 及散布（`applied_consistency`）；离群帧带文件名；
  NaN 一律转 null（此前产出不是合法 JSON）；卷名可显式指定、相对路径按仓库根解析。
- **测试可信度**：新增 `tests/unit/test_v6_2_fixes.py`（27 项，覆盖 F1-F5、F9、
  导出计数、存储优先级）；`test_rescue.py` 增加**整图对照参考**的两项断言——
  此前救援残差只在"被求解的那个 patch"上度量，属于自证（审查 §3.5）；
  新增会话级 QApplication 持有，修掉全量测试在 offscreen 下的非确定性崩溃。

实测证据：`outputs/v6_2/20260908-state-pick-fixes/`（一致性 5 帧实测：
de_mean 19.5、片基漂移 0.0057 decade、**逐帧 dmax 散布 0.241** —— 同卷片基
高度一致而软件逐帧施加的白点差 24%，正是过去没有指标暴露的问题）。

**仍未完成（不许谎报）**：真实 Windows 桌面人工流程验收（§8 阶段 D）、
人眼判定基准集（`decisions.json` 仍为空）、方向根因确认。

## v6.1 质量修复（开发版，未发布）— 2026-09 上旬

规格：`负片裁切与去色罩_AI执行规格_v6.1_质量修复.md`。
审计：`outputs/audit_20260907/v6.1_completion_audit.md`（0 致命 / 6 重要 / 2 轻微）。

- 技术正片与显示层分离：`keep_rebate`（保留片边）贯穿草稿、显示与导出；
- 右栏显示卷级片基一致性（实测帧色度极差，generic_prior 不参与统计）；
- 中性点单点救援（`pipeline/rescue.py`，(温度, 色调) 二自由度族内精确求解）
  接入画布拾点、撤销栈与可信度警告；
- 遗留：F1-F8 状态一致性问题（本轮 v6.2 处理其中 6 项 + 新增 F9）。

## v6 UI 优先（开发版，未发布）— 2026-09 上旬

规格：`负片裁切与去色罩_AI执行规格_v6_UI优先.md`（其 §5.6 部分条款已被用户作废）。

- 取消"裁切页 / 转换页"双页，改单工作区 **分析 / 调整** 两阶段
  （Lightroom 式 Convert/Edit）；分析阶段显示原负片、调整阶段显示缓存正片；
- 三类 ROI 同屏叠加编辑；片基变化才走慢路径重算，裁切与微调零管线调用；
- 三候选（保留氛围 / 中性 / 偏暖）+ 少量微调，风格只存在于 RenderIntent。

## v5 P0（开发版，未发布）— 裁切与无色卡去色罩

规格基线：`负片裁切与去色罩_AI执行规格_v5.md`（唯一生效基线，基线提交
`e795bbbf49a3ac7b782b2f569629adae97c31281`）。v1-v4 规格全部降级为历史资料。

### 产品范围（v5 §0.2 用户决策落地）

- **普通转换不需要色卡**：删除 `needs_calibration`/`needs_regions` 产品状态
  与全部"请提供色卡/必须加载校准/没有中性点"阻断文案（§8.1）；
- 黑白/反转片移出 P0，不再是发布门禁（§0.2.5）；
- C-41 与 ECN-2 同为 P0，且均有真实可解码样片（ECN-2 为 Z8 高效 NEF 的
  16 位 TIFF 导出，capture_type 如实标记 `camera_raw_export_tiff`）；
- 主工作区只有 **裁切 / 转换** 两个一级页面；Dmin/Dmax/矩阵/指数/校准/
  除尘等控件不再暴露给普通用户。

### 新增

- `library/`：`ImportSession`/`RollGroup`/`FrameRecord`/`GeometryRegions`/
  `RollAnalysis`/`CandidateSet` 六个可序列化对象 + 手动分卷（建立/重命名/
  拆分/移动）+ 状态机（§3.3，阻断只剩 decode_error/processing_error）；
- 自动采集指纹（§4.4）：不含文件名/卷名，metadata 不足时只降置信度；
  卷内冲突帧标记 + 一键拆卷建议；跨帧片基漂移检测（容差预先冻结）；
- 三类独立 ROI（§3.2）：`output_crop_roi` / `image_area_roi` /
  `base_sample_rois[]`（可多个）；标注 schema v3，旧 v1/v2 只读迁移并降级
  draft（用途必须重新确认）；
- 裁切页（§5）：图标工具栏、三类 ROI 的八向手柄/越界钳制/多片基区、
  细微旋转 ±10°、水平镜像、归一化相对位置的整卷同步；
- 卷级片基（§6.2）：多取样区密度域稳健估计（q75 中心 + decade 容差）、
  卷级汇总、无片基帧自动继承卷级模型、全卷失败回退**版本化通用先验**
  （全卷共用一个曝光尺度，非物理测量、如实标记）；
- 三个自动候选（§6.5-6.6）：保留氛围 / 中性 / 偏暖，共享同一技术正片；
  多估计器（SoG p=1/2/6、白点、Gray-Edge）方向与幅度比较、高饱和/单色/
  夜景降置信、分歧过大回退 preserve、色温/色调硬上限并记录触发；
- `RenderIntent` 基础微调（§7.2）：色温/色调（Bradford 色适应）、曝光、
  对比（受保护中间调支点）、高光/阴影/白色/黑色（非负斜率累积的单调
  影调曲线）、自然饱和度/饱和度；默认恒等逐比特不变技术正片；
  单项重置/全部重置/复制粘贴/勾选式整卷同步；
- sidecar **schema v4**（§3.4）+ v1-v3 只读兼容迁移；记录三类 ROI、
  采集指纹、片基来源与 provenance、三候选、微调、技术/渲染独立诊断、
  git commit 与规则版本；
- 整卷两遍处理与事务批量导出（§12 P0-7）：TIFF16 + JPEG8 + sidecar +
  `roll-report.json`，唯一命名、失败回滚、取消、源文件只读校验；
- CLI 子命令 `import/invert/batch/roll/profile`（`--base-sample` 可重复）；
  混合文件夹导入会话 JSON。

### 算法修正（真实样片实测教训）

- 片基污染判定改为**密度域**（|log10 t - log10 center|）：相对透射率容差
  在橙罩 B 通道（t≈0.03）上把正常颗粒误判成污染，真实卷 47-93% "污染"
  全是误判，曾导致整卷退化通用先验；
- 片基中心用 **q75**（未曝光片基物理上是条带最亮均匀区）+ 最亮 25% 兜底；
- 自动片基条带**收窄到最亮稳定子带**（`refine_base_roi`），避开齿孔/边码；
- flat-field 只保留**相对梯度**（片基取样区内每通道中位 = 1）：旧实现把
  窄条带二阶外推到全幅，实测把 Dmin 抬高 1.44 倍；
- 技术正片剪裁统计口径移到**软趾之后、最终 clamp 之前**（§8.2 字面），
  并按最终 image_area mask 逐通道统计；输入端剪裁降 warn（采集事实）；
- 通用模型资产（§6.3/§9.4）：无监督密度解混训练（82 帧、相关性 4.14→1.52、
  留出卷验证），资产哈希/单调性/条件数校验，失败自动回退确定性密度模型。

### 已知限制（诚实声明）

- 无真实扫描仪 TIFF / DNG / 第二套设备样片：扫描仪路线、DNG 路线、
  跨设备泛化一律未验证（docs/support-matrix.md）；
- ECN-2 证据来自相机 RAW 的 Lightroom 导出 TIFF（sRGB 显示编码线性化），
  不是扫描仪原生线性数据；
- 全部 169 帧基准的 ROI 仍为自动检测/历史草稿（`rois_user_confirmed=0`），
  需用户在裁切页确认后重生成验收清单；
- 逐卷联系表为预览分辨率（最长边 1200），最终验收以 GUI 全分辨率导出为准；
- 视觉验收（逐张接受/相对 NLP 判定）只能由用户签字，本版本交付时待定。

## v0.1.0（历史版本）

密度域 + 每通道幂指数混合模型；单张 GUI / 批处理 / CLI；Nikon Z8 高效
NEF（AV1）无法解码（LibRaw 0.22.1），需导出 16 位 TIFF——v5 已把该路线
纳入 P0 输入矩阵。
