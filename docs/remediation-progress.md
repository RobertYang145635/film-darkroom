# 整改进度（负片软件整改_AI执行规格.md §5 固定格式）

## 2026-09-01 复核撤销（v3 F-ALG-9，不覆盖旧日志）

- **P0-B 色彩管线：PASS -> FAIL**。撤销理由：output_color=6 实为
  rawpy.ColorSpace.ACES（实测枚举表 raw=0,sRGB=1,...,ACES=6），被错误
  命名为 camera-linear；密度输出为 ACES 基色语义却仅加 sRGB TRC 并嵌
  sRGB ICC（v3 F-ALG-1）。工作空间已改为 srgb-linear-d65-normalized
  （临时默认，待 bake-off），ACES 断言测试已删除并替换为命名枚举回归锁。
- **P0-C 自动颜色：PASS -> FAIL**。撤销理由：Dmin 被画面内 p99.5 覆写
  （v3 F-ALG-3 禁止——场景像素不得改物理基准；暗角应走 flat-field）；
  统计 gray-world 自动改写指数/白平衡属自证循环（v3 F-ALG-4）。
  两处均已移除：Dmin 纯片基来源；gray-world 降级为"建议不应用"。
- **P0-D 统一主路径：PASS -> PARTIAL**。管线统一/sidecar v2/fail 阻断
  结构有效，但其上的色彩结论随 P0-B/C 撤销失效；GUI 冒烟在历史输出
  目录上不可重复（v2 P0-I 待做）。
- QualityGate 新增 needs_calibration（v3 P0-G 部分）：彩色负片缺
  verified 标定/中性点时不再伪装为 warn。
- 三路径 bake-off 首轮（outputs/benchmarks/20260901-030604）：
  A 路径矩阵方向错误（G 通道全零）待修；B/C 正常产出对比数据。
- G200-1 Dmin 基线重测为 [0.2836,0.1093,0.0315]（sRGB-linear 工作空间、
  纯片基来源）——语义变更导致的合法重基线，非放宽阈值。

## P0-A

- 状态：PASS（附条件：发布门禁测试 test_release_gate_blocked_until_fixtures 按规格保持红色，直到用户补充黑白/ECN-2-TIFF/色卡样片）
- 修改文件：
  - 新增 `src/negative_lab/io/probe.py`（真实最小解码探针，稳定枚举 ok/unsupported/corrupt/missing_dependency/unknown）
  - 重写 `scripts/build_fixture_manifest.py`（schema v2：decode_status 来自探针；记录解码参数/位深/尺寸/设备/ICC；按卷 split）
  - 新增 `scripts/sample_intake.py`（只读接收扫描）、`scripts/build_support_matrix.py`、`scripts/annotate_preview.py`、`scripts/write_p0a_annotations.py`
  - 新增 `src/negative_lab/calibration/annotations.py` + `fixtures/annotations/*.json`（9 条，1 条 verified）
  - 重写 `fixtures/manifest.json`（169 条：ok=97 / unsupported=72）
  - 更新 `docs/blockers.md`（矛盾修复）、新增 `docs/support-matrix.md`、`docs/sample-calibration.md`
  - 新增 `tests/unit/test_fixture_manifest.py`、`tests/regression/test_decode_matrix.py`
- 实际执行命令：
  - `.venv\Scripts\python.exe scripts/build_fixture_manifest.py`（真实探针全量重建）
  - `.venv\Scripts\python.exe scripts/build_support_matrix.py`
  - `.venv\Scripts\python.exe scripts/annotate_preview.py ...`（方向/框目检）
  - `.venv\Scripts\python.exe -m pytest tests/unit/test_fixture_manifest.py tests/regression/test_decode_matrix.py -q`
- 测试结果：（见下方运行输出；发布门禁 1 红为规格要求）
- 真实样片结果：F-5 矛盾消除——5219/5207 全部 72 张由探针标 `unsupported`，
  与 rawpy 实测一致；抽查重探针与清单一致（回归锁）。
- 视觉证据路径：`outputs/ann/G200-1_overview.jpg`、`G200-1_imagearea.jpg`、
  `G200-1_orient.jpg`、`orient_grid.jpg`
- 峰值内存/耗时：全量探针重建 ~14 min（169 文件，RAW unpack 为主）
- 未验证项：8 条标注的画面框（P0-C 核对）；黑白/ECN-2/TIFF/DNG/第二设备（缺样片）
- 下一门禁：P0-B 色彩管线 ADR + color/management.py + ICC 闭环测试

## P0-B

- 状态：PASS（发布门禁测试仍按规格保持红色）
- 修改文件：
  - 新增 `docs/color-pipeline.md`（ADR：RAW 解码策略/工作空间/输入 ICC 分类/输出编码与 ICC/固定顺序）
  - 新增 `src/negative_lab/color/management.py`（RAW_DECODE_PARAMS 显式化+回读防呆、
    sRGB/Adobe/linear 传递函数、classify_tiff、sRGB ICC 装配、OutputEncoding 枚举）
  - 重写 `src/negative_lab/io/images.py`（TIFF 无证据即拒绝+显式 tiff_encoding 假设通道；
    JPEG/PNG 按 sRGB 线性化并警告；save_tiff16/save_jpeg8 强制枚举+嵌 ICC+写后重读验证；
    JPEG 4:4:4；meta 记录 sha256/编码/ICC 哈希/白平衡策略）
  - CLI（invert/batch `--tiff-encoding`）、批处理（tiff_encoding 透传+旁车色彩字段）、
    GUI（TIFF 未知编码弹窗选假设；旁车记录 input/output 色彩语义+旋转+实际裁切）
  - 新增 `tests/unit/test_color_management.py`(10)、`tests/unit/test_icc_roundtrip.py`(5)、
    `tests/regression/test_color_pipeline_real_files.py`(3)
- 实际执行命令：
  - `.venv\Scripts\python.exe -m pytest tests/unit/test_color_management.py tests/unit/test_icc_roundtrip.py tests/regression/test_color_pipeline_real_files.py tests/unit/test_export_safety.py -q`
  - `.venv\Scripts\python.exe -m pytest -q`
- 测试结果：99 passed + 1 intentional red（release gate，规格要求）。
  期间发现并修复 2 个真实缺陷：
  ① `user_mul=(1,1,1,1)` 破坏 LibRaw 内部归一化（片基 R 0.17→0.11，ADR 改为
  如实记录默认 as-shot 策略——逐通道增益在 z 归一化中抵消）；
  ② rawpy 0.27 属性名是 `gamm`，`setattr(params,"gamma",…)` 被静默忽略导致
  输出带 2.2 gamma（片基 0.172→0.399）——已加回读防呆断言，未知属性不再可能溜过。
- 真实样片结果：G200-1 片基 Dmin（verified 标注 base_roi，LINEAR 语义）
  = [0.1722, 0.1174, 0.0444]，回归锁 <5%；NLP 对照图确认含内嵌 ICC（审查证据复现）。
- 视觉证据路径：`outputs/ann/G200-1_imagearea.jpg`（区域）；ICC 证据在测试断言内
- 峰值内存/耗时：全量 33s；RAW 解码单张 ~3s（LINEAR 16b）
- 未验证项：真实带 ICC TIFF 输入（缺样片，missing-fixture）；多查看器外观一致性
  （需人工在两个 ICC-aware 查看器中比对——列入 P0-C 联系表环节）
- 下一门禁：P0-C 自动颜色先验/统计中性 + apply_geometry 统一裁切方向 +
  quality_gate + 真实质量回归 + 联系表（然后暂停请用户视觉确认）

## P0-C

- 状态：PASS（机器门禁）/ AWAITING-USER（视觉门禁，按规格必须暂停）
- 修改文件：
  - 新增 `pipeline/geometry.py`（SOURCE/ORIENTED 坐标语义唯一定义、rotate_cw、
    roi_source_to_oriented 含边缘钳制、apply_geometry+GeometryRecord）
  - 新增 `pipeline/workflow.py`（prepare_and_analyze：旋转->全图检测->裁画面->
    以裁切前片基做分析；GUI/批处理/回归共用，杜绝 I-1/I-2 类漂移）
  - 新增 `diagnostics/quality_gate.py`（结构化 ok/warn/fail + 逐通道方向比例
    的剪裁警告，F-4；批处理状态不再靠中文字符串猜）
  - 新增 `calibration/neutral_stats.py`（受限 gray-world 统计中性候选，升级链
    第 2 级，置信度 0.4 + 强制可见警告）
  - `calibration/border.py`：画面区域检测 v2（三段结构扫描 + 片基平坦度守卫 +
    齿孔双峰列修剪）；G200-1 对 verified 标注 IoU=0.9926（门禁 0.95）
  - `pipeline/auto.py`：Dmin 有效片基校正（画面内 p99.5 补暗角/导光，实测教训：
    帧外灯箱与帧内齿孔都会污染分位数）；Dmax 先验取上界补单帧低估
  - `calibration/base.py`：offset 改为固定黑位余量 log10(1+black)-0.03
    （旧"压最暗0.5%到黑位"策略压碎贴近片基的场景暗部 -> 低端剪裁 8-9%）
  - `pipeline/batch.py`：接入 workflow/quality_gate/geometry，keep_rebate 选项，
    旁车记录 geometry+quality
  - 新增 `scripts/learn_priors.py`（标定卷先验，留出隔离，版本化 JSON）、
    `scripts/build_visual_contact_sheet.py`
  - 新增 `fixtures/priors/*.profile.json` + learn-report.json；
    `docs/visual-acceptance.md`
  - 测试：`tests/unit/test_geometry.py`(8)、`tests/unit/test_quality_gate.py`(5)、
    `tests/regression/test_reference_quality.py`(7：旋转一致/IoU>=0.95/无大片边/
    剪裁<1%/Dmin参考有效/不冲淡/幂等)
- 实际执行命令：
  - `.venv\Scripts\python.exe scripts/learn_priors.py`
  - `.venv\Scripts\python.exe -m pytest tests/regression/test_reference_quality.py -q`
  - `.venv\Scripts\python.exe -m pytest -q --deselect ...test_release_gate_blocked_until_fixtures`
  - `.venv\Scripts\python.exe scripts/build_visual_contact_sheet.py`
- 测试结果：117 passed + 1 deselected（发布门禁红色为规格要求）。
  质量回归 7/7：G200-1 低端剪裁 R 0.018%/G 0.009%/B 0.6%（审查基线为
  9.5/10.0/16.2%），高端 ~0%；IoU 0.9926；幂等逐比特一致。
- 真实样片结果：G200-1 方向 180° 正确、画面裁切无片边/片基、黑位真实、
  无青蓝罩（统计中性 wb 生效）；与 NLP 相比对比度仍偏软（见联系表）。
- 视觉证据路径：`outputs/visual/G200-1_1298_G200_sheet.jpg`
  （四格：原负片|整改前基线|整改后|NLP 对照）
- 峰值内存/耗时：质量回归单张 ~205s/7 测试（全分辨率 LINEAR 解码为主）
- 未验证项：用户视觉确认（本阶段暂停点）；CN400/1299 的画面框 verified 标注
  （检测器对亮沙场景仍保守返回 None，走手动路径）
- 下一门禁：**用户确认联系表达到可继续试用水平**后才进入 P1；
  否则停在 P0-C 继续修自动颜色。

## P0-D（v2 规格）：统一 GUI/CLI/批处理主路径

- 状态：PASS
- 修改文件：
  - 新增 `pipeline/service.py`：process_image 单一管线（v2 §2.1 固定顺序：
    rotate_cw -> 全图检测 -> 裁切前片基 -> crop -> 除尘 -> auto -> invert ->
    QualityGate），ProcessResult.sidecar() 为唯一 schema **v2** builder
    （SOURCE 坐标 ROI、crop_oriented 单列、quality/ICC/raw_decode 全记录，
    coordinate_space:"preview" 消除）；force_params 支持用户显式覆盖
  - `ui/main_window.py`：load 重置全部图像状态（rotation/ROI/点/蒙版/参数/
    confidence/_pr）；verified 标注驱动初始旋转与框（F-1 证据 rotation=0
    vs 标注 180 修复）；旋转清空并重检测（I-1 两案之一）；run_auto/export
    全部走服务；gate.fail 阻断导出；配置加载按钮启用（损坏可见错误）
  - `cli.py`：invert 走服务，新增 --rotation/--frame-roi/--keep-rebate/
    --profile/--dust-auto；输出 ASCII 状态词；旁车 v2
  - `pipeline/batch.py`：走服务；**fail 在写文件前阻断、零输出**（I-3）；
    run() 收尾强制写报告（I-5/I-8）；状态 ASCII 枚举；旁车 v2 原子写
  - `ui/batch_panel.py`：行按规范化完整路径关联（同名文件错行修复）
  - `pipeline/geometry.py`：oriented_to_source 逆变换（往返测试锁定，
    期间修正 90/270 两处反解代数错误——被测试抓住）
  - 测试：`tests/unit/test_service.py`(14：旋转语义/ROI 往返/sidecar v2
    字段/无 preview/force_params/keep_rebate/fail 零输出阻断)
- 实际执行命令：pytest -q（125 passed + 1 deselected 发布门禁红）；
  compileall src scripts tests 通过；e2e_cli_smoke.py 通过（v2 断言）；
  gui_smoke.py 通过（v2 断言：rotation/geometry/quality/ICC/TIFF 嵌 ICC）
- 真实样片结果：GUI 加载 G200-1 自动应用标注 180°+verified 框，
  dmin=[0.1728,0.1181,0.0448] 与基线一致；导出裁切生效、旁车 v2
- 视觉证据路径：outputs/gui_smoke/（冒烟三件套）
- 峰值内存/耗时：全量 317s；GUI 导出全分辨率服务单次 ~90s（P2 优化项）
- 未验证项：事务导出（P0-H）；幂等临时目录（P0-I）；ColorSpace 枚举语义
  （P0-E，下一项）
- 下一门禁：P0-E RAW 色彩空间枚举修正

## v3-P0-C：flat-field 与卷级测量

- 状态：PASS
- 修改文件：
  - 新增 `calibration/flatfield.py`：blank frame -> 2D 多项式增益场；
    片基条带 -> 沿条带轴 1D 拟合且**如实声明横向不可观测（部分校正）**；
    场归一化到全局中位=1（不改 Dmin 量级）
  - 新增 `calibration/rollmeasure.py` + `scripts/measure_rolls.py`：
    卷级测量（片基统计/噪声/曲面覆盖/z 分位/超基占比/饱和占比）
  - `pipeline/service.py`：flat_mode=strip(默认)/blank/off；blank 参数；
    sidecar 记录 flatfield.coverage 与警告
  - `cli.py`：--flat-mode / --blank-frame
  - 测试 `tests/unit/test_flatfield.py`(4)：blank2D 上下片基一致<3%、
    1D 部分覆盖诚实声明、超基占比 14%->0%（阈值 z>0.04 物理正确：
    clear film z≈0 不算剪裁）、flat 后 Dmin 仍=片基真值
- 实际执行命令：pytest tests/unit tests/synthetic -q；scripts/measure_rolls.py
- 测试结果：123 passed（unit+synthetic）；开发中测试抓住 3 个自身 bug：
  1D 拟合掩码不一致、设计矩阵维度、ROI 越界切片
- 真实样片结果：4 卷测量 JSON 落盘 fixtures/measurements/；
  **ECN-2 转换卷 6/6 帧可测**（base=[0.449,0.232,0.059]，dmax_med=1.41）；
  1299 与 1298 片基统计差异大（0.178 vs 0.279）——按卷标定的必要性实锤
- 视觉证据路径：fixtures/measurements/*.measure.json
- 峰值内存/耗时：measure_rolls 全 4 卷 ~4min（ECN2 272MB/帧抽样 6 帧）
- 未验证项：真实 G200-1 上 flat 效果（P0-G 联系表时验证；无 blank frame
  样片，strip-1D 只能部分校正，符合诚实声明）
- 下一门禁：v3-P0-D backend bake-off（power-reciprocal / density-affine /
  density-curves 三后端 + 留出集对照）
