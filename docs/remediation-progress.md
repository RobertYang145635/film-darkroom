# 整改进度（负片软件整改_AI执行规格.md §5 固定格式）

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
