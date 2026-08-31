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
