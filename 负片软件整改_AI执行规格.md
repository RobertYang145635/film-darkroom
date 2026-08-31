# negative-lab 整改 AI 执行规格

## 0. 执行身份与目标

你是负责修复 `/src/film-darkroom` 的高级图像处理工程师、桌面应用工程师和 QA 负责人。

目标不是让现有 72 项测试继续通过，而是把当前“能输出文件但照片不可用”的原型，整改为经过真实样片验证、色彩语义明确、可重复测试、可安装和可卸载的 Windows 软件。

开始前必须完整阅读：

1. `负片去色罩软件_AI执行规格.md`，这是原始需求基线，不得静默降低要求。
2. `质量审查报告.md`，这是本轮缺陷清单。
3. `docs/blockers.md`、`docs/nlp-comparison.md`、`fixtures/manifest.json`。

## 1. 不可违反的约束

1. 原始 RAW、TIFF、DNG、NLP 对照图只读。不得覆盖、移动、重压缩或提交进 Git。
2. Negative Lab Pro 只作对照，不是真值；不得以“更接近 NLP”替代色卡、中性区、剪裁和人工视觉验收。
3. 第一轮继续采用可解释的密度域、色彩管理、胶片/设备先验和统计标定。禁止先引入神经网络或声称小样本能训练通用模型。
4. 不得为通过 G200-1 写死单张或单卷参数。所有参数学习必须只使用标定集，并在按卷隔离的留出集上验收。
5. 不得用内嵌 JPEG 冒充不支持的 RAW 解码。
6. 不得修改公式语义却沿用旧 `schema_version`。
7. P0 质量门禁未通过前，禁止重新执行 Phase 5、生成发布 ZIP、写“已完成”或交付朋友试用。
8. 任一阶段失败时，停止后续阶段，保留可复现证据，报告实际失败原因。不得用降低阈值、扩大容差或删除测试掩盖失败。
9. 工作区可能含用户文件和已有修改。只改本规格列出的项目文件；不得清理或回滚不属于本任务的内容。
10. 所有新增大图输出写入 `outputs/` 或系统临时目录，并保持 Git 忽略。

## 2. 完成定义

只有同时满足以下条件，才可宣称整改完成：

- 用户指定的代表性 C-41 样片经过正确方向和默认裁切后，自动首版不再严重偏青蓝、反相错误或冲淡。
- 正常曝光留出样片的非蒙版输出，每通道低端和高端剪裁均低于 1%；异常片允许带明确警告导出，但不得计入正常样片通过率。
- 有灰卡/色卡或用户确认中性区域的样片，中性区 DeltaE2000 不高于 5；优秀目标不高于 3。
- 无中性物体的样片必须走胶片/设备先验或提示用户取点；不能无声使用全 1.0 通道参数并给出看似正式成品。
- TIFF16 与 JPEG 的像素编码、ICC、sidecar 声明一致，并能在至少两个支持 ICC 的查看器中得到一致外观。
- 裁切、旋转、ROI、参考点和除尘蒙版在 GUI、CLI、批处理和 sidecar 中使用同一坐标语义。
- 三件套导出是事务性的；任一工件失败时不留下半套结果。
- 真实样片回归、幂等冒烟、GBK 控制台、冻结版自检、安装/卸载验证全部通过。
- 用户已经选择项目许可证；或者构建被明确标记为仅内部验证、禁止分发。

## 3. 阶段和停止条件

严格按 P0-A 至 P3 顺序执行。每个阶段先写测试，再修实现，再记录证据。前一阶段未通过时不得开始下一阶段。

### P0-A：重建可信样片清单与验收数据

#### 任务

1. 修复 `scripts/build_fixture_manifest.py`：
   - `decode_status` 必须来自真实最小解码探针，不能只依赖未经验证的 24 字节启发式。
   - 对每个文件记录 `ok`、`unsupported`、`corrupt`、`missing_dependency` 等稳定枚举和简短原因。
   - 记录实际 RAW 解码参数、工作色域、白平衡策略、位深、尺寸、设备和输入 ICC 状态。
2. 修复 `fixtures/manifest.json` 与 `docs/blockers.md` 的矛盾。
3. 新增 `scripts/sample_intake.py`：
   - 只读扫描用户样片；计算 SHA-256；验证格式/位深/可解码性；不复制大图。
   - 支持用户继续提供 C-41、黑白、ECN-2、RAW、16 位 TIFF/DNG 样片。
4. 新增 `fixtures/annotations/`，只保存小型 JSON 标注：
   - `frame_roi`、`rotation_degrees`、可选 `neutral_rois`、`mask_rois`、`reference_file`、用户备注。
   - 坐标必须声明基于原始方向还是旋转后方向。
5. 按胶卷分组切分 calibration/holdout；同一场景及其不同导出不得跨集合。至少 20% 胶卷作为留出集。
6. 生成 `docs/sample-calibration.md`，列出当前覆盖和缺口。不得把“同一相机拍了很多张”写成多设备覆盖。

#### 当前必须查清的事实

- `20260506_SAMPLE-ID/1171_5219/翻拍/5219-1.NEF` 实测 rawpy 解码失败，但当前 manifest 标为 `ok`。
- 当前回归固定选择 G200，未覆盖 ECN-2、黑白、多设备或多光源。
- 用户已经表示可以提供样片。若缺少灰卡/色卡、黑白、可解码 ECN-2 或第二套设备样片，列出最小补充清单并暂停对应支持声明。

#### 新增测试

- `tests/unit/test_fixture_manifest.py`
- `tests/regression/test_decode_matrix.py`

测试至少断言：manifest 状态与实际探针一致；未知/不支持格式不会进入可解码回归；每个宣称支持的矩阵格至少有一张留出样片，否则测试以明确的 `xfail/skip reason` 标记且发布门禁失败。

#### P0-A 通过条件

- 清单、实际解码和 blocker 文档一致。
- 生成一份支持矩阵，逐项写 `verified`、`unsupported` 或 `missing-fixture`。
- 不再用总图片数掩盖类型/设备覆盖不足。

### P0-B：定义并实现闭环色彩管线

#### 先写决策文档

新增 `docs/color-pipeline.md`，至少明确：

1. RAW 解码时是否使用相机白平衡、自动白平衡、单位乘数或 daylight fallback。
2. rawpy 的 `use_camera_wb`、`use_auto_wb`、`user_wb/user_mul`、`output_color`、`gamma`、`no_auto_bright`、高光模式和位深的显式值。
3. 核心反相工作的线性工作色域及其 primaries/white point。
4. 带 ICC TIFF 如何转换；不带 ICC TIFF 允许用户选择什么假设；未知时如何阻止误处理。
5. `positive_linear`、显示预览、TIFF16、JPEG8 各自的传递函数和 ICC。
6. 色域外值、负值、高光压缩、量化和剪裁的顺序。

#### 实现要求

1. 新增 `src/negative_lab/color/management.py`，集中实现输入到线性工作空间、线性工作空间到输出空间和 ICC 装配。
2. 修改 `src/negative_lab/io/images.py`：
   - RAW 参数全部显式指定并写入 `LoadResult.meta`。
   - 读取 TIFF/JPEG 的 ICC、位深和传递函数状态；不再无条件把 TIFF 标为线性。
   - `save_tiff16()` 和 `save_jpeg8()` 接受明确的输出编码枚举，拒绝含义不明的 ndarray。
   - TIFF/JPEG 嵌入与像素值一致的 ICC，写后重新读取验证。
3. 修改 GUI、CLI 和批处理调用方，禁止把 `preview_srgb` 当作未说明语义的 TIFF 数据。
4. sidecar 记录输入配置、工作空间、输出配置、解码白平衡参数和 ICC 哈希。

#### 新增测试

- `tests/unit/test_color_management.py`
- `tests/unit/test_icc_roundtrip.py`
- `tests/regression/test_color_pipeline_real_files.py`

至少覆盖：带 ICC TIFF、不带 ICC TIFF、线性 TIFF、显示编码 TIFF、RAW、JPEG 预览；ICC 往返；灰阶中性保持；不支持 profile 的清晰失败。

#### P0-B 通过条件

- 所有输入和输出都能回答“这些 RGB 数字属于哪个色域、什么传递函数、什么白平衡策略”。
- TIFF/JPEG 写后检查存在 ICC，像素编码与 sidecar 一致。
- 不得保留 `input_profile: linear-normalized` 这种未经事实验证的硬编码。

### P0-C：修复自动颜色、裁切、方向与剪裁诊断

#### 实现要求

1. 修改 `src/negative_lab/pipeline/auto.py`：
   - 自动结果必须返回 frame ROI、base ROI、坐标空间、置信度和失败原因。
   - 缺少中性点时，按“设备/光源/胶片先验 -> 统计中性候选 -> 用户取点 -> 带警告允许导出”的顺序执行。
   - 统计中性候选必须排除片边、强饱和区、剪裁区和尘点；置信度不足时不得伪装成功。
   - 胶片/设备先验从标定集学习，版本化保存；留出集不得参与调参。
2. 修改 `src/negative_lab/pipeline/batch.py`、`src/negative_lab/cli.py`、`src/negative_lab/ui/main_window.py`：
   - 共用同一个 `apply_geometry()`；默认裁切到检测/确认的胶片画面。
   - 提供“保留片边”选项，默认关闭。
   - 旋转作为显式参数进入 sidecar 和输出，不依赖 GUI 临时状态。
3. 新增 `src/negative_lab/diagnostics/quality_gate.py`：
   - 统一生成输入/输出逐通道剪裁、NaN/Inf、色偏、ROI、置信度警告。
   - `>1%` 输出剪裁必须包含通道、方向和比例。
   - GUI、CLI、批处理、sidecar、报告共用同一结果，禁止各自靠字符串猜状态。
4. 任何正常曝光留出样片若严重偏色、反相方向错、默认裁切包含大片片边或剪裁超标，P0-C 失败。

#### 新增测试

- `tests/unit/test_quality_gate.py`
- `tests/unit/test_geometry.py`
- `tests/regression/test_reference_quality.py`

真实质量回归至少检查：

- `rotation_degrees` 与人工标注完全一致。
- 默认 frame crop 与人工标注 IoU 不低于 0.95；G200-1 输出不得保留左右编号和大面积黑边。
- 正常曝光留出样片逐通道 low/high clip 均小于 0.01。
- 有中性标注时 DeltaE2000 小于等于 5。
- 对齐后的参考比较只作为异常筛查，并记录亮度、彩度和结构差；不得用未对齐全图 DeltaE 得出通过结论。
- 固定输入和参数重复两次，参数与像素在定义容差内一致。

#### 人工视觉门禁

新增 `scripts/build_visual_contact_sheet.py` 和 `docs/visual-acceptance.md`：

- 每张并排显示原负片、本软件结果、可选 NLP/用户参考；使用相同方向和画面范围。
- 至少记录高光、肤色、中性灰、阴影、整体色偏、裁切和方向七项。
- G200-1 当前输出必须列为整改前失败基线。
- 自动生成联系表后暂停，请用户确认代表性结果。没有用户确认不得进入 P3 发布。

#### P0-C 通过条件

- 机器指标通过。
- 联系表不存在明显青蓝罩、冲淡、反相错误和大片默认片边。
- 用户确认代表性首版结果达到可继续试用的水平。

### P1：修复状态一致性、旁车和事务导出

#### GUI 状态

修改 `src/negative_lab/ui/main_window.py` 和 `src/negative_lab/ui/preview_view.py`：

- 建立统一图像状态对象，至少含 source id、rotation、frame/base ROI、reference points、dust mask、params、profile id。
- 加载新图时重置全部图像相关状态；检测失败时显式清空旧 ROI。
- 旋转时必须正确变换所有 ROI、点和蒙版，或清空后重检。二者选一并写测试。
- 导出前校验所有坐标在当前图像范围内。

#### sidecar

新增单一 `build_sidecar()` 和 schema v2 校验：

- 记录 source SHA-256、原始尺寸、方向、输出尺寸、frame/base ROI、坐标空间、几何变换、profile id、完整 warnings、quality gate、色彩配置和软件版本。
- GUI、CLI、批处理不得各写一套不同 JSON。
- 提供 v1 读取迁移；不得假装旧错误 ROI 可以恢复为真实坐标。

#### 事务导出

新增 `src/negative_lab/io/export_bundle.py`：

- 预先确定 TIFF/JPEG/sidecar 三个唯一目标名。
- 全部写入同目录 staging 文件并校验可重读。
- 全部成功后提交；任何失败清理本次工件，绝不留下半套输出。
- 不覆盖源文件；默认不覆盖已有输出。

#### 批处理

修改 `src/negative_lab/ui/batch_panel.py` 和 `src/negative_lab/pipeline/batch.py`：

- GUI 完成后必须实际写出 JSON/TXT 报告，再显示路径。
- 使用稳定 item id 或规范化完整路径关联表格，支持不同子目录同名文件。
- 报告也使用原子写。
- 失败、警告、取消和重试状态来自结构化枚举，不靠中文字符串包含关系。

#### 必测场景

- 新图 -> 旋转 -> 取点 -> 画蒙版 -> 再旋转 -> 导出。
- 图 A 有 ROI/点/蒙版，图 B 检测失败，确保图 A 状态不泄漏。
- TIFF 成功后 JPEG 失败、sidecar 失败、磁盘满、已有目标、取消。
- 两个不同目录包含同名 NEF 的批处理。
- GUI 批处理完成后报告文件真实存在且内容与表格一致。

### P2：性能、可重复验证和桌面体验

#### 内存与响应

1. 新增 `scripts/benchmark_memory.py`，记录 45 MP RAW 的峰值 RSS、耗时和输出尺寸。
2. 重构 `invert_chunked()`：不得同时长期保留输入、完整 positive、完整 preview、uint16 和完整 TIFF `BytesIO`。
3. 优先采用分块 TIFF 写入、memmap 或逐块输出；JPEG 可使用受控的单份 8 位缓冲。
4. 全分辨率导出移入 worker thread，显示进度；取消必须安全且不留工件。
5. 为最低支持内存写出明确预算。未实测前不得宣传“大图分块低内存”。

#### 配置与易用性

- 启用单张 GUI 配置加载、当前配置显示、清除和错误提示。
- 损坏配置不能静默消失。
- 实现文件拖放。
- 将 film type 的人类可读名称和内部枚举分离。

#### 可重复脚本

- `scripts/e2e_cli_smoke.py`、`scripts/gui_smoke.py`、`scripts/verify_package.py` 接受 `--work-dir`，默认使用唯一临时目录。
- 同一脚本连续运行两次必须都通过。
- 控制台状态使用 ASCII `OK/WARN/FAIL`，或实现无法编码字符的安全回退。
- GUI smoke 的文件计数只检查本次 manifest，不扫描固定目录历史文件。

#### 中文 Windows 验证

至少执行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
uv pip check --python .\.venv\Scripts\python.exe
.\.venv\Scripts\python.exe scripts\e2e_cli_smoke.py
.\.venv\Scripts\python.exe scripts\e2e_cli_smoke.py
.\.venv\Scripts\python.exe scripts\verify_package.py
```

另在默认 GBK 控制台执行验证脚本，不设置 `PYTHONUTF8=1` 逃避兼容问题。记录 Windows 版本、代码页和输出。

在真实 Windows 桌面而非只在 `QT_QPA_PLATFORM=offscreen` 下验证：

- 中文字体正常，无方框。
- 100%、150%、200% DPI 下按钮、参数、警告和表格不重叠或截断。
- G200-1 真实预览与导出外观一致。

### P3：Windows 安装发布，仅在前面全部通过后执行

#### 发布前决策门

1. 暂停并让用户选择项目许可证：GPL-3.0、MIT/Apache、闭源免费或其他明确方案。
2. 若用户明确接受“仅绿色便携版”，记录对原规格“可安装软件”的变更；否则必须制作真正安装包。
3. 默认推荐在现有 PyInstaller onedir 外层使用 Inno Setup，保留 Qt/LibRaw 动态库和许可证文件。

#### 安装包要求

新增 `packaging/windows/negative-lab.iss` 或等价 MSIX 配置，至少包含：

- 版本、发布者、安装目录、开始菜单、可选桌面快捷方式、卸载程序。
- Qt、rawpy/LibRaw、LittleCMS 等许可证和可替换 DLL。
- 不包含用户样片、outputs、测试缓存或完整个人路径。
- 升级安装不破坏用户 profiles；卸载明确处理程序文件和用户配置选择。

#### 干净机验收

在没有 Python、uv 和源码仓库的干净 Windows VM/机器上执行并留证：

1. 安装。
2. 启动中文 GUI。
3. 处理一张已授权分发到测试机的 RAW 和一张 16 位 TIFF。
4. 检查 TIFF/JPEG/sidecar/ICC/警告。
5. 关闭、重启、重复导出。
6. 升级安装。
7. 卸载并确认程序文件移除、用户数据处理符合选项。

生成安装包 SHA-256、SBOM、变更日志、支持矩阵和已知限制。任何一步未实测时必须写“未验证”，不得写“通过”。

## 4. 统一测试门禁

整改后至少保留以下分层：

| 层级 | 必须验证 | 失败后动作 |
|---|---|---|
| 单元 | 色彩转换、ICC、几何变换、quality gate、sidecar schema、事务导出 | 停止 |
| 合成 | 密度方向、单调性、边界、已知中性恢复 | 停止 |
| 真实回归 | 按类型/卷/设备参数化，裁切、方向、剪裁、中性 DeltaE、稳定性 | 停止 |
| 视觉 | 同画面联系表，高光/肤色/中性灰/阴影/整体色偏 | 请求用户确认 |
| E2E | CLI、GUI、批处理、失败注入、连续运行两次 | 停止 |
| 冻结版 | 自检、RAW/TIFF、ICC、中文 GUI | 停止 |
| 安装 | 干净机安装、处理、升级、卸载 | 停止发布 |

严禁以下“伪通过”：

- 只检查输出文件存在。
- 只检查均值在 `0.05-0.95`。
- 只检查没有 NaN/Inf。
- 使用未对齐全图 DeltaE 后以“指标不可信”为由忽略肉眼失败。
- 因样片缺失而 skip 后仍宣称对应格式/胶片已支持。
- 只运行 offscreen GUI 就宣称中文 Windows 界面通过。

## 5. 每阶段交付格式

每完成一个阶段，更新 `docs/remediation-progress.md`，使用以下固定格式：

```markdown
## P0-X

- 状态：PASS | FAIL | BLOCKED
- 修改文件：
- 实际执行命令：
- 测试结果：
- 真实样片结果：
- 视觉证据路径：
- 峰值内存/耗时：
- 未验证项：
- 下一门禁：
```

最终报告必须逐条回填 `质量审查报告.md` 中 F-1 至 M-4 的状态，附具体测试名、样片 ID 和产物路径。不得只写“已修复”。

## 6. 首次执行顺序

拿到本文件后立即按以下顺序行动：

1. 确认 cwd 为 `/src/film-darkroom`，记录 `git status --short`，保护现有用户改动。
2. 运行现有测试并保存基线，但明确现有绿灯不代表成像合格。
3. 完成 P0-A，修复 manifest 与真实解码矛盾。
4. 请求或整理用户样片元数据和人工 ROI；按卷切分 calibration/holdout。
5. 完成 P0-B 色彩管线和 ICC 测试。
6. 完成 P0-C 自动颜色、裁切、剪裁和真实质量回归。
7. 生成整改前后联系表，请用户做视觉确认。
8. 用户确认后执行 P1、P2。
9. 前述门禁全绿后再执行 P3，并在许可证决策处暂停等待用户选择。

这份顺序不可被“先重新打包看看”或“先补几个单元测试”打乱。当前真正的第一优先级是让真实负片输出成为一张方向、裁切、颜色和层次都可接受的照片。
