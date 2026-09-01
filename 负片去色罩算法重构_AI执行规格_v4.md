# 负片去色罩算法重构 AI 执行规格 v4

> 状态：P0 阻断整改规格，供下一位 AI 直接读取并执行  
> 审查基线：`f9269b2cba134ae0119635e0a52101876daaace1`  
> 审查日期：2026-09-01  
> 适用仓库：`/src/film-darkroom`  
> 核心结论：当前程序可以运行和导出，但去色罩生产管线、颜色标定、质量门禁和若干基础设施仍未真正完成，禁止发布。
> 最高优先级：去色罩/技术反相算法是 P0；除双 ROI、标定取样和结果对比所需的最小界面外，任何 UI 美化与扩展都必须排在算法门禁之后。

## 0. 给执行 AI 的强制指令

1. 先完整阅读本文件、`负片去色罩算法重构_AI执行规格_v3.md`、`docs/color-pipeline.md` 和本文点名的源文件，再改代码。
2. 不得删除、放宽、跳过失败测试，不得用更新 golden 图或只调 G200-1 固定参数来追认错误输出。
3. 每个 P0 阶段必须先新增能在当前基线上失败的测试，再实现，再运行该阶段测试。测试必须调用生产入口，不能只直接调用孤立 helper/backend。
4. 保留工作区内不属于本任务的用户改动。不得覆盖、移动或修改原始样片。
5. 临时输出必须使用 `tmp_path` 或 `TemporaryDirectory`；只有最终审查证据可写入 `outputs/quality/<run-id>/`。
6. `PASS` 只允许用于“实现已接入全部入口且验收命令通过”的项目。只有类、函数或单元测试存在，不算功能完成。
7. 不得把 Negative Lab Pro（NLP）描述为开源或声称掌握其内部公式。NLP 导出只能作为同帧用户偏好对照，不是绝对色彩真值。
8. 当前缺黑白样片、可解码 ECN-2、真实 16-bit TIFF、第二设备/光源。在这些样片到位且门禁通过前，发布状态必须保持 `BLOCKED`，不得伪造 fixture 或改掉发布门禁。
9. 阶段全部完成后，输出一份逐项证据报告：改动文件、失败测试红灯、修复后绿灯、真实指标、截图路径、未完成项和发布结论。
10. 执行顺序锁定：先完成第 3-7、11-12 节的算法、标定和真实样片门禁，再执行第 13 节 UI 重构。不得以竞品界面接近、截图好看或 RenderIntent 调色替代技术反相通过。

## 1. 当前基线的可复现实证

### 1.1 真实 G200-1 当前输出

同一帧：

- 输入：`20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-1.NEF`
- 手工标注：`fixtures/annotations/1298_G200__G200-1.json`
- NLP 对照：`20260709_SAMPLE-ID(1)/1298_G200/翻拍/NLP/G200-1.jpg`
- 当前 GUI 截图：`outputs/gui_smoke/gui_window.png`

当前 HEAD 重新运行 GUI 冒烟虽然退出 0，但截图仍严重偏青、冲淡、低色度，并保留大片底部片边/非画面区域。

| 指标 | 当前 `process_image()` | 同帧 NLP 对照 | 判定 |
|---|---:|---:|---|
| 线性亮度 p50 | 0.646558 | 0.405010 | 当前明显冲淡 |
| sRGB 饱和度 p50 | 0.163561 | 0.171717 | 中部接近，但不能代表高色度恢复 |
| sRGB 饱和度 p90 | 0.335651 | 0.666667 | 当前高色度端约少一半 |
| 线性相对彩度 p90 | 0.833491 | 1.503275 | 当前色度动态范围塌缩 |
| 输入 B 低端剪裁 | 5.0801% | 不适用 | 应失败，当前仅警告 |
| 输出 R/G/B 低端剪裁 | 3.6958% / 2.3574% / 3.6921% | 0.0472% / 0.0234% / 0.2799% | 当前输出细节丢失 |
| 质量状态 | `warn` | 不适用 | 不足以阻断错误成品 |

### 1.2 已实锤的生产路径失败

1. `src/negative_lab/pipeline/service.py:260` 固定调用旧 `invert_chunked()`；`src/negative_lab/pipeline/density.py` 不读取 `density_matrix`。
2. 对 `invert_chunked()` 改成强非单位矩阵，输出最大差为 `0.0`；直接调用 `density-affine` backend 时最大差为 `0.70148`。说明 backend 存在，但 GUI/CLI/批处理实际输出完全不使用它。
3. `RollCalibration` 没有从 GUI、CLI 或批处理加载并传入 `process_image(calibration=...)`。
4. `RenderIntent`/`apply_render()` 只被测试调用，没有接入 `process_image()` 或任一用户入口。
5. `QualityGate` 对 `calibration={level: None, confidence: 0.9}` 返回 `ok`，违反“彩色负片缺 verified 颜色标定必须 needs_calibration”。
6. `QualityGate._linear_to_lab()` 把线性 RGB 先加 sRGB TRC，再直接乘线性 RGB→XYZ 矩阵。输入线性灰 `0.5` 得到 `L*=88.70`；正确值约 `76.07`。
7. `valid_mask`、`neutral_ref_positive` 与 DeltaE API 没有从主服务传入，属于孤立接口。
8. 当前色度塌缩阈值 `0.02` 且只在已有 calibration level 时运行，抓不住已知的 G200-1 失败。
9. `backends.py:33-34` 计算了通道指数 `k`，但 `power_reciprocal()` 后续完全不使用；实测把 `red_ratio` 或 `blue_ratio` 改为 2.0，backend 输出最大变化均为 `0.0`。
10. `power_reciprocal()` 和 `density_affine()` 都忽略 `exposure`；实测 exposure 1→2 的最大变化为 `0.0`。单位矩阵 backend 与现有 `invert()` 在合法非默认参数下最大差为 `0.33999`，因此“单位阵等价旧模型”的声明不成立。
11. `density_affine()` 在应用 calibration 矩阵后又按矩阵行范数缩放，并把通道指数强制重置为 1。拟合得到的矩阵并不是运行时实际使用的完整变换，矩阵、通道斜率和 tone 参数存在隐藏重参数化。

### 1.3 新增回归与假绿

1. `src/negative_lab/calibration/annotations.py` 重复定义 `save_annotation`，第二个定义覆盖第一个；`scripts/write_p0a_annotations.py:63` 的单参数调用会 `TypeError`。
2. `load_all_verified()` 已不存在；其原逻辑位于提前 `return` 后，成为死代码。
3. `save_annotation(rel, ann)` 写盘前不调用 `Annotation.validate()`，无效数据可被写成 `verified=true`。
4. `export_triplet()` 为 TIFF、JPEG、sidecar 分别分配唯一名。在只预先存在 `s-positive.jpg` 时，实测得到：
   - `s-positive.tif`
   - `s-positive-1.jpg`
   - `s-positive.tif.negative.json`
5. `BatchJob(export_jpeg=False/export_tiff=False)` 仍无条件导出 TIFF+JPEG+sidecar，构造参数是无效开关。
6. `_inject_fail="commit"` 在移动任何文件之前失败，所谓“已提交工件回滚”测试没有覆盖部分提交。
7. `scripts/e2e_cli_smoke.py` 强制按 UTF-8 解码 Windows 子进程的 GBK 输出。两个 reader thread 均抛 `UnicodeDecodeError`，脚本仍打印 `E2E OK x2` 并退出 0。
8. 当前相关单测 `44 passed`，但上述行为仍全部可复现，证明现有测试覆盖的是孤立实现或错误断言，不是生产契约。

### 1.4 竞品截图新增证据（用户提供，2026-09-01）

参考截图：

- `/home/lab/imports Files/2995032148/nt_qq/nt_data/Pic/2026-09/Ori/915ff1883ebe793cc71b8946c80449ef.png`
- `/home/lab/imports Files/2995032148/nt_qq/nt_data/Pic/2026-09/Ori/f2602a873a187016266f20cb19875eaa.png`
- `/home/lab/imports Files/2995032148/nt_qq/nt_data/Pic/2026-09/Ori/bce29955fde203da92c41046dc651cf5.png`
- `/home/lab/imports Files/2995032148/nt_qq/nt_data/Pic/2026-09/Ori/5dbba569d91865618ca696ad04011b06.png`

这些文件都是 3600x2338、带 ICC 的页面截图。它们只作为工作流、界面和用户偏好参照，
不能作为线性像素真值、色卡真值或竞品内部算法证据；不得跨场景直接用截图直方图制定颜色阈值，
也不得复制竞品品牌、图标、文案或视觉资产。

截图能确认的产品级事实：

1. 竞品把卷作为一级对象：左侧胶片库展示同卷缩略图、已转换/未转换状态和当前帧，说明卷级共享分析是主工作流，不是隐藏在单张参数里的可选功能。
2. 它明确分成“阶段/转换/编辑”：先处理 RAW 和胶片几何，再完成技术转换，最后做曝光、对比度、高光、阴影、白色、黑色、曲线和色彩编辑。这与本规格要求的 TechnicalPositive/RenderIntent 分离一致。
3. 它使用两个不同矩形：蓝色是最终输出裁切，绿色是胶片画面/分析区域。输出可以有意保留齿孔和片边，但这些区域不参与画面统计、自动色彩和质量门禁。
4. 两张夜景结果都具备可信黑位、受控高光、接近中性的建筑白色、可分辨的红/青/黄招牌和稳定的卷内风格；没有当前软件的全局青罩、冲淡和高色度端塌缩。
5. 转换结果保留胶片边框是明确审美选择，而不是检测失败。因此“保留 rebate”与“让非画面像素污染算法”必须在数据模型上彻底分开。

## 2. 唯一允许的目标架构

生产处理顺序必须是：

```text
load + input color management
  -> linear sRGB D65 working RGB
  -> geometry / output-crop + image-area + base-sample / flat-field / dust preparation
  -> roll calibration selection and validation
  -> technical inversion through selected backend
  -> technical quality gate
  -> RenderIntent (optional style layer)
  -> output quality diagnostics
  -> transactional export + sidecar
```

必须满足以下边界：

- GUI、CLI、批处理只能调用 `process_image()`，不得在入口中重新反相或重新构造 QualityGate。
- backend 只负责技术反相；风格参数不得参与标定拟合或技术质量判定。
- `ProcessResult` 必须同时保留：
  - `technical_positive_linear`
  - `rendered_positive_linear`
  - `preview_srgb`
  - 实际 backend 名称
  - 实际 calibration ID/source/provenance
  - 实际 RenderIntent
- 质量门禁先检查技术正片，再单独记录渲染后剪裁；不得拿风格渲染掩盖技术失败。
- 导出默认写 `rendered_positive_linear`；中性 RenderIntent 时它必须与技术正片逐像素等价。
- sidecar schema 必须升为 v3，明确记录技术参数、风格参数、backend、calibration、mask、技术门禁和渲染后诊断。不得继续把新语义伪装成 schema v2。
- ROI 必须拆成三个独立语义：`output_crop_roi` 决定最终构图，`image_area_roi` 决定算法/质量统计，`base_roi` 只用于片基/色罩估计。不得再用一个 `frame_roi` 同时承担裁切和统计。
- `output_crop_roi` 可以包含用户有意保留的齿孔/片边；`image_area_roi` 必须排除它们。QualityGate 的有效 mask 来自 image area，而不是整个输出 crop。
- `keep_rebate: bool` 只能作为旧兼容入口，最终必须迁移为明确的 output crop；一个布尔值不足以表达竞品截图中的双 ROI 工作流。

## 3. P0-A：先建立会失败的生产契约测试

在修改实现前新增以下测试，并保存第一次失败摘要。测试文件名可调整，但测试语义不可缩减。

### 3.1 backend 接线测试

新增至少四项：

1. 对具有已知通道串扰的合成负片，调用 `invert_chunked()`，非单位 `density_matrix` 相对单位阵的最大输出差必须 `> 1e-3`。
2. 对同一输入调用 `process_image(force_params=...)`，非单位矩阵必须影响 `technical_positive_linear`。
3. 单位矩阵的 `density-affine` 与兼容 `power-reciprocal` 基线必须在明确容差内等价；测试必须覆盖非整块尺寸和最后一个 chunk。
4. 参数敏感性表必须覆盖 red/blue ratio、矩阵、Dmax/technical endpoints 和所有保留参数；当前 `red_ratio`、`blue_ratio`、`exposure` 在 backend 中变化为 0 的事实必须先形成红灯测试。若 exposure 被迁移到 RenderIntent，则测试应证明它只改变 rendered、不改变 technical。

禁止只测试 `get_backend("density-affine")`。

### 3.2 全入口接线测试

为 GUI service adapter、CLI 和 BatchJob 分别验证：

- 同一个 calibration 文件确实到达 `process_image()`；
- 同一个 RenderIntent 确实影响渲染结果但不改变技术正片；
- sidecar 记录的是实际生效对象；
- 无 calibration 的 C-41/ECN-2 结果不可能是 `ok`。

### 3.3 数学与门禁测试

至少新增：

- `linear_srgb_to_lab([0.5,0.5,0.5])` 的 `L*` 约为 `76.069`；误差不超过 `0.02`。
- CIEDE2000 使用项目已有 Sharma 标准向量继续验证。
- 输入任一通道剪裁 `0.11%` 至少 warn；`1.01%` 必须 fail。
- 彩色负片无 verified 颜色标定时，无论 confidence 是 0.1 还是 0.99，都必须 `needs_calibration`；如果另有 failure，`fail` 优先。
- `valid_mask` 形状错误、空 mask、全 False 必须显式报错，不得产生 NaN 比例。
- 彩色输出已知低色度失败样本必须被色度门禁抓住。

### 3.4 annotation、导出和冒烟回归测试

至少新增：

- annotation 只有一个公开保存 API；旧脚本和 GUI 都使用它。
- 保存必经 `validate()`；越界/反向/空 ROI、非法旋转、`verified=true` 但缺验证来源必须拒绝。
- `load_all_verified()` 可调用且不会执行死代码。
- 三件套在任意单个目标名已存在时必须共享同一新 suffix。
- 提交第一件、第二件之后分别注入失败，最终目录零新增且 staging 清空。
- BatchJob 四种 TIFF/JPEG 开关组合的实际输出集合与声明一致；sidecar 始终存在。
- CLI 冒烟子进程输出解码失败、reader thread 异常、Traceback 或 `stdout is None` 都必须使测试失败。
- 双 ROI 测试必须证明：改变 output crop 只改变构图；在 image area 不变时，技术参数和画面内 QualityGate 指标不变。把齿孔加入 output crop 不得造成剪裁 fail 或改变颜色标定。

## 4. P0-B：重建去色罩技术反相并接入生产

### 4.1 先定义唯一的技术反相数学契约

不得继续从现有函数拼装语义。先在 `docs/algorithm-notes.md` 写出并测试一个无歧义的前向/逆向模型。推荐使用正密度表述：

```text
b_c        = 未曝光片基的线性透射率
d_c        = -log10(max(t_c, eps) / max(b_c, eps))
u          = A d + a
v_c        = f_c(u_c)                # 单调胶片通道响应；线性模型时 f_c(x)=s_c*x
technical  = G(v, technical_endpoints)
```

若保留现有负向 `z=log10(t/b)`，必须证明与上式逐项等价，并在 API、sidecar 和文档中只暴露一种符号约定。

强制语义：

- `b/Dmin` 只来自未曝光片基或 verified base ROI，不能被场景 p99、齿孔或最亮物体改写。
- `A` 只表达密度域通道混合/解混。拟合矩阵必须是运行时实际矩阵；禁止 backend 再做未记录的行归一化。
- 若为可辨识性需要把 `A` 分解成单位对角矩阵和通道尺度，必须把分解后的 `A` 与 `s_c/f_c` 全部序列化，并证明重组等于拟合变换。
- `f_c` 必须单调。完整 3x3 `A` 和非线性曲线不能从单一片基或一条中性密度射线中估出。
- `technical_endpoints` 只做技术范围映射；曝光、对比度、高光、阴影、白色、黑色、饱和度和审美曲线属于 RenderIntent。
- 每个保留参数必须有参数敏感性测试：在受控输入上改变参数，目标输出必须按预期变化；无效果或语义重叠的参数删除或迁移，禁止保留假旋钮。
- 建立前向合成模型并做 round-trip：已知 scene/color target → 模拟染料密度、橙罩、串扰和相机采集 → 逆算法。仅用三个独立灰通道不能验证颜色恢复。

### 4.2 候选算法同源 bake-off

必须实现并用同一生产接口比较，不能先凭偏好指定赢家：

| 路线 | 技术模型 | 用途 |
|---|---|---|
| A：diagonal-density | `A=I`，每通道密度斜率 | 可解释基线，证明片基归一化和方向正确 |
| B：density-affine | verified 3x3 `A` + 明确通道尺度 | 处理染料/光源/传感器造成的一阶串扰 |
| C：density-affine-curves | B + 每通道受约束单调 spline/curve | 处理不同染料层非线性特性曲线，当前优先候选 |

路线 C 的曲线节点数必须受限并正则化，训练和留出严格分离；不得用高自由度曲线逐张拟合场景内容。

神经网络、任意 3D LUT 或从竞品截图反推 LUT 不属于当前 P0 候选。只有积累足够的“同一 RAW + verified 参考/色卡 + 跨卷留出”数据后，才可作为后续路线 D，并必须与路线 C 在未见卷上比较。

所有候选共享完全相同的 RAW 解码、working space、双 ROI、Dmin、RenderIntent=neutral 和导出编码。每条路径必须输出：实际参数、运行时间、剪裁、非有限值、单调性、色卡/中性指标、卷内一致性和同裁切联系表。

对竞品的有效比较必须使用同一源 RAW 与竞品“转换后、编辑参数归零”的导出。只有页面截图时，只记录视觉目标，不计算算法排名。

### 4.3 统一 backend 选择

1. 在 `InvertParams` 或独立、可序列化的技术配置中增加经过验证的 `backend_name`，默认选择必须明确。
2. bake-off 完成前，开发和测试必须显式指定 A/B/C 路线，不得提前把任一路线写成“彩色负片默认”。生产默认只能由第 4.2/4.5 节留出集证据确定；`power-reciprocal` 仅作为明确选择的旧兼容对照。
3. `invert()` 与 `invert_chunked()` 必须通过同一个 backend 注册表执行，不得保留一套绕过 backend 的旧公式。
4. chunked 版本只能负责分块和诊断聚合；每块必须调用与非分块相同的技术反相核心。
5. 未知 backend、非 3x3 矩阵、非有限矩阵、病态或超出约束的矩阵必须在处理前失败，并给出可读错误。

### 4.4 技术反相与风格分离

技术阶段允许包含：片基归一化、密度变换、解混矩阵、通道响应斜率、Dmax/offset 的技术映射。技术阶段不得包含用户风格饱和度或隐藏 tone profile。

现有 `black/gamma/exposure/highlight_*` 语义必须做一次明确裁决：

- 若属于风格，迁移到 RenderIntent，并为旧 sidecar 提供一次性兼容映射；
- 若保留在技术模型，必须在文档中给出物理/算法定义和可辨识性依据；
- 禁止在 backend 和 RenderIntent 中重复应用。

### 4.5 算法验收

- `invert()` 与 `invert_chunked()` 在相同参数下 `allclose`，覆盖奇数尺寸和小于 chunk 的图。
- `process_image()` 对非单位矩阵的效果不为 0。
- red/blue ratio 及所有保留的技术参数通过敏感性测试；现有 exposure 必须迁移到 RenderIntent 或删除旧字段，并证明它只按新语义生效。不得再出现公开合法参数被静默忽略。
- 单位矩阵 backend 与定义的 diagonal-density 基线在相同参数语义下等价，不允许 `0.33999` 量级的隐藏差异。
- 合成串扰留出集在不牺牲亮度单调性的前提下，路线 B/C 的颜色恢复指标优于路线 A。
- 有色卡的留出集目标：neutral ΔE2000 mean `<=3`，全部色块 mean `<=5`、p95 `<=10`；若数据不足或达不到，报告真实值并保持 experimental，不得放宽阈值追认。
- 无色卡但有同源竞品零编辑导出的样片，只在配准 ROI 上比较中性色偏、色相角、彩度比、亮度分位和剪裁；不能把整图 DeltaE 当真值。
- 同卷使用同一 calibration 时，各帧技术转换不得重新拟合颜色矩阵/曲线；帧间差异只能来自实际影像和帧级技术 endpoints。
- sidecar 中实际 backend、矩阵和公式版本与运行时完全一致。

## 5. P0-C：RollCalibration 全入口闭环

### 5.1 API 与选择规则

1. `process_image()` 接受强类型 `RollCalibration | None`，不再使用无类型 `calibration=None`。
2. GUI 提供选择/清除卷级标定的实际控件，并显示 source、roll、film type、device/light、verified 状态。
3. CLI 增加 `--calibration <path>`；批处理允许显式 calibration 或按 roll 映射，不能静默猜错卷。
4. calibration 的 film type、roll、device/light 与当前输入冲突时必须 fail 或要求用户确认，不能仅 warning 后继续。
5. base-only calibration 不得升级成颜色标定；只有可辨识且 verified 的 user-points/chart calibration 才可让彩色负片越过 `needs_calibration`。
6. `density_matrix`、通道斜率和白平衡必须从同一个版本化 calibration 原子采用，禁止只取其中几个字段而隐藏其余 provenance。

### 5.2 可辨识性

- 纯片基只能估 Dmin。
- 单组中性灰/中性梯度在密度空间只张成一维，不能辨识完整 3x3 串扰矩阵。
- 完整矩阵只能来自有足够秩的已知目标（色卡/监督样本）或明确的多颜色人工取点。
- 拟合必须报告样本数、矩阵秩、条件数/正则化、残差、训练/留出划分和回退原因。

### 5.3 验收

- GUI、CLI、批处理对同一图、同一 calibration、同一 RenderIntent 的技术正片哈希一致。
- 去掉 calibration 后三入口都变成 `needs_calibration`。
- calibration ID/source/hash/provenance 写入 sidecar v3。
- 无法辨识时回退单位阵并保持 `needs_calibration`，不得标为已校准。

## 6. P0-D：RenderIntent 接入且不污染技术基线

1. `process_image(..., render_intent=RenderIntent())` 统一生成技术正片和渲染正片。
2. QualityGate 的颜色正确性、DeltaE 和技术剪裁只读取技术正片。
3. 渲染后另记剪裁/非有限值；渲染后出现 NaN/Inf 必须阻断导出。
4. GUI 的 grade/contrast/lift/saturation 控件必须绑定 RenderIntent；不得继续直接改写 `InvertParams` 中的技术字段。
5. CLI 支持中性默认和显式 render 参数文件；批处理可复用同一 intent。
6. CLI `--params` 分支不得先跑一次 service、再手工替换 `pr.result` 和重建 QualityGate。它必须把参数作为 `force_params` 一次传入 service。
7. sidecar 同时记录 `technical_parameters` 和 `render_intent`；不得把两者混在一个 parameters 字段。

验收：中性 intent 逐像素恒等；改变 intent 只改变 rendered，不改变 technical、calibration 或 technical gate。

## 7. P0-E：重写 QualityGate 的真实语义

### 7.1 状态优先级

固定优先级：

```text
fail > needs_regions > needs_calibration > warn > ok
```

- C-41/ECN-2 彩色负片缺 verified 颜色标定时，无条件 `needs_calibration`，与自动估计 confidence 无关。
- `film_type=unknown` 不得被当成可正式通过的彩色成品；要求用户选择类型，或保守返回 needs_calibration。
- B&W 可跳过颜色标定，但仍执行剪裁、非有限值和动态范围门禁。

### 7.2 剪裁门禁

在有效画面 mask 内逐通道计算：

- 输入或技术正片剪裁 `>0.1%`：至少 warn；
- 输入或技术正片剪裁 `>1.0%`：fail，禁止正式导出；
- 交互预览可继续显示，但必须明确标红且导出按钮受阻；
- 渲染后剪裁单独报告，不得覆盖技术指标。

阈值必须在 sidecar 中记录，不能只写模块常量。

### 7.3 mask 接线

`process_image()` 必须构造并传入最终工作图坐标的 `valid_mask`：

- 默认是 `image_area_roi` 映射到最终 output crop 后的全部有效像素；
- 用户排除区、片边、齿孔、无效几何区和明确 mask 置 False；
- dust mask 是否排除必须在 sidecar 记录；
- mask 必须与技术正片 HxW 一致且至少含一个有效像素。

### 7.4 DeltaE 与比色学

1. 删除 `quality_gate.py` 中重复且错误的 Lab/CIEDE2000 实现。
2. 在 `diagnostics/colorimetry.py` 保留唯一实现，并增加 `linear_srgb_to_lab()`。
3. 线性 sRGB→XYZ 矩阵直接作用在线性值上；Lab 低亮度分支使用标准 `epsilon=216/24389`、`kappa=24389/27` 形式。
4. 若复用 `srgb_to_lab()`，只能先对线性值做 `linear_to_srgb()`，再由该函数正确解码；必须用已知 L* 测试锁定，避免再次错接语义。
5. DeltaE 只对有已知目标的色卡/verified 中性 ROI 计算。没有参考时必须是 `null`，且文案不得暗示通过。
6. 不得用未配准的全图 NLP DeltaE 充当颜色真值。

### 7.5 色度塌缩门禁

不再使用无依据的固定中位彩度 `<0.02`。至少报告：

- sRGB saturation p10/p50/p90；
- 线性相对彩度 p50/p90；
- 有同帧参考时的 p90 比率；
- 色卡/已知颜色 ROI 的彩度恢复误差。

在同帧参考的 holdout 上，技术正片的 saturation p90 至少达到参考的 75%，且不得通过全局饱和度旋钮作弊；最终阈值由多卷 calibration/holdout 数据确定并写入报告。

## 8. P0-F：手工区域与 annotation 数据完整性

用户裁决是“手工选择画面区域和片基区域为主”。执行如下：

1. `annotations.py` 只保留一个公共 `save_annotation(rel, ann)`；所有调用方统一迁移。
2. 保存前无条件执行 `ann.validate()`，再使用临时文件 + 原子替换。
3. 恢复可测试的 `load_all_verified()`，清除提前 return 后的死代码。
4. agent 自动检测结果只能保存为 draft；只有用户在 GUI 明确确认后才能 `verified=true`。
5. 当前 `verified_by="agent-inspected"` 的区域不得自动视为用户签字。逐项重新确认，未确认则降级 draft。
6. annotation schema 升级并明确区分：`output_crop_roi`、`image_area_roi`、`base_roi`、neutral/color reference ROI、exclude mask。现有语义含混的 `frame_roi` 只能迁移为 draft，必须由用户重新确认其用途。
7. GUI 使用两种稳定且不混淆的框：输出裁切框和画面分析框，并分别显示名称/颜色；保存前显示 SOURCE 坐标和旋转后的预览。
8. `image_area_roi` 必须排除片边、齿孔和明显非画面区域；`output_crop_roi` 是否包含这些区域由用户构图决定。
9. 提供“仅画面”和“保留胶片边框”两个快捷模式，但两者最终都落成显式 ROI，不得只存一个 `keep_rebate` 布尔值。
10. annotation 版本和源图 SHA-256 写入文件；源图变化时 annotation 失效。

验收：G200-1 当前截图中的底部暗带只有在用户明确把它纳入 output crop 时才可导出，且永不进入 image-area 统计；联系表同时显示原图、双 ROI 叠加、仅画面输出和保留边框输出。

## 9. P0-G：事务导出与批处理开关修复

### 9.1 原子分配统一 basename

一次事务先选择一个所有目标均未占用的 basename，例如：

```text
s-positive-1.tif
s-positive-1.jpg
s-positive-1.negative.json
```

suffix 只能分配一次；不得分别对三个文件调用 `_unique()`。候选检查必须覆盖本次启用的全部输出文件。

### 9.2 格式开关

- `export_tiff=True, export_jpeg=True`：TIFF + JPEG + sidecar。
- `True, False`：TIFF + sidecar。
- `False, True`：JPEG + sidecar。
- `False, False`：构造或运行前明确拒绝，不能产生只有 sidecar 的“成功任务”。
- GUI 和 CLI 对外选项必须与 BatchJob 行为一致。

### 9.3 真正的部分提交回滚

增加 `commit_after_1`、`commit_after_2` 等失败注入点。失败后：

- 删除本事务已经移动的全部文件；
- 清空 staging；
- 保留事务前已有文件且哈希不变；
- 若回滚删除失败，错误必须包含残留路径，不能静默吞掉。

## 10. P0-H：修复 Windows CLI 假绿

1. 子进程调用使用 `text=False` 捕获 bytes，或显式给子进程设置 `PYTHONUTF8=1` 与 `PYTHONIOENCODING=utf-8` 后严格解码。
2. 不得依赖父控制台当前 code page。
3. 捕获并验证 stdout/stderr：任何 `Traceback`、`UnicodeDecodeError`、reader thread 异常、`None` 输出都必须失败。
4. CLI 的机器可解析状态建议写 JSON/sidecar；人类控制台文案可本地化，但 smoke 不应靠乱码字符串判断成功。
5. 连续两个全新临时目录运行；每次独立断言返回码、三件套/选定格式、schema、ICC、质量状态和源文件哈希不变。
6. GUI smoke 除了控件/导出断言，还必须对截图做非空像素检查，并对真实输出调用质量指标；已知偏青冲淡图不能打印 `GUI SMOKE OK`。

## 11. P0-I：色彩管理和 benchmark 纠偏

### 11.1 保留已修正的 RAW 语义

`color/management.py` 当前已把 rawpy `ColorSpace.sRGB + gamma=1` 说明为线性 sRGB D65。保持这条自洽路径，并同步修正：

- `io/images.py` 仍写“线性相机 RGB/LINEAR/as-shot”的旧警告；
- `docs/color-pipeline.md` 中残留的旧 `output_color=6 (LINEAR)` 和 `camera-linear-normalized`；
- 任何 sidecar/报告中的过时 working-space 名称。

### 11.2 TIFF ICC

非 sRGB ICC TIFF 不能只去 TRC 而忽略 primaries。使用 LittleCMS/Pillow ImageCms 或等价的受测色彩管理，把源 ICC 真正变换到线性 sRGB D65；无法变换时明确拒绝。Adobe RGB 测试必须使用有已知色块的文件验证基色变换，不得只做 gamma 往返。

### 11.3 重写 bake-off

`scripts/benchmark_color_pipeline.py` 必须满足：

1. 每条候选路径都记录并实际应用声称的相机/工作空间矩阵。
2. 所有路径先进入明确工作空间，再执行同一个技术反相 backend；不能比较反相前数组。
3. 不得在反相前用 `[0,1]` 剪裁掩盖越界；越界必须进入诊断。
4. calibration 与 holdout 按卷/设备隔离，禁止同帧调参又评估。
5. 输出同裁切联系表、色卡/中性 ROI DeltaE、剪裁、亮度和彩度指标、环境版本、Git commit 和源文件哈希。
6. A/B/C 路径没有实际差异时测试必须失败，防止“名字不同、计算相同”。

## 12. P0-J：真实样片验收和用户签字

### 12.1 数据集要求

用户可以继续提供样片。先新增 `docs/calibration-protocol.md`，把本节的取样、
哈希、分组、标定和留出规则写成可重复流程；接收样片后按该流程建立：

- C-41 彩色至少两卷；
- 黑白至少一卷；
- 可解码 ECN-2 至少一卷；
- 真实 16-bit TIFF 输入；
- 第二设备或第二光源；
- 每卷至少包含片基、正常曝光、高反差、低反差、肤色/中性色/高饱和色场景；最好有色卡帧。

原图只读，记录 SHA-256。calibration/holdout 按卷或拍摄帧严格隔离，不得从 holdout 学参数。

### 12.2 G200-1 最低回归线

同一手工裁切下：

- 输入和技术正片逐通道剪裁都 `<1%`；
- 相对 NLP 的亮度 p50 绝对差 `<=0.12`，或由用户明确接受不同调性；
- saturation p90 至少为 NLP 同帧对照的 75%；
- 不出现整体青蓝罩、冲淡或非预期的非画面区域；用户明确选择的胶片边框可以保留，但不得参与技术统计；
- 技术改进必须来自 calibration/backend，不得通过 RenderIntent saturation 提升来冒充；
- 生成 `before / current technical / current rendered / NLP reference` 同裁切联系表。

### 12.3 发布签字

最终联系表必须交给用户逐卷选择：`接受 / 需调整 / 拒绝`。没有用户明确“接受”，`docs/visual-acceptance.md` 和发布状态都保持未通过。

## 13. P1：按竞品证据重构桌面工作流

这不是复刻竞品皮肤，而是补齐胶片用户需要的操作模型。整体方向为安静、克制、
高信息密度的数字暗房，不使用默认 Qt 外观、营销式卡片或装饰性渐变。

本节只有在第 4.5、5.3、7 和 12 节门禁通过后才可全面实施。算法阶段只允许实现双 ROI、标定取样、原片/技术正片对比和指标查看所需的最小界面。

### 13.1 信息架构

- 左侧“胶片库”：按卷组织缩略图，明确 `未分析 / 需区域 / 需标定 / 已转换 / 失败`，支持键盘快速切换相邻帧。
- 中央画布：原始负片/技术正片/渲染结果三态切换；双 ROI、参考点和 mask 作为可开关叠层，100% 缩放下可检查颗粒与边缘。
- 右侧检查器：固定为“阶段 / 转换 / 编辑”三个页签。阶段管 RAW、旋转和 ROI；转换管 film/calibration/backend；编辑只管 RenderIntent。
- 顶部工具栏：旋转、翻转、裁切、双 ROI、撤销/重做使用标准图标按钮并提供 tooltip，不用大段文字按钮。
- 直方图同时显示 RGB 和亮度，并明确当前统计对象是 image area、技术正片还是渲染结果。

### 13.2 编辑交互

- 曝光、对比度、高光、阴影、白色、黑色、饱和度使用滑杆 + 可编辑数值 + 单项重置。
- 曲线支持 RGB/单通道、节点添加删除、复位，并稳定保存到 RenderIntent/sidecar。
- 参数拖动使用降采样预览和缓存，控件不得因全分辨率同步计算卡死；松手后再调度全分辨率结果。
- 切帧时保留卷级 calibration，但帧级 RenderIntent、ROI 和状态必须正确隔离。
- 中文界面必须使用可用的 CJK 字体回退；任何方框字、乱码、文字截断或控件重叠都视为 GUI smoke 失败。

### 13.3 UI 验收任务

自动化和人工联合完成以下流程：导入一卷 → 标记 output crop 和 image area → 选择/建立卷级 calibration → 批量技术转换 → 原片/技术/渲染对比 → 调整一帧编辑参数 → 导出并重开验证状态。

保存至少两个桌面尺寸的完整截图和关键局部截图；验证缩略图状态、双 ROI、三阶段页签、直方图、曲线、数值控件、中文字体、键盘切帧和高 DPI。界面通过不代表颜色算法通过，两项必须分别签字。

## 14. 文档与状态修正

1. 立即把 `docs/remediation-progress.md` 中 P0-E/F/G/H/I 的虚假 `PASS` 改为与证据一致的 `PARTIAL` 或 `FAIL`。
2. 每个阶段只有在本规格验收通过后才能恢复为 PASS，并附命令、退出码、测试数、报告路径和 commit。
3. 更新 README、quickstart、color-pipeline、support-matrix：明确 experimental、支持范围、需要手工 ROI/颜色标定、缺样片阻断项。
4. 文档不得把“GUI smoke 退出 0”写成颜色质量通过。
5. 文档不得把“无参考 DeltaE=None”或“高 confidence 自动估计”写成已校准。
6. 新增 `docs/calibration-protocol.md`，记录样片目录约定、只读/哈希规则、annotation 用户确认、calibration/holdout 隔离和参考图用途边界。

## 15. 必须执行的验收命令

在仓库根目录使用项目虚拟环境：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_color_management.py
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_colorimetry.py tests\unit\test_quality_gate.py tests\unit\test_render_gate.py
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_service.py tests\synthetic\test_backends.py
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_manual_regions.py tests\unit\test_export_transaction.py tests\unit\test_batch.py
.\.venv\Scripts\python.exe -m pytest -q tests\synthetic
.\.venv\Scripts\python.exe -m pytest -q tests\regression
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
uv pip check --python .\.venv\Scripts\python.exe
.\.venv\Scripts\python.exe scripts\e2e_cli_smoke.py
.\.venv\Scripts\python.exe scripts\gui_smoke.py
.\.venv\Scripts\python.exe scripts\benchmark_color_pipeline.py --manifest fixtures\manifest.json
.\.venv\Scripts\python.exe scripts\verify_package.py
```

额外必须保存：

- `outputs/quality/<run-id>/metrics.json`
- `outputs/quality/<run-id>/contact-sheet.jpg`
- `outputs/quality/<run-id>/environment.json`
- `outputs/quality/<run-id>/commands.txt`
- `outputs/quality/<run-id>/user-acceptance.md`

任一 P0 命令失败、reader thread 报错、真实门禁 fail、fixture 缺失或用户未签字，发布结论必须是 `BLOCKED`。

## 16. 完成定义

只有以下项目同时成立，才可称 v4 完成：

- 非单位 `density_matrix` 已影响 `process_image()`、GUI、CLI、批处理的技术正片。
- calibration 与 RenderIntent 已接入三入口，sidecar 可追溯且 schema 已升级。
- Lab/DeltaE 数学通过标准向量；没有重复实现。
- 无颜色标定的彩色负片始终 `needs_calibration`；剪裁和低色度失败能被阻断。
- annotation API 唯一、验证后原子保存，手工区域经过用户确认。
- output crop、image area、base ROI 已分离；有意保留胶片边框时不会污染颜色、剪裁或质量统计。
- 事务输出共享 basename，格式开关有效，部分提交能真实回滚。
- Windows CLI smoke 不再出现解码线程异常或假绿。
- 桌面工作流完成“胶片库 + 三阶段 + 双 ROI + 原片/技术/渲染对比”，中文字体与高 DPI 验收通过。
- 全量测试、真实 regression、benchmark、CLI/GUI smoke、package verify 均通过。
- 所需样片类别齐全，calibration/holdout 无泄漏。
- 用户对逐卷联系表明确签字接受。

只要缺少其中任何一项，最终报告必须写 `未完成 / BLOCKED`，并列出缺口，不得写“核心 PASS”或“可发布”。
