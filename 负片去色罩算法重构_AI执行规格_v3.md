# negative-lab 去色罩算法重构 AI 执行规格 v3

> 文档日期：2026-09-01  
> 用途：直接交给 AI 编程代理执行。  
> 优先级：本文档取代旧规格中关于 RAW 工作空间、去色罩、自动中性和色彩验收的互相冲突部分。其他输入、几何、事务导出、批处理、样片覆盖和发布要求继续以 `负片软件整改_AI执行规格_v2.md` 为准。

## 0. 执行结论

当前代码不是“只需微调参数”，而是颜色数学和验收方法同时失效。当前 C-41/ECN-2 RAW 自动转换必须标记为 **experimental / P0 fail**，禁止重新打包或声称可交付。

现有 `G200-1` 结果已经直观失败：画面过淡、中调被抬高、色度塌缩、底部仍有非画面区域。本软件结果与 NLP 对照的缩略统计也支持这一点：

| 指标 | 当前软件 | NLP 对照 | 解读 |
|---|---:|---:|---|
| 线性亮度 p01 / p50 / p99 | 0.087 / 0.735 / 0.894 | 0.0006 / 0.403 / 0.942 | 当前黑位浮起、中调过亮、动态范围明显收窄 |
| HSV 饱和度中位 / p90 | 0.065 / 0.131 | 0.179 / 0.667 | 当前色度分离明显塌缩 |

NLP 不是真值，上述数字只用于证明“当前输出不是正常的另一种风格”，不用于训练或像素级追随 NLP。

## 1. 已确认的根因

### F-ALG-1（致命）ACES 数值被冒充为 camera-linear，且输出缺少色域变换

- 位置：`src/negative_lab/color/management.py:115-172`、`src/negative_lab/io/images.py:214-217`、`src/negative_lab/io/images.py:246-247`。
- 当前环境 `rawpy 0.27.1` 中 `rawpy.ColorSpace.ACES.value == 6`，但代码把 `output_color=6` 写成 `LINEAR` 和 `camera-linear-normalized`。
- 密度模型输出仍是 ACES 基色语义，但导出仅施加 sRGB 传递函数并嵌入 sRGB ICC，没有 ACES -> XYZ -> sRGB 矩阵与白点适配。这是错误标记色彩数据，不是风格差异。
- 现有测试 `tests/unit/test_color_management.py:42-52` 把错误语义当成回归基线，必须删除这种“6=LINEAR”断言，而不是修改注释继续保绿。

### F-ALG-2（致命）合成测试丢掉了所有场景色度

- 位置：`src/negative_lab/testing/synthetic.py:73-81`。
- 生成器先执行 `L = scene.mean(axis=2)`，再用同一标量 `L` 生成三个负片通道。所谓红/绿/蓝/黄色块在前向模型中已被变成灰度，因此现有合成测试只能证明亮度方向，不能证明色罩、色相、饱和度或通道解混正确。
- 更严重的是，合成前向模型与被测反向公式几乎同构，属于循环验证。

### F-ALG-3（致命）Dmin 被场景内容改写

- 位置：`src/negative_lab/pipeline/auto.py:93-120`。
- `Dmin` 的物理语义是未曝光片基/本底透射基准，但当前代码将其逐通道替换为 `max(片基统计, 画面内 p99.5)`。这会把场景高光、齿孔漏光、眶光和构图色彩注入色罩估计。
- 片基标量不能用场景内容“补亮”。解决暗角和导光不均应该使用 blank-light flat field 和/或片基空间曲面，而不是改写物理基准。

### F-ALG-4（致命）“统计中性”是自证循环

- 位置：`src/negative_lab/calibration/neutral_stats.py:17-56`、`src/negative_lab/pipeline/auto.py:176-201`。
- 当前方法先用当前参数反相，再从当前输出中选“看起来接近中性”的像素，然后用它们重估同一组通道参数。这只会优先选中已符合当前假设的像素，无法证明它们在原场景中真的是中性物体。
- 强色场景、无灰物体或整体色偏时，该算法可能改正场景本色。它不得再自动改写正式的卷级指数和白平衡。

### F-ALG-5（重要）技术反相、卷特性和风格曲线耦合，参数不可识别

- 位置：`src/negative_lab/pipeline/density.py:80-150`、`src/negative_lab/pipeline/params.py:15-33`、`src/negative_lab/calibration/neutral.py:41-156`。
- `Dmax`、`reference_exponent`、`red_ratio/blue_ratio`、`wb_high`、`wb_low`、`gamma`、`black`、`offset`、`exposure` 对亮度和通道颜色有重叠作用。没有标定目标时，多组参数可以在一两个取样点上给出相似结果，但在其他密度上迅速分岔。
- 当前默认 `gamma=1`、逐帧估计 `Dmax~0.8`，导致 G200-1 中调大量挤到高亮。darktable negadoctor 将技术密度范围与后端 paper grade 分开，其 paper grade 默认值为 4；NLP 也明确把 tone profile 放在转换与色彩分析之外。不得简单把 `gamma` 改成 4 就声称修复，必须先拆分参数语义。

### F-ALG-6（重要）逐通道标量公式无法表达染料串扰

- 位置：`src/negative_lab/pipeline/density.py:72-168`。
- C-41 橙色遮罩不是只叠在画面上的均匀橙色滤镜。彩色偶合剂在显影时形成与染料像反号的积分遮罩，用来抵消青/品红/黄染料的非理想次要吸收。因此数字反演至少需要在光学密度域评估通道混合，只用逐通道除以 `Dmin` 和幂指数不足以保证色分离。

### F-ALG-7（重要）质量门禁会放行视觉上明显失败的图像

- 位置：`src/negative_lab/diagnostics/quality_gate.py:36-77`。
- 现有门禁只检查 NaN/Inf 和像素是否超过 0/1，不检查工作空间一致性、色度塌缩、中性漂移、黑白点范围、肤色、色相次序或标定置信度。
- G200-1 的 `Dmax confidence=0.398`、统计中性置信度 0.4，仍被当作 `warn` 导出。对“自动默认结果”而言，这应是 fail/needs-user-calibration，不是成功但带警告。

### F-ALG-8（重要）RAW 方向和白平衡记录不真实

- 位置：`src/negative_lab/color/management.py:115-172`。
- `user_flip=-1` 表示按 RAW 元数据自动旋转，不是注释所说的“不自动翻转”。这会让所谓 SOURCE 坐标实际上依赖 LibRaw 隐式几何。
- `use_camera_wb=False` 和 `use_auto_wb=False` 时不得在 sidecar 写成笼统的 `as-shot`。需要依 rawpy/LibRaw 当前 API 验证实际 daylight/default multiplier，并记录真正数值。

### F-ALG-9（重要）整改进度文档已与实际证据冲突

- 位置：`docs/remediation-progress.md:28-58`、`docs/remediation-progress.md:60-103`、`docs/remediation-progress.md:105-135`。
- 文档将 P0-B 色彩管线、P0-C 自动颜色和 P0-D 统一主路径标成 PASS，还声称 GUI 冒烟通过。本轮已确认 P0-B/C 的色彩结论无效，GUI 冒烟实际失败。
- 不覆盖旧日志；执行 AI 应追加“2026-09-01 复核撤销 PASS”记录，将 P0-B/C 回退到 FAIL，P0-D 改为 PARTIAL，直到本规格的新门禁通过。

## 2. 公开资料能够证明什么

### 2.1 Negative Lab Pro（闭源，仅限公开事实）

NLP 公开指南能确认：

1. 相机 RAW 在转换前使用片边做白平衡，可以按卷同步；不同胶片应分别取样。
2. 使用专用、线性化且“untwisted”的相机 RAW profile；缺少该 profile 会损害转换质量。
3. 非胶片元素、片夹和直接漏光必须排除在场景分析之外；`border buffer` 会改变分析范围。
4. 转换会逐像素分离颜色通道、计算颜色校正并构造 tone curve；这不等于公开了具体算法。
5. 预饱和在反相前作用，会改变色分离和色相；Basic/Frontier/Noritsu 色彩模型、tone profile、中调/高光/阴影色彩调整是不同层。
6. Roll Analysis 要求同卷使用一致曝光和白平衡；可用含色卡的单帧作校准，并对帧间动态范围做自适应。

禁止写“NLP 内部使用某公式/神经网络”，除非有新的可审核一手证据。

### 2.2 darktable negadoctor（公开文档与 GPL 源码）

1. 输入是已经 input profile 处理的线性 RGB；官方文档建议线性 Rec.2020 或胶片乳剂 ICC，不是无名“相机线性 RGB”。
2. Dmin 从未曝光片基取样；Dmax 从只含已曝光画面的区域设定；扫描曝光偏置用来定黑点。
3. 阴影色偏和高光白平衡使用用户选定的中性阴影/高光区域，不假定任意场景自动满足 gray-world。
4. 相纸曝光、纸黑、paper grade 和高光压缩属于后端创意/显示映射，不应混入片基测量。

可以独立实现同类数学思路，但不得复制 GPL 源文件到本项目，除非用户先选择与 GPL 兼容的项目许可证。

### 2.3 RawTherapee Film Negative（公开文档）

1. 使用逐通道倒数幂模型，三通道指数依胶片类型、老化和条件变化。
2. 要求两个原场景中真正中性、亮度不同且未剪裁的取样点；每卷只需标定一次，可从同卷任意帧选点。
3. Film Negative 位于 RAW 工具链，中性点改变发生在白平衡之前，随后再对正片白平衡。

### 2.4 橙色遮罩的物理含义

Eastman Kodak 的积分遮罩专利 US2449966A 说明：青、品红、黄染料除主吸收带外还有次要吸收，彩色偶合剂在染料像形成的区域被破坏，未显影区域留下反号遮罩来抵消这些次要吸收。所以：

- 片基取样可以去除常量本底/光源色偏；
- 影像密度内的通道串扰仍需要卷/胶片/设备标定；
- “除以片基 RGB”不等于完成了整个去色罩。

## 3. 目标管线和数据模型

必须把“测量”、“技术反相”和“风格渲染”分成三层，任何一层不得偷偷改另一层的参数。

```text
RAW sensor / TIFF ICC
 -> 显式解码与线性化（真实命名的工作空间）
 -> 几何与有效画面 mask
 -> blank-light flat-field / 暗角校正
 -> 卷级片基 B_c(x,y) 估计（绝不使用场景 p99.5 覆盖）
 -> d_c = log10(B_c / max(T_c, eps))
 -> 密度域通道解混 M_density（可标定时启用）
 -> 卷/胶片通道特性曲线 f_c(d)
 -> 场景线性正片基线（Basic/Neutral）
 -> 明确的 working RGB -> output RGB 色域矩阵与白点适配
 -> 独立 tone mapping / 黑白点 / 高光滚降
 -> 可选风格 LUT（后续版本）
 -> sRGB 传递函数 -> 量化 -> ICC -> 事务导出
```

### 3.1 必须新增的领域对象

```text
CaptureProfile
  camera_id, lens_id, light_id, decoder, decoder_params
  input_space, matrix_to_xyz/output, flat_field_id

RollCalibration
  roll_id, stock_id, process(c41|ecn2|bw), capture_profile_id
  base_observations[], base_rgb_or_surface
  density_matrix, characteristic_curves, neutral_observations[]
  calibration_method, confidence, source_hashes, version

SceneAnalysis
  frame_roi, valid_mask, black/white density endpoints
  exposure/tone suggestions only
  must_not_modify: base, density_matrix, characteristic_curves

RenderIntent
  neutral|lab-like|cinematic
  tone_curve, contrast, black, white, highlight_rolloff
  color_look_id (optional; never part of technical calibration)
```

## 4. 执行阶段

### P0-A：立即阻断错误色彩语义

1. 在 GUI/CLI/批处理显示 `C-41 RAW automatic conversion: experimental`；自动结果缺少有效卷标定或工作空间未验证时，QualityGate 返回 `needs_calibration` 或 `fail`，不再只是 warn。
2. 禁止 `output_color` 裸整数。通过命名枚举记录 `name` + `value`，并修正 `user_flip`、WB policy 和 sidecar 语义。
3. 如保留 ACES，实现并测试 ACES 到输出 sRGB 的线性基色变换、D60 -> D65 色适应和最终 TRC；不得只加 sRGB TRC。
4. 在没有实证前不盲目选择 ACES。建立 `scripts/benchmark_color_pipeline.py`，在同一 NEF 上对照至少三条路径：
   - A：`rawpy.ColorSpace.raw` 传感器线性域反相，反相后用经验证的相机矩阵转 XYZ/输出空间（RawTherapee 类路径）；
   - B：经 input profile 的 linear Rec.2020 反相，再显式转 sRGB（negadoctor 类路径）；
   - C：ACES-linear 反相，再完整转 sRGB。
5. 每条路径记录有效画面内的逐通道输入剪裁、负值处理、片基 RGB、中性残差、色卡 DeltaE2000 和联系表。只有留出集更好的路径才能成为默认。

P0-A 通过标准：sidecar/ICC/像素数值的工作空间一致；灰阶 ramp 、标准基色和色域边界的独立参考变换误差在双精度容差内；现有“6=LINEAR”测试消失。

### P0-B：重建非循环合成测试

1. 删除“彩色块 -> `mean(RGB)` -> 三通道同一密度”的前向路径。
2. 前向模型使用与逆算实现不同的组件：
   - 每通道独立 H-D/characteristic curve；
   - 非对角 3x3 染料/扫描器串扰矩阵；
   - 片基颜色、空间不均、眶光、暗角、颗粒、读出噪声和逐通道剪裁；
   - 真正保留 RGB 的色块与 ColorChecker 近似光谱响应。
3. 新增必败组：交换通道、漏掉工作空间矩阵、对强红/蓝场景执行 gray-world、用场景 p99.5 改写 Dmin。
4. 断言至少包含：色相次序、色度不塌缩、中性阶跨密度不漂移、单调性、剪裁和已知参数恢复误差。

P0-B 通过标准：每个错误实现都会至少触发一个失败测试，且测试不调用生产反相函数生成“真值”。

### P0-C：建立卷级测量与 flat-field

1. 从 `auto_analyze()` 移除 `dmin=max(dmin, scene_hi)`。片基只能来自：同帧 verified 片基 ROI -> 同卷独立未曝光片基 -> 同卷经审核先验，三级之外必须 needs_calibration。
2. 支持 blank-light 帧。使用正值、平滑、低频的每通道 flat-field 修正照明不均和镜头暗角，保留修正前/后剪裁诊断。
3. 片基跨画面明显不均时，拟合稳健低阶曲面 `B_c(x,y)`；不均性不显著时才退化为标量 `B_c`。
4. 卷级标定锁定 camera + lens + light + capture exposure/WB + film stock/process。任一项变化必须降低置信度或建新 profile。
5. 场景分析只能提议黑白密度端点和 tone，不得改卷级 base/matrix/curve。

### P0-D：密度域模型 bake-off

在不删除现有 negadoctor-like 实现的前提下，以独立 backend 实现并对照：

1. `power-reciprocal`：RawTherapee 类逐通道倒数幂基线，指数只由同卷两个及以上不同密度中性点或胶片先验约束。
2. `density-affine`：以 `d=log10(B/T)` 为特征，用带 ridge 正则和中性约束的 3x3 密度矩阵 + offset 映射到场景 log-RGB。
3. `density-curves`：只有当线性矩阵在校准集和留出集都显示系统性密度残差时，增加单调 1D 样条/分段曲线；禁止一开始上高阶多项式或 3D LUT 过拟合小样本。

所有 backend 必须输出同一个 `TechnicalPositive`（场景线性、已命名色域、未套风格）。黑白点、中调、对比度、高光滚降和 LUT 不得参与 backend 标定损失函数。

### P0-E：取消自证循环，建立明确校准层级

自动色彩参数优先级固定为：

```text
verified 色卡/灰卡卷标定
 -> verified 同卷亮+暗中性 ROI
 -> 同胶片+设备+光源的版本化先验
 -> 用户互动取点
 -> 仅片基的 neutral baseline + needs_calibration
```

- `statistical gray-world` 只能作为显示给用户的可选建议，不得默认写入 RollCalibration。
- 单个中性点不得同时自由拟合 exponent + wb_high + wb_low。优化前列出可识别参数和观测方程；观测不足则锁定先验并降低置信度。
- 每个校准结果记录 ROI、原始 source hash、求解方法、残差、条件数、样本数和是否进入留出评估。

### P0-F：技术基线与风格彻底分离

1. 默认首先产生 `Neutral/Basic` 技术正片，不使用 Frontier/Noritsu 风格。
2. 为 tone engine 建立独立参数：black point、white point、mid-gray、contrast/paper grade、highlight/shadow rolloff。其变化不得重算 Dmin、密度矩阵或卷特性曲线。
3. 预饱和若实现，必须有独立 ADR，说明在哪个线性/对数空间作用、为什么不破坏色相和中性轴；未验证前默认关闭。
4. 风格 LUT 不属于本轮 P0。禁止用 LUT 掩盖技术反相的色偏。

### P0-G：重写色彩质量门禁

QualityGate 新增状态 `needs_calibration`，并至少接收：

- `valid_image_mask`：排除片边、齿孔、直接灯箱、手动蒙版和非画面区域；
- 工作空间/输出 ICC 一致性；
- 片基、flat-field、卷级标定置信度；
- 逐通道 input/output clip 与转换前矩阵剪裁；
- 存在验证中性 ROI 时的 DeltaE2000 和跨密度中性漂移；
- 存在色卡时的色度块 DeltaE2000、色相次序和色度压缩；
- 指定视觉回归图的亮度/色度分位和人工验收签字。

门禁分级：

- `fail`：NaN/Inf、色彩空间标记不一致、必要色域变换缺失、严重剪裁或真实回归超出硬阈值；
- `needs_calibration`：彩色负片缺卷级片基/设备 profile，或有效标定信心不足；
- `warn`：技术管线已验证，但局部剪裁或场景色彩建议需用户注意；
- `ok`：所需 profile 存在，且本帧机器门禁全部通过。

## 5. 样片与标定协议

用户已明确可提供样片。执行 AI 首先生成一份可勾选的接收清单，然后用 `scripts/sample_intake.py` 只读探针、哈希和分组。

### 5.1 每个标定组合必需

1. 同卷未曝光片头/片边 1-3 张，显影工艺与正常帧相同。
2. 不放胶片的 blank-light RAW 3 张，与正常翻拍同机身、镜头、光圈、对焦位置、灯箱亮度和几何。
3. 同卷正常、偏暗、偏亮、强色、无中性物体场景各至少 2 帧。
4. 同卷拍摄的 ColorChecker/灰卡帧，记录原场景光源；至少一张包含两个以上不同亮度中性块。
5. 所有翻拍 RAW 使用固定手动曝光、固定光源，并确保传感器原始通道不剪裁；不以相机 JPEG 直方图代替 RAW 通道检查。
6. 标定集和留出集按整卷分割，不得把同卷相邻帧随机拆到两侧。

### 5.2 NLP 对照产物

每个对照样片提供：

- NLP 版本、RAW profile、片边 WB 数值、Roll Analysis 开关、Color Model、Pre-saturation、Tone Profile、Temp/Tint、黑白剪裁设定的截图或文本记录；
- 16-bit ProPhotoRGB TIFF 技术对照，以及可选的用户喜好风格对照；
- 对齐的 frame ROI 和中性/色卡 ROI。

NLP 文件只用于行为研究和并排视觉对照，不抽取、复制或反编译商业资源，不作为唯一训练真值。

## 6. 验收标准

### 6.1 硬门禁

1. 空间语义：从 RAW 到导出的每个矩阵、白点、TRC、ICC 和 sidecar 名称一致；不再存在“ACES 数值 + sRGB 标记”。
2. 信息保留：有效画面内任一输入通道的解码剪裁 >0.1% 则至少 warn，>1% 则 fail；标定帧的核心色卡/中性 ROI 不允许剪裁。
3. 中性：verified 灰卡中位 DeltaE2000 <=3，p95 <=5；同卷不同密度灰阶的 a*/b* 不得呈系统漂移。
4. 色卡：与记录的参考照明/适配条件一致时，留出色块 DeltaE2000 中位 <=6，p95 <=10；若达不到，保持 P0 fail 并报告最差色块，不降阈值。
5. 卷一致性：留出卷同一中性物的帧间 DeltaE2000 中位 <=3；不允许逐帧 gray-world 伪造一致。
6. 结构：新合成色块的色相次序必须保留；强红或强蓝单色场景不得被自动拉成灰色。
7. 人工：标定卷和留出卷联系表的方向、裁切、黑位、高光、阴影、肤色、中性、整体色偏必须逐张记录，并由用户确认。

上述 DeltaE 阈值仅对有可追溯色卡/中性参考的样片生效。无真值普通帧不得伪造 DeltaE 结论。

### 6.2 G200-1 指定回归

- 不得再导出底部非画面灰区。
- 不得使用场景 p99.5 覆写片基。
- 在相同裁切下，技术基线的线性亮度 p50 必须脱离现有 0.735 的冲淡区间，但最终阈值必须由用户确认的 Neutral/Basic 输出固化，不直接复制 NLP p50。
- 色度分布不得像现有输出一样整体塌缩；使用色卡和用户确认联系表定阈值。

## 7. 与 v2 整改项的合并顺序

1. 先执行本文 P0-A/P0-B，否则所有真实颜色回归都在错误空间上运行。
2. 执行 P0-C/P0-D/P0-E，只产生 Neutral/Basic 技术基线。
3. 执行 P0-F/P0-G，生成新联系表并等待用户签字。
4. 再回到 v2 P0-H：完成事务性 TIFF+JPEG+sidecar 三件套导出，禁止任一步失败留下半套。
5. 再执行 v2 P0-I：修复 GUI/E2E 唯一临时目录、当次 manifest 和 GBK 控制台输出。
6. 保持发布样片门禁红灯，直到黑白、可解码 ECN-2、真实 TIFF/DNG、第二设备/光源、许可证和安装/卸载验证全部完成。

## 8. 每阶段必须留下的证据

```text
docs/color-pipeline-v2.md              工作空间、矩阵、白点、TRC ADR
docs/demask-research.md                候选模型、公开依据、未知项
docs/calibration-protocol.md           样片拍摄、取样、分组和污染防护
outputs/benchmarks/<run-id>/           数值报告、联系表、参数、环境和哈希
outputs/quality/<run-id>/               色卡/中性 DeltaE、剪裁、卷一致性
fixtures/manifest.json                 calibration/holdout 按卷隔离
fixtures/calibrations/*.json           版本化 RollCalibration，不含原图
```

每个 benchmark 必须记录 Git commit、Python/rawpy/LibRaw 版本、平台、源文件 SHA-256、实际解码参数、实际工作空间和运行时间。

## 9. 验证命令

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_color_management.py
.\.venv\Scripts\python.exe -m pytest -q tests\synthetic
.\.venv\Scripts\python.exe -m pytest -q tests\regression\test_real_samples.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
uv pip check --python .\.venv\Scripts\python.exe
.\.venv\Scripts\python.exe scripts\benchmark_color_pipeline.py --manifest fixtures\manifest.json
.\.venv\Scripts\python.exe scripts\e2e_cli_smoke.py
.\.venv\Scripts\python.exe scripts\gui_smoke.py
.\.venv\Scripts\python.exe scripts\verify_package.py
```

任一 P0 命令失败时停止发布流程，保留失败证据，不删测试、不改宽阈值、不更新 golden 图来追认错误输出。

## 10. 明确禁止

- 禁止将 NLP 称为开源，或声称已知其内部公式。
- 禁止用全图 NLP DeltaE 代替对齐的色卡/中性 ROI 与人工验收。
- 禁止使用场景最亮/最暗像素修改卷级色彩参数。它们只能建议场景 tone endpoints。
- 禁止把强色场景的 gray-world 结果当成高置信度标定。
- 禁止只改 `gamma`、`Dmax` 或某一张 G200-1 的固定参数就声称算法修复。
- 禁止在未经输出基色变换时嵌入 sRGB ICC。
- 禁止复制 darktable/RawTherapee GPL 源码到当前许可证未决的项目。
- 禁止在 P0 真实色彩门禁通过前训练神经网络。当前问题是色彩管理、可识别性和标定不足，不是网络规模不足。

## 11. 调研来源（访问日期 2026-09-01）

1. Negative Lab Pro, Getting Started: <https://www.negativelabpro.com/guide/basics/>
2. Negative Lab Pro, Understanding the RAW Camera Profiles: <https://forums.negativelabpro.com/t/understanding-the-raw-camera-profiles-in-negative-lab-pro/2796>
3. Negative Lab Pro, Roll Analysis Q&A: <https://forums.negativelabpro.com/t/roll-analysis-q-a-help/6430>
4. darktable User Manual, negadoctor: <https://docs.darktable.org/usermanual/development/en/module-reference/processing-modules/negadoctor/>
5. darktable source, `src/iop/negadoctor.c`: <https://github.com/darktable-org/darktable/blob/master/src/iop/negadoctor.c>
6. RawPedia, Film Negative: <https://rawpedia.rawtherapee.com/Film_Negative>
7. Eastman Kodak Co., US2449966A Integral mask for color film: <https://patents.google.com/patent/US2449966A/en>
8. Cuchara Valley Landscapes, Light Source Color Compensation: <https://cuchara.photography/2018/06/light-source-color-compensation-for-film-negatives/>

其中 1-7 用于算法/工作流事实；8 是实践性二手资料，只支持“光源光谱和通道曝光余量会影响翻拍质量”，不作为算法真值。

## 12. 本轮实际验证快照

| 项目 | 结果 | 解读 |
|---|---|---|
| `pytest -q` | **1 failed, 125 passed**, 288.97 s | 失败是发布样片门禁：缺黑白、可解码 ECN-2、真实 16-bit TIFF、第二设备/光源 |
| `compileall` | PASS | 只证明 Python 语法/导入层面可编译 |
| `uv pip check` | PASS，21 packages | 只证明已安装依赖的元数据兼容 |
| `e2e_cli_smoke.py` | PASS，但中文控制台乱码 | 仍使用固定 `outputs/e2e`，未证明冒烟幂等 |
| `gui_smoke.py` | **FAIL** | 历史三件套被一起计数，断言 `len(exports)==3` 失败 |
| `verify_package.py` | **FAIL** | GBK 控制台无法编码 `✓/✗`，在真正验证前就崩溃 |
| G200-1 视觉 | **FAIL** | 冲淡、色度塌缩、非画面底边；现有 `docs/visual-acceptance.md` 仍等待用户确认 |

测试中的中文在当前 GBK 终端中也呈现乱码；失败原因仍可由测试名和矩阵内容确认。日志文件应使用 UTF-8，控制台状态使用 ASCII。
