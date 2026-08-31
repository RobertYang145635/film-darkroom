# 色彩管线 ADR（整改 P0-B；决策 docs/color-pipeline.md）

> **2026-09-01 v3 修订**：本 ADR 原"output_color=6=LINEAR"决策被复核
> 证伪（rawpy 0.27 中 6=ACES）。现行决策：临时默认 sRGB-linear(D65)
> （oc=sRGB + gamm=1，与导出 sRGB TRC/ICC 自洽）；最终默认由
> scripts/benchmark_color_pipeline.py 三路径 bake-off 在留出集上决定。
> 工作空间名：srgb-linear-d65-normalized。Dmin 只能来自片基（禁止场景
> 分位覆写）；gray-world 仅作建议不自动应用。详见
> docs/remediation-progress.md 撤销记录。

状态：生效（v3 修订版）。任何实现与本 ADR 冲突时以本 ADR 为准；修改本
ADR 必须提升 sidecar `schema_version` 或公式等价性测试（原规格 §4.3 末段、
整改 §1.6）。

## 1. RAW 解码策略（回答"这些数字从哪来"）

显式参数（全部写入 `LoadResult.meta.raw_decode_params`，禁止依赖库默认值）：

| 参数 | 值 | 理由 |
|---|---|---|
| `use_camera_wb` | False | 不应用相机白平衡增益矩阵 |
| `use_auto_wb` | False | 自动白平衡不可复现 |
| `user_mul` | 不设置 | 实测 `(1,1,1,1)` 会改变 LibRaw 内部通道归一化（片基 R 0.17→0.11）；采用 LibRaw 默认 as-shot 乘数。**任何逐通道增益在 z=log10(t/Dmin) 中被 Dmin 归一化完全抵消**，不影响密度模型正确性，但策略必须如实记录 |
| `output_color` | 6 (LINEAR) | 应用相机色彩矩阵、无传递函数：原规格 §4.1.1 的"线性 RGB"语义与 darktable negadoctor / RawPedia 一致；实测 sensor-raw(0) 的 R/G 通道在橙片基下几乎不可分（串扰），不适合逐通道密度模型 |
| `gamma` | (1,1,1,1) | 线性输出（§4.1.2 反相前禁止显示曲线） |
| `no_auto_bright` | True | 禁止自动亮度 |
| `highlight` | 0（none，不过滤） | 高光剪裁由诊断报告，不静默修复 |
| `output_bps` | 16 | 线性 16 位进入归一化 |
| 黑/白电平 | LibRaw 内部扣除 | 记录 `black_level_per_channel`、`white_level` 到 meta |

归一化：解码后线性值除以运行实测满值（保守，不 clip_percent），
得到"归一化线性透射率 t∈[0,~1]"。

## 2. 工作空间（回答"这些数字属于哪个色域"）

- **线性工作空间**：相机线性 RGB（LibRaw LINEAR 输出：应用相机色彩矩阵、
  无传递函数；基色=该相机/镜头组合的拍摄原色，非 sRGB/Adobe 等标准 RGB），
  数值语义=相对透射率（除以 Dmin 后进入 z 域）。
- 密度管线在矩阵之后**不再做通道间混合**；去色罩=逐通道除法（对数域减法）。
- 该空间没有标准名称，sidecar 记录 `working_space: "camera-linear-normalized"`
  + 输入设备/镜头/光源（unknown 就写 unknown）+ `raw_decode_params`。

## 3. 输入侧 ICC / 传递函数（TIFF/JPEG）

| 输入 | 策略 |
|---|---|
| TIFF 带 ICC | 读 ICC；若 TRC 为线性（或 gamma=1）→ 视为线性数据，记录 profile 描述与哈希；若 TRC 非线性 → 用 TRC 反解到线性并记录；无法判定 → 拒绝（见下） |
| TIFF 无 ICC | **不允许默认线性**。CLI/GUI/批处理必须显式给 `--tiff-encoding linear\|srgb\|adobe`；未指定 → ReadError，说明风险与选项（"未知时阻止误处理"） |
| JPEG/PNG | 永远按 sRGB 显示编码读入再线性化；警告"显示编码源仅可用于预览对比"（§4.1.2） |
| 不支持的 profile | 清晰失败：报路径、profile 描述、建议 |

## 4. 输出侧（回答"这些数字要去哪"）

| 输出 | 传递函数 | ICC |
|---|---|---|
| `positive_linear`（内存/诊断） | 线性 | 无（工作空间内） |
| 显示预览 | sRGB 分段编码 | 屏幕 sRGB |
| **TIFF16** | sRGB 编码，16 位 | **嵌入 sRGB IEC61966-2.1**（与像素一致） |
| **JPEG8** | sRGB 编码，8 位 | **嵌入同一 sRGB profile** |
| sidecar | — | 记录 `output_encoding: "srgb-encoded"`、`output_icc_sha256`、输入 profile 状态、RAW 白平衡参数 |

TIFF 与 JPEG 编码/ICC 一致 ⇒ 任何支持 ICC 的查看器中外观一致（整改 §2）。
线性 16 位 TIFF 导出（scene-linear + 无 ICC 或自定义）留待后续版本，
需要时另立枚举值，不改现有语义。

## 5. 顺序（固定，写入 quality_gate 测试）

```
解码/线性化 -> 几何(旋转/裁切) -> 除尘修复(透射率域) -> sanitize(NaN/Inf 记录+替换)
-> z = log10(max(t,eps)/max(Dmin,eps)) -> z' = k_c*z_c -> 密度域模型(paper/gamma)
-> 高光软压缩(线性域) -> 负值/超界 clamp 到 [0,1]（clamp 前记录逐通道剪裁）
-> sRGB 编码 -> 量化(16b/8b) -> 嵌入 ICC -> 写后重读验证
```

色域外值：密度模型输出为逐通道标量，不存在色域外（无矩阵混合），
只有越界（<0 或 >1）——统一在第 8 步记录并 clamp。

## 6. 验证责任

- `save_tiff16/save_jpeg8` 写后重新读取：ICC 存在且哈希与声明一致，
  像素编码枚举一致；不一致 → 抛错并清理（事务导出在 P1 收口）。
- 灰阶中性保持测试：R=G=B 输入任意编码往返后仍 R=G=B。
