# 算法笔记（Phase 1）

## 符号约定（最重要，勿改）

`z_c = log10(max(t_c,ε)/max(Dmin_c,ε))`，与 darktable negadoctor 源码同方向：

- 未曝光片基：`t = Dmin` → `z = 0`；
- 画面内容：`t ≤ Dmin` → `z ≤ 0`；场景越亮 → 胶片越密 → `z` 越负；
- 反相输出随场景亮度单调递增。

方向由 `tests/synthetic/test_direction.py`（斜坡单调、片基映射到黑）与
`tests/unit/test_density_model.py::TestSignConvention` 锁定。
若改用正密度 `d = log10(Dmin/t)`，必须重写 §4.3 全部反算并新增等价性测试（规格原文要求）。

## 密度域主模型（§4.3 逐字实现）

```
wb_high_internal_c = wb_high_c / max(Dmax, ε)
offset_c           = wb_high_c · offset · wb_low_c
q_c                = wb_high_internal_c · z_eff_c + offset_c
black_internal     = -exposure · (1 + black)        # 内部带符号量
paper_linear_c     = max(0, -(exposure·10^q_c + black_internal))
positive_linear_c  = paper_linear_c ^ gamma
```

代数等价形式 `paper = exposure·((1+black) − 10^q)` 用于解释方向：
`z=0, offset=0, black=0` 时 paper=0（片基→纸黑）；`z→−Dmax` 时 paper→exposure·(1+black−0.1)。

`black`（用户语义，≥0）与 `black_internal`（带符号内部量）严格分离，见 `params.py` 注释。

## 幂指数修正（§4.4）

`z'_c = k_c·z_c`，`k_G=reference_exponent`，`k_R=k_G·red_ratio`，`k_B=k_G·blue_ratio`。
校正比例与乳剂差异**互反**：若红通道密度斜率是 1.15 倍，则 `red_ratio=1/1.15` 才能拉平。
由亮、暗两个不剪裁中性点估计；样本不足时保持 1.0 并降置信度（不猜测）。

**信噪加权**（合成测试暴露的问题）：z 域噪声 σ_z≈σ_t/(t·ln10)，
低透射率（亮场景）样本的通道比例噪声可达数个百分点；等权平均会让噪声样本
在 k=1 的真值上产生 ~1% 的假校正，进而在暗部引入可见色偏。
估计器按 `w=(t̄·|z̄|)²` 加权中位，生产路径与测试一致。

## Dmax / offset 自动估计（§4.5.4）

- Dmax：剔除零剪裁像素（t≤1e-5，密度信息已丢失）后取 z 的 2% 分位相反数。
  教训：用 0.5% 分位时，高密度胶片（Dmax=2.8）的噪声零剪裁像素把估计推到
  clip 上限 4.0（误差 43%）；剔除+2% 分位后误差 <3%。
- offset：令画面最暗 0.5% 像素恰落到纸黑的启发式，截断 [-0.5, 1.0]，可手动覆盖。
- Dmax 非法（≤0/NaN/字符串）→ 回退 2.0 并警告，绝不产生 Inf（§4.3）。

## 置信度与警告

Dmin：样本量、尘点剔除保留率、通道散布三分量取 min；ROI 亮度低于全图 99 分位
的 80% → 提示“ROI 可能不在片基上”。无中性点 → 置信度 0 + “无中性参考”警告，
仍输出结果（§4.6.1），不静默。

## 未验证 / 已知边界

- 以上全部基于合成数据；真实 NEF 的镜头眩光、片基不均（边缘渐晕）
  对 robust 统计的影响未验证（Phase 2/3 用 1298_G200 样片回归）。
- 高光软压缩为 Reinhard 型单调尾巴，strength=0 默认关闭；与 negadoctor 的
  压缩曲线不追求一致。
- ICC/sRGB 仅做了固定 sRGB 编码；输入侧 ICC 配置处理属 Phase 2。
