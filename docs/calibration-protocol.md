# 标定与留出协议（v4 §12.1/§14.6）

## 1. 样片目录约定

- 原始样片（RAW/TIFF/JPG/NLP 对照）只读，不进 Git，不覆盖/移动/重压缩。
- 目录约定：`<拍摄日期>_<批次>/<卷号>_<胶片>/翻拍/`，NLP 对照放同目录
  `NLP/` 子文件夹；用户转换的 ECN-2 TIFF 放 `<批次>/ECN2/`。
- 每个新样片批次入册时运行 `scripts/build_fixture_manifest.py`：
  登记路径、真实解码状态、尺寸、位深、SHA-256（manifest v2 schema）。

## 2. 卷级分组与隔离

- calibration / holdout **按卷隔离**（当前：1298/1300/5219=标定，
  1299/5207=留出，ECN2-converted=留出）。
- 禁止从 holdout 学任何参数（Dmax 先验、解混矩阵、通道曲线）。
- 卷级标定文件放 `fixtures/calibrations/<roll>.calib.json`
  （RollCalibration，含 source/provenance/confidence）。

## 3. 标注与用户确认

- 标注文件 `fixtures/annotations/*.json`，schema v2：output_crop_roi /
  image_area_roi / base_roi 三语义分离 + source_sha256。
- **agent 只能产生 draft**；`verified_by="agent-inspected"` 不构成用户签字。
  只有用户在 GUI 框选并点「保存区域为标注」才是 verified。
- 旧 schema（frame_roi 单字段）加载时迁移为 draft + needs_reconfirm。

## 4. 标定来源层级（v4 §5）

1. user-points：用户单帧中性点（当场生效）
2. chart-calibration：已知目标监督标定（解混矩阵；当前**无色卡，不可用**）
3. base-only：仅片基（needs_calibration，experimental）

可辨识性边界（docs/algorithm-notes.md §2.3）：中性样本只张成一维密度射线，
解混矩阵从中性点**原理上不可辨识**；完整 3x3 需要多色相×多密度的已知目标。

## 5. 参考图用途边界

- NLP 导出与竞品截图只作**用户偏好对照**，不是色彩真值；
  不得跨场景用截图直方图定阈值，不得整图 DeltaE 充当验收。
- 同帧对比只允许：中性色偏、色相角、彩度比、亮度分位、剪裁。

## 6. 当前样本清单与缺口（2026-09-01）

| 类别 | 状态 |
|---|---|
| C-41 彩色 ≥2 卷 | ✅ 1298/1300（+1299 留出） |
| 可解码 ECN-2 | ✅ 用户转换 72 张 TIFF（ECN2/） |
| 真实 16-bit TIFF 输入 | ✅ ECN2 转换卷（NX Studio 导出） |
| 黑白 ≥1 卷 | ⛔ 用户裁决 2026-09-01：跳过验证 |
| 色卡帧 | ⛔ 无法补拍（胶卷已冲洗）——监督标定/ΔE2000 验收不可用 |
| blank frame | ⛔ 无法补拍——flat-field 上限=strip-1D 部分校正 |
| 第二设备/光源 | ⛔ 无——跨设备结论一律不出 |

## 7. 验收口径（依据用户裁决修订）

- G200-1 最低线（v4 §12.2）：剪裁 <1%、亮度 p50 差 ≤0.12、
  sat_p90 ≥ 75%·NLP 对照——当前 3/3 达标。
- 最终验收 = 最低线 + 用户逐卷联系表签字（接受/需调整/拒绝）。
- 缺口项在 docs/support-matrix.md 如实标注，不伪造 fixture。
