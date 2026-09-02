# 支持矩阵（由真实解码探针生成，v5 §4.2/§14.3）

状态含义：`verified`=可解码且 calibration+holdout 均有样片；`missing-fixture`=无样片，**禁止宣称支持**；`unsupported`=样片存在但解码器不支持；`out-of-scope`=v5 P0 明确不覆盖。

> v5 §0.2.5：黑白负片与反转片不在 P0，因此**不再**作为发布门禁项。

| 能力格 | 状态 | 样片卷 | 说明 |
|---|---|---|---|
| C-41 彩色负片（相机 RAW 翻拍） | `verified` | 1298_G200, 1299_G200 | 1298_G200 标定 / 1299_G200 留出（Nikon Z8 + 105 micro，LED 背光） |
| ECN-2 彩色电影负片 | `verified` | 1171_5219, 1172_5207, ECN2_5207, ECN2_5219 | ECN2_5207 标定 / ECN2_5219 留出：Z8 高效 NEF 经 Lightroom 导出的 16 位 TIFF（LibRaw 0.22 无法解码 AV1 RAW，docs/blockers.md B-1） |
| 未知处理类型（CN400） | `missing-fixture` | 1300_CN400 | film_type=unknown 不阻断转换与导出（v5 §6.3），只是不参与类型宣称 |
| 黑白负片 | `out-of-scope` | — | v5 P0 排除项（§2.3），不作为门禁 |

## 输入路径

- 可解码输入格式：.nef, .tif
- 相机 RAW：97 张（卷：1298_G200, 1299_G200, 1300_CN400）
- 16 位 TIFF：72 张（采集类型：camera_raw_export_tiff）
- 真实**扫描仪** TIFF：0 张 -> missing-fixture，扫描仪路线未验证
- DNG（mosaic / Linear）：0 张 -> missing-fixture，DNG 路线未验证（代码路径已实现，仅缺真实样片）
- 设备覆盖：1 台（NIKON） -> **单设备，跨设备泛化一律未验证（v5 §9.3.4）**
- 光源覆盖：全部 unknown -> 多光源结论未验证

## 未验证即不宣称（v5 §14.3）

- 没有真实 fixture 的输入/胶片/设备组合一律标 `missing-fixture`，不得伪造 PASS；
- 通用先验回退、卷级片基共享与三候选在缺样片组合上只做过合成与跨卷迁移测试，不写成设备级结论。

生成命令：`.venv\Scripts\python.exe scripts/build_support_matrix.py`
