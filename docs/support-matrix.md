# 支持矩阵（由真实解码探针生成，整改 P0-A）

状态含义：`verified`=可解码且 calibration+holdout 均有样片；`missing-fixture`=无样片，**禁止宣称支持**；`unsupported`=样片存在但解码器不支持。

| 能力格 | 状态 | 样片卷 | 说明 |
|---|---|---|---|
| C-41 彩色负片 (camera RAW) | `verified` | 1298_G200, 1299_G200 | Kodak Gold 200 两卷（1 标定 1 留出） |
| 黑白负片 | `missing-fixture` | — | 无任何样片 -> 不得宣称支持（原规格 §2.1 胶片类型） |
| ECN-2 彩色电影负片 | `unsupported` | 1171_5219, 1172_5207 | 5219/5207 共 72 张为 Z8 AV1 高效 NEF，LibRaw 0.22.1 实测不支持（docs/blockers.md B-1）；待用户转 16 位 TIFF 后重新探针 |
| 未知类型 (CN400) | `missing-fixture` | 1300_CN400 | film_type=unknown，不参与类型宣称 |

- 可解码输入格式：.nef（16 位 TIFF 仅合成样片，真实扫描/导出 TIFF 为 missing-fixture）
- 真实 16 位 TIFF 样片：0 张 -> **missing-fixture，不得宣称扫描仪路线已验证**
- DNG 样片：0 张 -> missing-fixture
- 设备覆盖：1 台（NIKON） -> **单设备，多设备结论一律未验证（§12 末段）**
- 光源覆盖：全部 unknown -> 多光源结论未验证
- 灰卡/色卡样片：无 -> 中性区 ΔE2000 验收（§8.4）当前无法执行，需用户提供含色卡样片或人工取点

生成命令：`.venv\Scripts\python.exe scripts/build_support_matrix.py`
