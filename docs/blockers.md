# 阻塞与风险记录（执行契约 §6：具体证据，不静默降级）

## B-1（已确认，影响 ECN-2 样片）：Nikon Z8 高效 NEF（AV1 压缩 RAW）无法解码

- **现象**：`20260506_SAMPLE-ID/1171_5219`（36 张）与 `1172_5207`（36 张）全部
  `rawpy.imread().unpack()` 抛 `LibRawFileUnsupportedError: Unsupported file format or not RAW file`。
- **证据**：
  - 两卷文件结构完整：RAW 条带 `offset+bytecount` 恰好等于文件大小（未截断）；
    IFD 与可正常解码的 G200/CN400 卷完全同构（NIKON Z 8，固件 Ver.03.10，
    SubIFD1 comp=34713/Deflate，14bit，8280×5520）。
  - RAW 条带首 24 字节为 `ff 10 ff 50 00 22 43 4f 4e 54 52 41 43 54 ...`，
    ASCII 含 `CONTRACT`，`ff10ff50` 为 AV1 OBU 定界符模式 →
    这是 Z8 的「NEF(高效)/High Efficiency」记录格式（RAW 用 AV1 帧内编码）。
  - 本机 rawpy 0.27.1 内嵌 LibRaw 0.22.1（当前 PyPI 最新），对该格式返回 UNSUPPORTED。
  - 对照：`1298/1299_G200`、`1300_CN400/翻拍`、`1300_CN400/翻拍2` 共 97 张
    RAW 条带首字节为 `eb??eb??`（传统 Nikon 有损压缩），全部解码成功。
- **影响范围**：仅 ECN-2 候选样片（5219/5207）；P0 的 C-41/TIFF/DNG 路线不受影响。
- **不做的降级**：不提取 NEF 内嵌 JPEG 预览冒充 RAW 输入（预览经过相机白平衡与
  色调曲线，违反 §4.1.2；且会静默伪造 RAW 支持，违反执行契约 §6）。
- **用户可行的解锁路径**（任选其一）：
  1. 用 Nikon NX Studio（免费）把这 72 张批量导出 16 位线性 TIFF（本项目 TIFF 路线直接支持）；
  2. 相机内或 NX Studio 转存为「无损压缩/未压缩 NEF」；
  3. 等待 LibRaw 增加 Nikon AV1-NEF 支持后升级 rawpy（本项目会持续检测）。
- **清单标记**：`fixtures/manifest.json` 每条含 `decode_status` 字段
  （`ok` / `av1_unsupported`），批处理遇到 `av1_unsupported` 将逐张报告失败原因。

## B-2（观察项）：真实翻拍样片的结构差异

- 真实翻拍外圈是黑边（透射率 ~0.0004），片基在黑边内侧。早期"画面外框"启发式
  把黑边当片基，Dmin 置信度归零。已由 `calibration/border.py`（帧修剪 + 边缘带
  对比选择）修复，合成测试 `tests/unit/test_border_detect.py` 锁定。
- 帧内仍可能包含带字样的不透明片边（"KODAK 500"），其超暗像素曾把逐通道 Dmax
  估计推到 2.96（真值 ~1.9）。已改为亮度域分位统计修复。

## B-3（未决，发布前必须处理）：项目许可证

- 见 `docs/dependency-licenses.md` 结论：无技术阻塞，但 §11 要求用户本人选择
  GPL-3.0 / MIT / Apache / 闭源免费。当前按"闭源可行"最严口径准备。
