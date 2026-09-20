# 阻塞与风险记录（执行契约 §6：具体证据，不静默降级）

> 2026-08-31 整改 P0-A 更新：本文件与 `fixtures/manifest.json` 的矛盾（旧清单
> 用 24 字节启发式把 72 张 AV1 NEF 标为 `ok`）已修复——清单现由
> `negative_lab.io.probe` 的**真实最小解码探针**生成（schema v2），
> 实测 `ok=97 / unsupported=72`，与 rawpy 行为一致。

## B-1（已确认，影响 ECN-2 样片）：Nikon Z8 高效 NEF（AV1 压缩 RAW）无法解码

- **现象**：`20260506_SAMPLE-ID/1171_5219`（35 张）与 `1172_5207`（37 张）全部
  真实解码探针返回 `unsupported`（rawpy `LibRawFileUnsupportedError`）。
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
- **清单标记**：`fixtures/manifest.json`（schema v2）每条含 `decode_status`
  （真实探针枚举：`ok/unsupported/corrupt/missing_dependency/unknown`）与
  `decode_reason`；批处理遇到 `unsupported` 逐张报告失败原因。
  支持矩阵见 `docs/support-matrix.md`：ECN-2 格 = `unsupported`，
  黑白/真实 TIFF/DNG/第二设备 = `missing-fixture`，发布门禁测试保持红色阻断。

## B-2（观察项）：真实翻拍样片的结构差异

- 真实翻拍外圈是黑边（透射率 ~0.0004），片基在黑边内侧。早期"画面外框"启发式
  把黑边当片基，Dmin 置信度归零。已由 `calibration/border.py`（帧修剪 + 边缘带
  对比选择）修复，合成测试 `tests/unit/test_border_detect.py` 锁定。
- 帧内仍可能包含带字样的不透明片边（"KODAK 500"），其超暗像素曾把逐通道 Dmax
  估计推到 2.96（真值 ~1.9）。已改为亮度域分位统计修复。

## B-3（未决，发布前必须处理）：项目许可证

- 见 `docs/dependency-licenses.md` 结论：无技术阻塞，但 §11 要求用户本人选择
  GPL-3.0 / MIT / Apache / 闭源免费。当前按"闭源可行"最严口径准备。

## B-4（未决，打包发布前必须处理）：可写状态根依赖「仓库布局」

- **现象**：`storage.REPO_ROOT = Path(__file__).resolve().parents[2]`
  （2026-09-08「仓库根解析统一」把 11 处 `parents[3]` 收进这一处，方向是对的），
  标注目录与 GUI 草稿目录都挂在它下面：`REPO_ROOT/fixtures/annotations`、
  `REPO_ROOT/outputs/drafts`。
- **只在仓库布局下成立**：
  - 仓库内开发（当前全部测试与冒烟）：正确，且被 conftest 的写入物理隔离守着；
  - `pip install .` 布局：`__file__` 落在 `site-packages/negative_lab/`，
    `parents[2]` 算到 `<venv>/Lib`（Windows）一类**无意义目录**；
  - PyInstaller onedir（`packaging/windows/negative-lab.spec`）：`_MEIPASS`
    即 `<安装目录>/_internal`，`parents[2]` 算到**安装目录本身**——
    装进 `C:\Program Files` 时标准用户不可写，用户确认标注会写失败。
- **本轮为什么不动**：改道（例如退回 `%APPDATA%\negative-lab`）本身就是一次
  「位置由环境决定」的静默变更，而本仓库刚把「环境变量改道被无声吞掉」立为
  反面教材（交接 §2）；`profiles/profile.py` 已经在用 APPDATA，
  两处根要不要统一、老用户已有标注怎么迁移，都是**产品决定**，
  不该由一轮代码审查顺手拍板。
- **建议方案（确认后再实施）**：可写根解析改为
  **显式改道 > 环境变量 > 用户数据目录（APPDATA/XDG）> 仓库目录（仅当可写且
  像仓库）**，并在右栏技术信息里显示生效路径——用户看得见才不会漂；
  同步给 `app.selftest` 补一条「可写目录自检」（写一次读一次）。
