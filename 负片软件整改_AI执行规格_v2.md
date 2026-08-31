# negative-lab 整改 AI 执行规格 v2

> 本文件是 2026-08-31 复核后的可执行修订版。它补充并收紧
> 负片去色罩软件_AI执行规格.md，不得降低原规格的输入、色彩、质量、
> 样片、事务导出和发布要求。开始前必须阅读原规格与
> 质量审查报告_20260831_复核.md。

## 0. 执行目标和当前状态

目标是把当前“能计算和写文件”的原型整改成：

1. GUI、CLI、批处理调用同一条可追溯处理管线；
2. RAW/TIFF 的色彩空间和传递函数语义真实、可重现；
3. 默认方向和裁切正确，不能把片边当照片导出；
4. 质量失败不会被伪装成成功，三件套导出具备事务性；
5. 测试可重复运行，真实视觉质量和样片覆盖达到发布门槛。

当前已知事实：

- 最新代码提交：dffecf9。
- pytest：118 passed、1 failed；失败是缺黑白、ECN-2 可解码/TIFF、真实 TIFF/DNG 和第二设备的发布门禁。
- compileall、uv pip check 通过。
- CLI 冒烟、GUI 冒烟和 verify_package.py 不能在当前工作区稳定通过。
- GUI G200-1 仍导出片边并偏淡；联系表只有 1 张 verified 样片且等待用户确认。
- dist 早于最新代码，不能作为本轮交付物。

**停止规则：** 任意 P0 门禁失败，停止后续阶段；保留命令、日志、输出路径和失败原因。禁止降低阈值、删除测试、把 fail 改成 warn 或用 NLP 结果冒充真值。

## 1. 不可违反的约束

1. 原始 RAW、TIFF、DNG、NLP 图像只读；不得覆盖、移动、重压缩或提交 Git。
2. Negative Lab Pro 只作风格对照，不是真值；不得用“更像 NLP”替代色卡/中性区、剪裁、方向、裁切和人工视觉验收。
3. 不引入神经网络或云端处理来掩盖标定不足；第一版保持可解释的密度域、色彩管理、胶片/设备先验和统计标定。
4. 不得为 G200-1 写死单张万能参数。参数只能来自标定集，留出集按卷/设备隔离验收。
5. 不得用内嵌 JPEG 冒充不支持的 RAW；AV1 NEF 必须明确 unsupported，或由用户提供可解码的 16 位 TIFF/无损 NEF。
6. 公式、工作空间语义或 sidecar 字段改变时提升 schema_version，不得静默复用旧语义。
7. P0 全部通过前禁止重新打包、生成发布 ZIP、写“已完成”或交给朋友试用。
8. 新增大图只写 outputs/ 或临时目录；原始样片不进源码和 Git。

## 2. P0-D：统一 GUI、CLI、批处理主路径（第一优先级）

### 2.1 建立单一服务入口

在 src/negative_lab/pipeline/ 增加一个窄接口（可复用 workflow.py，不要复制三份逻辑），建议返回：

~~~text
ProcessResult:
  loaded_meta
  prepared: t_work + GeometryRecord
  auto_report
  invert_result
  quality_gate
  sidecar_payload
~~~

固定顺序：

~~~text
load/linearize
 -> source 中读取标注和用户 ROI
 -> rotate_cw(source, rotation)
 -> 全图检测 frame/base/image_area
 -> 用裁切前的 source/oriented 片基 ROI 求 Dmin
 -> crop(oriented)
 -> 透射率域除尘（可回退）
 -> sanitize NaN/Inf
 -> auto_analyze / prior
 -> invert_chunked
 -> QualityGate
 -> 预览或事务导出
~~~

入口禁止直接调用 auto_analyze、invert 或自行实现旋转/裁切，除非是在上述服务内部。

### 2.2 GUI 必须改造

修改 src/negative_lab/ui/main_window.py：

- 删除/替换 _rot() 中直接 np.rot90 的入口逻辑，统一使用 geometry.rotate_cw；90° 必须是顺时针。
- load() 时重置 rotation、旋转下拉框、frame/base ROI、亮暗中性点、除尘蒙版、参数、confidence、quality 和旧报告；不能把上一张图的状态带到下一张。
- 旋转时变换所有 source 坐标对象；无法可靠变换时清空并重新检测。不能只替换数组。
- 打开即显示 image_area 候选，不要把 detect_regions()["frame"] 当最终照片区域。
- run_auto() 调用统一服务，使用当前 rotation、ROI、配置先验和中性点；服务返回的参数同时驱动预览和导出。
- _refresh() 显示 QualityGate warnings/failures；除尘函数返回的 warnings 不得丢弃。
- export_all() 不得重新计算自己的 crop，也不得把 preview 坐标写入 sidecar；使用服务保存的 geometry/quality/diagnostics。
- QualityGate 为 fail 时禁止导出；为 warn 时允许导出但必须在界面、日志、旁车明确显示通道、方向和比例。
- 单张“加载配置文件”按钮不得永久禁用：加载、版本校验、当前配置显示、重置和损坏错误都要完成。

### 2.3 CLI 必须改造

修改 src/negative_lab/cli.py：

- invert 使用同一服务入口；支持 --rotation 0|90|180|270、--frame-roi y0 x0 y1 x1、--base-roi、--keep-rebate、--params 和 --tiff-encoding。
- 默认不能把全图当照片区；检测失败时显式 warning，并要求 --keep-rebate 才允许保留片边。
- 统一输出 QualityGate、geometry、warnings、source hash、输入/输出色彩语义和 schema v2 sidecar。
- batch 与 invert 使用 ASCII 状态词（OK/WARN/FAILED/CANCELLED），默认 GBK 控制台不能崩溃。

### 2.4 批处理必须改造

修改 src/negative_lab/pipeline/batch.py 与 ui/batch_panel.py：

- 保留 prepare_and_analyze 的共享实现，但通过同一服务完成除尘、反相、质量和导出。
- QualityGate.failures 在写文件前映射为 failed，不生成任何输出；剪裁超限属于可见 warn，不得伪造为 ok。
- BatchJob.run() 成功收尾时必须写 batch-report.json 和文本摘要；GUI 完成标签只能显示实际存在的报告。
- UI 行使用规范化完整路径或稳定 item id，不得按文件名匹配同名文件。
- 失败项可独立重试；取消在文件边界生效，且 staging 和本次输出全部清理。

## 3. P0-E：修正 RAW 色彩空间语义

修改 src/negative_lab/color/management.py、docs/color-pipeline.md、测试和 sidecar：

1. 禁止使用 output_color: 6 这种裸整数和“6=LINEAR”注释。运行时使用 rawpy.ColorSpace.<member> 命名枚举，并记录 name 与 value。
2. 先通过当前 rawpy API 和最小样片验证实际含义：rawpy.ColorSpace.raw=0、sRGB=1、ACES=6 等。决定工作空间到底是 camera-raw、ACES-linear 或另一个明确空间。
3. 若保留 ACES，必须把工作空间和矩阵语义改为 ACES-linear，不能继续写 camera-linear-normalized；若要 camera RGB，选择真实对应的 rawpy 输出并验证通道/矩阵含义。
4. gamm=(1,1,1,1)、no_auto_bright=True、highlight=0 等参数仍需回读校验；未知属性不得静默通过。
5. 增加测试：命名枚举与运行值一致、灰阶线性 ramp 不被意外 gamma、meta/sidecar 与 ADR 一致、同一输入重复解码结果稳定。
6. 更新所有回归基线和文档后再跑完整测试；不能只改测试断言让旧语义继续绿灯。

## 4. P0-F：sidecar v2 和几何往返

sidecar 至少包含：

~~~json
{
  "schema_version": 2,
  "source_sha256": "...",
  "source_size": [0, 0],
  "rotation_degrees": 0,
  "geometry": {
    "crop_oriented": [0, 0, 0, 0],
    "output_size": [0, 0],
    "keep_rebate": false,
    "roi_source_space": {
      "frame": [0, 0, 0, 0],
      "film_base": [0, 0, 0, 0]
    }
  },
  "input": {"encoding": "...", "profile_sha256": "...", "raw_decode": {}},
  "working_space": "明确的实际工作空间",
  "output": {"encoding": "srgb", "icc_sha256": "..."},
  "parameters": {},
  "quality": {},
  "warnings": [],
  "software_version": "..."
}
~~~

要求：

- 所有 ROI 对外以 SOURCE 坐标声明；crop_oriented 单独声明 ORIENTED 坐标。
- coordinate_space: "preview" 不得再出现。
- 由 sidecar 重建旋转、裁切和输出尺寸，往返结果必须与原导出一致。
- 旧 schema 读取要显式迁移或拒绝，不能静默改变含义。

## 5. P0-G：质量门禁和视觉验收

### 5.1 机器门禁

扩展 QualityGate：

- 接收有效画面 mask/非蒙版 mask；剪裁统计不能把片边或手动蒙版计入正常照片指标。
- 精确计算输入/输出逐通道 low/high clip、NaN/Inf、Dmin/Dmax 置信度和 warnings。
- 有灰卡/色卡或用户确认中性区时计算 DeltaE2000；中性区目标 <=5，优秀目标 <=3。
- 没有中性参考时输出低置信度 warning，不能把全 1.0 通道参数当正式成功。
- fail 在导出前阻断；warn 可导出但必须显示并写入旁车。

### 5.2 真实视觉门禁

重新生成联系表，至少覆盖：

- 1298 G200 标定集和 1299 G200 按卷隔离留出集；
- 每张方向、裁切、黑位、高光、阴影、整体色偏、肤色/中性灰均由用户或明确验收人签字；
- 不以 NLP 全图 DeltaE 作真值；对齐后只在中性区比较；
- 视觉结果仍偏淡、含片边或方向不对时，测试必须失败，禁止扩大 p05/p50 容差来消绿。

### 5.3 样片发布覆盖

保持 test_release_gate_blocked_until_fixtures 红灯，直到获得并探针验证：

- 至少 3 种 C-41、2 种黑白、2 种 ECN-2；
- 至少 2 套相机/扫描仪/光源组合；
- 每种组合至少 5 张，含正常、偏暗、偏亮、强色场景和无中性物体；
- 真实 16 位 TIFF、DNG、可解码 ECN-2 输入；
- 不足时支持矩阵必须写 missing-fixture/unsupported，不能宣称支持。

## 6. P0-H：事务性三件套导出

新增统一 export_triplet()（可放 src/negative_lab/io/atomic.py 或 io/export.py）：

1. 为 TIFF、JPEG、sidecar 创建同目录唯一 staging 名称。
2. 写入并重读验证像素位深、ICC hash、schema 和 source hash。
3. 三件都通过后再提交到最终文件名；提交顺序和回滚策略必须明确。
4. 任一写入、校验或磁盘异常时删除 staging 和本次已提交文件，不留下半套。
5. 不覆盖 source；目标已存在时使用唯一 run id 或明确的“不覆盖”错误。

新增失败注入测试：TIFF 失败、JPEG 失败、sidecar 失败、ICC 缺失、磁盘写失败、取消中断；每项都断言 source 未变且无半成品。

## 7. P0-I：验证脚本和运行环境

修改 scripts/e2e_cli_smoke.py、scripts/gui_smoke.py、scripts/verify_package.py：

- 使用 tempfile.TemporaryDirectory() 或唯一 run 目录；失败时打印目录并保留证据。
- 断言本次运行写出的 manifest，不统计目录历史文件。
- 连续运行两次必须通过，且重复运行不覆盖 source。
- 默认 stdout/stderr 使用 ASCII 状态文本；中文内容在 UTF-8 文件中记录，不让 GBK 控制台崩溃。
- GUI 冒烟必须断言 rotation、geometry、quality、ICC 和三件套，而不是只数文件。
- verify_package.py 不得在验证阶段偷偷重建过期 ZIP；构建和验证分离，并核对构建 commit、文件树和 SHA-256。

## 8. 实施顺序、命令和交付证据

严格按顺序执行：

1. P0-D 统一 GUI/CLI/批处理服务和坐标状态。
2. P0-E 修正 RAW 枚举/工作空间语义并更新 ADR/测试。
3. P0-F sidecar v2 与几何往返。
4. P0-G QualityGate、视觉联系表和真实样片门禁。
5. P0-H 事务导出与失败注入。
6. P0-I 幂等冒烟、GBK、冻结运行时验证。
7. 只有以上全部通过，才进入安装包、干净 Windows 安装/启动/处理/卸载验证。

每个阶段完成后运行并记录：

~~~powershell
.\\.venv\\Scripts\\python.exe -m pytest -q <本阶段测试>
.\\.venv\\Scripts\\python.exe -m compileall -q src scripts tests
uv pip check --python .\\.venv\\Scripts\\python.exe
.\\.venv\\Scripts\\python.exe scripts\\e2e_cli_smoke.py
.\\.venv\\Scripts\\python.exe scripts\\gui_smoke.py
~~~

完整 P0 交付证据必须包含：

- 实际命令、退出码、耗时和失败日志；
- 样片 manifest、哈希、解码状态和标注版本；
- 至少一张用户确认的联系表和中性区 DeltaE 记录；
- CLI、GUI、批处理使用同一 sidecar v2 的示例；
- 事务失败注入结果；
- 两次连续冒烟结果；
- 仅在许可证由用户选择后，才记录安装包、卸载结果和 SHA-256。

## 9. 禁止的“修复”

- 不得把 QualityGate.fail 改成 warn 来保持输出。
- 不得删除发布门禁失败测试或把缺样片标成 verified。
- 不得把 GUI 继续留在旧管线而只修批处理测试。
- 不得用固定 G200-1 参数、NLP 全图距离或单一宽松亮度阈值证明通用颜色正确。
- 不得把便携 onedir ZIP 写成“安装包”；未完成安装/卸载验证前只能称内部验证构建。

## 10. 当前停止点

本文件生效时，项目停在 P0-C/P0-D 之间：算法底层和批处理骨架可继续复用，但 GUI、CLI、RAW 色彩语义、事务导出、视觉验收和发布门禁均未完成。下一位 AI 必须从第 2 节开始实现，完成一项就更新 docs/remediation-progress.md，并在任何红灯处停止，不得宣称整改完成。
