# 负片裁切与去色罩 (negative-lab)

离线、无订阅、**无需色卡**的胶片负片转换工具：导入混合文件夹 → 手动整理成卷 →
可靠裁切 → 自动得到三个可用颜色起点（保留氛围 / 中性 / 偏暖）→ 少量微调 → 整卷导出。

## ⚠ 状态的唯一权威来源（**防读到过期数字**）

- **权威 = git**：`docs/blockers.md`（B 系列阻塞与订正链）、`CHANGELOG.md`、
  以及**带锁的测试**；数字与状态**一律以当场实测为准**。
- **知识库/记忆页可能严重滞后**（2026-09 实测：某页至今仍在说
  **"537 项测试通过"**，而本仓库当时实测已 **561 passed**；
  且新写入的文档**连续 6 次检索不到**）⇒
  **不要**把记忆页里的测试数、"未解决问题"清单当现状；**先跑再说**。
- 引用"**UI 测试通过**"时**必须带边界**：**不含文字渲染正确性**
  （实测 **7 处**依赖文字布局：5× `setWordWrap(True)`、1× `font-size`、1× `setWordWrap(False)`，
  且全仓**无** `elidedText`；详见 `docs/visual-acceptance.md` 的覆盖边界小节）。

## 生效规格

> **产品范围基线：`负片裁切与去色罩_AI执行规格_v5.md`（仍然生效）。**
> **当前执行基线：`负片裁切与去色罩_AI执行规格_v6.2_交互与标定闭环修复.md`。**
>
> 分层规则（2026-09-08 起写明，此前 README 只声称"v5 是唯一执行基线"，
> 与实际已实现的 v6 / v6.1 / v6.2 脱节了三代）：
>
> - **v5 决定"做什么、不做什么"**：无需色卡、没有 verified 标定不阻断预览导出、
>   黑白/反转不在 P0、三类 ROI 独立、普通用户不暴露技术控件；
> - **v6.x 决定"交互与状态怎么工作"**：v6 单工作区 Stage 模型 →
>   v6.1 质量修复 → v6.2 交互与标定闭环（草稿 vs 确认基线、中性点救援、
>   导出计数完整性、测试写入物理隔离）；
> - v6.x 是在 v5 之上叠加，**不得**把 v5 的产品范围改回去；两者冲突时
>   产品范围以 v5 为准，交互与状态以 v6.2 为准。
>
> 历史资料（已降级）：`负片去色罩软件_AI执行规格.md`、`负片软件整改_AI执行规格.md`、
> `负片软件整改_AI执行规格_v2.md`、`负片去色罩算法重构_AI执行规格_v3.md`、
> `负片去色罩算法重构_AI执行规格_v4.md`。

v5 覆盖旧规格的具体内容（P0-0 冻结声明）：

1. **普通转换不需要色卡。** 程序不得再次要求用户为每种胶片/相机/扫描仪拍摄色卡。
2. **没有色卡、未知胶片型号、没有 verified 颜色标定，都不阻断预览与导出。**
   `needs_calibration` 不再是产品状态；研发用 calibration API 不进入普通路径。
3. **黑白负片与反转片不在 P0**（v5 §0.2.5）；旧发布门禁中"缺黑白样片即阻断"失效。
4. **首版同时支持**相机 RAW 翻拍与扫描仪 16-bit TIFF / DNG 输入。
5. **三类 ROI 独立**：最终输出裁切、有效画面分析区、未曝光片基取样区
   （可多个）不再混为一个 `FRAME`。
6. 主工作区只有 **裁切 / 转换** 两个一级页面；Dmin/Dmax/矩阵/指数/校准/除尘
   等技术控件不再暴露给普通用户。

## 开发环境

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e . pytest
.venv\Scripts\python.exe -m pytest
```

## 命令行

```powershell
# 自动分析 + 反相，导出 16 位 TIFF / 8 位 JPEG / sidecar
.venv\Scripts\python.exe -m negative_lab.cli invert input.tif -o out_dir\
```

## 目录

- `src/negative_lab/pipeline`  统一服务、密度技术底座、卷级分析、三候选、微调
- `src/negative_lab/library`   导入会话 / 卷分组 / 帧记录 / 几何 / 采集指纹
- `src/negative_lab/calibration`  片基与区域检测（研发用 calibration 亦在此）
- `src/negative_lab/diagnostics`  剪裁、NaN/Inf、质量门禁与报告
- `src/negative_lab/io`  读写与事务导出（原始文件永远只读）
- `src/negative_lab/ui`  单工作区主窗口（MainWindow + frame_loader /
  geometry_bridge / intent_controller / export_coordinator / state_store
  五个拆分模块，2026-09-08 结构整改）、预览画布、右栏面板、后台线程
- `src/negative_lab/testing`  合成负片生成器
- `tests/`  unit / synthetic / regression（unit 含分层门禁与回归锁）
- `docs/`  色彩管线 ADR、算法数学契约、支持矩阵、进度与验收记录

## 真实样片

`20260506_SAMPLE-ID/`、`20260709_SAMPLE-ID(1)/` 为用户原始底片翻拍 NEF 与
16 位 TIFF（只读、不入 Git）。清单与哈希见 `fixtures/manifest.json`；
逐卷视觉基准见 `fixtures/acceptance/manifest.json`。
