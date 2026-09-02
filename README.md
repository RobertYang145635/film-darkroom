# 负片裁切与去色罩 (negative-lab)

离线、无订阅、**无需色卡**的胶片负片转换工具：导入混合文件夹 → 手动整理成卷 →
可靠裁切 → 自动得到三个可用颜色起点（保留氛围 / 中性 / 偏暖）→ 少量微调 → 整卷导出。

## 生效规格

> **`负片裁切与去色罩_AI执行规格_v5.md` 是唯一执行基线。**
> `负片去色罩软件_AI执行规格.md`、`负片软件整改_AI执行规格.md`、
> `负片软件整改_AI执行规格_v2.md`、`负片去色罩算法重构_AI执行规格_v3.md`、
> `负片去色罩算法重构_AI执行规格_v4.md` 全部降级为历史问题记录与技术资料。
> 凡与 v5 在产品范围、色卡、校准门禁、胶片型号、UI 或验收方法上冲突，一律以 v5 为准。

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
- `src/negative_lab/ui`  裁切页 / 转换页 / 基础微调面板
- `src/negative_lab/testing`  合成负片生成器
- `tests/`  unit / synthetic / regression
- `docs/`  色彩管线 ADR、算法数学契约、支持矩阵、进度与验收记录

## 真实样片

`20260506_SAMPLE-ID/`、`20260709_SAMPLE-ID(1)/` 为用户原始底片翻拍 NEF 与
16 位 TIFF（只读、不入 Git）。清单与哈希见 `fixtures/manifest.json`；
逐卷视觉基准见 `fixtures/acceptance/manifest.json`。
