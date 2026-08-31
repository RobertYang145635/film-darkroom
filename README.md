# 负片去色罩软件 (negative-lab)

密度域 + 每通道幂指数混合模型的负片反相/去色罩工具。
规格文档：`负片去色罩软件_AI执行规格.md`（本仓库需求基线，勿改语义）。

## 开发环境

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e . pytest
.venv\Scripts\python.exe -m pytest
```

## 命令行（Phase 1，无 GUI）

```powershell
# 自动分析 + 反相，导出 16 位 TIFF / 8 位 JPEG / 旁车 JSON
.venv\Scripts\python.exe -m negative_lab.cli invert input.tif -o out_dir\
```

## 目录

- `src/negative_lab/pipeline`  密度域模型、幂指数修正、输出映射
- `src/negative_lab/calibration`  片基 Dmin、Dmax/offset、中性点估计与置信度
- `src/negative_lab/diagnostics`  剪裁、NaN/Inf、色偏诊断
- `src/negative_lab/io`  读写与原子导出（原始文件永远只读）
- `src/negative_lab/testing`  合成负片生成器
- `tests/`  unit / synthetic / regression
- `docs/`  依赖许可证清单、算法笔记

## 真实样片

`20260709_SAMPLE-ID(1)/` 为用户原始底片翻拍 NEF（只读、不入 Git）。
清单与哈希见 `fixtures/manifest.json`。
