"""合成负片端到端报告（Phase 1 交付：合成测试报告）。

生成多组合成负片 -> 自动分析 -> 反相 -> 指标汇总，写入 docs/synthetic-report.md。
运行：.venv\\Scripts\\python.exe scripts/run_synthetic_report.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from negative_lab.pipeline.auto import auto_analyze  # noqa: E402
from negative_lab.pipeline.density import invert  # noqa: E402
from negative_lab.testing.synthetic import make_synthetic_negative  # noqa: E402


def cast_at(img, point, half=4):
    """9x9 均值通道的残余色偏（更大取样以平均掉采样噪声）。"""
    r, c = point
    patch = img[r - half:r + half + 1, c - half:c + half + 1, :].reshape(-1, 3)
    m = patch.mean(axis=0)
    return float((m.max() - m.min()) / max(m.mean(), 1e-6))


def run_case(name, **kw):
    s = make_synthetic_negative(**kw)
    neg, truth = s["negative"], s["truth"]
    t0 = time.perf_counter()
    rep = auto_analyze(neg, base_roi=truth["base_roi"], film_roi=truth["film_roi"],
                       neutral_bright_points=[truth["neutral_bright"]],
                       neutral_dark_points=[truth["neutral_dark"]])
    res = invert(neg, rep.params)
    dt = time.perf_counter() - t0
    dmin_err = float(np.max(np.abs(np.array(rep.params.dmin) - np.array(truth["dmin"]))
                            / np.array(truth["dmin"])))
    dmax_err = abs(rep.params.dmax - truth["dmax"]) / truth["dmax"]
    out = res.positive_linear
    lum = out.mean(axis=2)
    a, b = truth["ramp_cols"]
    cols = np.arange(a + 4, b - 4, 8)
    prof = lum[:, cols].mean(axis=0)
    monotonic = bool((np.diff(prof) > -1e-4).all())
    return {
        "name": name,
        "dmin_err": dmin_err,
        "dmax_err": dmax_err,
        "cast_bright": cast_at(out, truth["neutral_bright"]),
        "cast_dark": cast_at(out, truth["neutral_dark"]),
        "conf": rep.confidence,
        "monotonic": monotonic,
        "finite": bool(np.isfinite(out).all()),
        "n_warn": len(set(rep.warnings)),
        "seconds": dt,
    }


def main() -> int:
    cases = [
        run_case("标准 C-41 橙片基 k=1", seed=101),
        run_case("低感光度薄乳剂 dmax=1.4", seed=102, dmax_true=1.4),
        run_case("高对比 dmax=2.8", seed=103, dmax_true=2.8),
        run_case("通道差异 k=(1.15,1,0.85)", seed=104, exponents=(1.15, 1.0, 0.85)),
        run_case("重尘点 dust=400", seed=105, dust=400),
        run_case("高噪声 0.004", seed=106, noise=0.004),
    ]
    lines = [
        "# 合成负片管线测试报告（Phase 1）",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        "模型：密度域主模型 + 每通道幂指数（规格 §4.2–§4.4）。",
        "指标：Dmin/Dmax 相对误差、中性点残余色偏 (max-min)/mean、斜坡单调性、置信度。",
        "",
        "| 用例 | Dmin 误差 | Dmax 误差 | 亮中性色偏 | 暗中性色偏 | 单调 | 有限 | 警告数 | Dmin 置信 | 指数置信 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    ok = True
    for c in cases:
        good = (c["dmin_err"] < 0.05 and c["dmax_err"] < 0.15 and c["cast_bright"] < 0.02
                and c["monotonic"] and c["finite"])
        ok = ok and good
        lines.append(
            f"| {c['name']} | {c['dmin_err']:.3%} | {c['dmax_err']:.2%} | "
            f"{c['cast_bright']:.3%} | {c['cast_dark']:.3%} | {'OK' if c['monotonic'] else 'FAIL'} | "
            f"{'OK' if c['finite'] else 'FAIL'} | {c['n_warn']} | {c['conf'].get('dmin', 0)} | "
            f"{c['conf'].get('exponent', 0)} |")
    lines += [
        "",
        "## 结论",
        "",
        f"- 全部用例满足门禁（Dmin<5%、Dmax<15%、中性色偏<2%、单调、无 NaN/Inf）：**{'通过' if ok else '未通过'}**",
        "- 符号方向由 `tests/synthetic/test_direction.py` 锁定；参数单调性由 `test_monotonicity.py` 锁定。",
        "- 本报告仅覆盖合成数据；真实 NEF/TIFF 输入属 Phase 2/3 门禁（RAW 解码依赖未验证）。",
    ]
    out_path = Path(__file__).resolve().parents[1] / "docs" / "synthetic-report.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"报告已写入 {out_path}")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
