"""回归：**Gold 200 的通道指数必须先验在留出卷上复验**，不许只凭平均值。

`fixtures/priors/KodakGold200-Z8-userfit.profile.json` 里 `blue_ratio=0.7`
是**在 11 张用户手工成品上扫出来的**（`tmp_ratio_cast.py`）：中间调 (a,b)
色度向量差中位 0.0492 → 0.0407，蓝偏 Δb 从 −0.0288 收到 −0.0092，
饱和偏差基本不动（−0.012 → −0.018）。**扫出来的东西必须留出复验**，
所以本文件只跑 `1299_G200`（manifest `split=holdout`，三帧有参考），
并且要求**逐帧**都变好——平均值变好而逐帧变差，是这个项目反复犯过的错
（B-11 追加⑨ 的黑点、B-13 追加① 的色相角都是这样暴露的）。

同时钉住方向：2026-08-31 学到的旧先验 `red 1.1702 / blue 1.0556` 的 blue 方向
**与真值相反**，只把向量差做到 0.0482 却把饱和压到 −0.047。

⚠ **本文件旧版此处还写着"`red_ratio` 那条路实测无效"，那句话 2026-10-11 作废**：
它是在若干帧片基错 0.19~0.36 decade 的条件下、且只看蓝黄轴 `Δb` 与饱和得出的，
而红斜率的误差在 `R−G` 轴上。修掉片基后重测（`docs/blockers.md` B-15⑧）：
`red_ratio` 1.0 → 1.30 让留出卷逐帧 abs(Δ带宽RG) 从 0.0377 降到 0.0197（三帧全变好）。
锁在 `tests/regression/test_red_slope_gain_pair.py`——**红斜率与色度增益是一对**，
分开评会得出相反结论（上一轮就是这么错的）。
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
from PIL import Image

from negative_lab.calibration.annotations import load_annotation
from negative_lab.io.images import load_linear_rgb
from negative_lab.library.fingerprint import capture_fingerprint
from negative_lab.library.models import FrameRecord, GeometryRegions
from negative_lab.pipeline.density import encode_srgb
from negative_lab.pipeline.model_assets import ModelAsset, canonical_hash
from negative_lab.pipeline.service import process_roll

ROOT = pathlib.Path(__file__).resolve().parents[2]

HOLDOUT_ROLL = "1299_G200"          # manifest split=holdout
MAX_DIM = 1200
BANDS = ((0.30, 0.45), (0.45, 0.60), (0.60, 0.75))
PROFILE = ROOT / "fixtures" / "priors" / "KodakGold200-Z8-userfit.profile.json"

_map = json.loads((ROOT / "fixtures" / "references" / "g200_manual_look.json")
                  .read_text(encoding="utf-8"))
REF = {e["frame_id"]: e for e in _map["frames"]
       if not e["variant_of"] and e["frame_id"].startswith(HOLDOUT_ROLL)}
MET = _map["method"]


@pytest.fixture(scope="module")
def inputs():
    missing = [f["reference_file"] for f in REF.values()
               if not (ROOT / f["reference_file"]).exists()]
    if missing:
        pytest.skip(f"用户手工参考大图不在库里（gitignore），缺 {len(missing)} 张，"
                    f"例如 {missing[0]}")
    man = json.loads((ROOT / "fixtures" / "acceptance" / "manifest.json")
                     .read_text(encoding="utf-8"))
    frs = sorted([f for f in man["frames"] if f.get("required")
                  and f["roll"] == HOLDOUT_ROLL], key=lambda f: str(f["id"]))

    def scale(roi, sy, sx):
        out = (int(round(roi[0] * sy)), int(round(roi[1] * sx)),
               int(round(roi[2] * sy)), int(round(roi[3] * sx)))
        return out if (out[2] - out[0] >= 4 and out[3] - out[1] >= 4) else None

    recs, imgs, kept = [], [], []
    for fr in frs:
        loaded = load_linear_rgb(ROOT / fr["file"])
        arr = loaded.rgb
        step = max(1, int(np.ceil(max(arr.shape[:2]) / float(MAX_DIM))))
        img = arr[::step, ::step, :].copy()
        ann = load_annotation(fr["file"])
        sy, sx = img.shape[0] / arr.shape[0], img.shape[1] / arr.shape[1]
        geo = GeometryRegions(
            rotation_degrees=ann.rotation_degrees,
            output_crop_roi=scale(ann.output_crop_roi or ann.image_area_roi, sy, sx),
            image_area_roi=scale(ann.image_area_roi or ann.output_crop_roi, sy, sx),
            base_sample_rois=[r for r in (scale(x, sy, sx)
                                          for x in (ann.base_sample_rois or [])) if r])
        recs.append(FrameRecord(source_path=fr["file"],
                                source_sha256=fr["source_sha256"],
                                input_facts=dict(loaded.meta),
                                capture_fingerprint=capture_fingerprint(
                                    loaded.meta, source_path=fr["file"]).value,
                                geometry=geo))
        imgs.append(img)
        kept.append(fr)
    return list(zip(recs, imgs)), kept


def _asset(rr: float, rb: float) -> ModelAsset:
    payload = {"density_matrix": np.eye(3).tolist(),
               "channel_curves": [None, None, None],
               "red_ratio": rr, "blue_ratio": rb,
               "reference_exponent": None, "dmax": None}
    return ModelAsset(asset_version="regression-ratio", payload=payload,
                      sha256=canonical_hash(payload), enabled=True)


def _oklab_ab(rgb01):
    c = np.asarray(rgb01, np.float64)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    l = 0.4122214708 * lin[..., 0] + 0.5363325363 * lin[..., 1] + 0.0514459929 * lin[..., 2]
    m = 0.2119034982 * lin[..., 0] + 0.6806995451 * lin[..., 1] + 0.1073969566 * lin[..., 2]
    s = 0.0883024619 * lin[..., 0] + 0.2817188376 * lin[..., 1] + 0.6299787005 * lin[..., 2]
    l, m, s = (np.cbrt(np.maximum(l, 0)), np.cbrt(np.maximum(m, 0)),
               np.cbrt(np.maximum(s, 0)))
    return np.stack([1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s], -1)


def _mid_chroma(disp):
    im = Image.fromarray((np.clip(disp, 0, 1) * 255).round().astype(np.uint8))
    d = np.asarray(im.resize((MET["resize"],) * 2, Image.BILINEAR), np.float32) / 255.0
    c = int(d.shape[0] * (1 - MET["center_crop_frac"]) / 2)
    r = int(d.shape[1] * (1 - MET["center_crop_frac"]) / 2)
    d = d[c:d.shape[0] - c, r:d.shape[1] - r, :]
    luma = 0.2126 * d[..., 0] + 0.7152 * d[..., 1] + 0.0722 * d[..., 2]
    ab = _oklab_ab(d)
    out = []
    for lo, hi in BANDS:
        m = (luma >= lo) & (luma < hi)
        if m.sum() >= 400:
            out.append((float(ab[..., 0][m].mean()), float(ab[..., 1][m].mean())))
    return out


def _ref_chroma(path):
    d = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    s = MAX_DIM / max(d.shape[:2])
    d = np.asarray(Image.fromarray((d * 255).round().astype(np.uint8))
                   .resize((int(d.shape[1] * s), int(d.shape[0] * s))),
                   np.float32) / 255.0
    return _mid_chroma(d)


def _gaps(inputs, rr, rb, gain=1.897):
    """逐帧量中间带色度向量差与 Δb。

    **增益在这里被钉成 1.897**，因为本文件的全部结论（0.7 比 1.0 好、反向更糟）
    都是在 red=1.0 × gain=1.897 下量出来的；2026-10-11 上线的
    red 1.30 × gain 2.50 会改变这些数（留出 `G200-2` 在 2.50 增益下
    从"逐帧都变好"变成"基本持平"），那是**那一对的结论**，
    由 `test_red_slope_gain_pair.py` 去钉，不是把本文件的判据放宽。
    """
    from negative_lab.library.render import RenderIntent, apply_render

    frames, kept = inputs
    _, results = process_roll(frames, model_asset=_asset(rr, rb))
    out = {}
    for fr, pr in zip(kept, results):
        if fr["id"] not in REF:
            continue
        name = pr.chosen_candidate or pr.candidates.default_name
        it = RenderIntent.from_dict(dict(pr.candidates.get(name).intent or {}))
        it.print_chroma_gain = float(gain)
        ours = _mid_chroma(np.clip(encode_srgb(apply_render(
            pr.result.positive_linear, it))[..., :3], 0, 1))
        his = _ref_chroma(ROOT / REF[fr["id"]]["reference_file"])
        pairs = [(a, b) for a, b in zip(ours, his) if a and b]
        out[fr["id"]] = (float(np.median([np.hypot(x[0] - y[0], x[1] - y[1])
                                          for x, y in pairs])),
                         float(np.mean([x[1] - y[1] for x, y in pairs])))
    return out


def test_userfit_profile_exists_with_the_measured_exponent():
    prof = json.loads(PROFILE.read_text(encoding="utf-8"))
    assert prof["exponent_prior"]["blue_ratio"] == pytest.approx(0.7)
    assert prof["film_type"] == "c41_color"


def test_blue_exponent_improves_every_holdout_frame(inputs, monkeypatch):
    """**逐帧**都要变好：留出卷三帧，任何一帧变差都算这条锁红。

    ⚠ **本锁把逐帧电平项钉回 `strength=0`（B-14⑮）再量**：这条锁量的是
    `red 1.00 × blue 1.00 vs 0.70` 这个**隔离角**上的色相向量差，
    而它的"三帧全部变好"是在没有电平项的工作点上量出来的。
    电平 δ 一上线（逐帧 −0.054~+0.058 decade），`1299::G200-2` 在这一角上
    从 0.0141 变到 0.0230 —— **变的是带内像素集合，不是蓝指数**
    （"中间带"是 `(0.30≤lum<0.72)` 的内容统计量）。
    ⇒ 按"常数变了就重钉条件、不放宽判据"的规矩，这里把电平关掉，
    本锁继续守它原本的条件；**蓝指数在产品工作点
    （`red 1.30 × gain 2.50`、电平开着）上的逐帧账**由
    `test_red_slope_gain_pair.py::test_blue_index_improves_every_holdout_frame_at_the_product_point`
    负责，那条锁不许跟着关掉电平。
    """
    monkeypatch.setattr("negative_lab.print_math.PRINT_LEVEL_STRENGTH", 0.0)
    #: ㉓ 之后电平有**两个**携带者（密度中位 + 高调面积），只钉第一个不等于回到 δ=0。
    monkeypatch.setattr("negative_lab.print_math.PRINT_LEVEL_AREA_STRENGTH", 0.0)
    base = _gaps(inputs, 1.0, 1.0)
    fit = _gaps(inputs, 1.0, 0.7)
    print()
    for fid in sorted(base):
        print(f"  {fid.split('::')[-1]:<10} 向量差 {base[fid][0]:.4f} → {fit[fid][0]:.4f}"
              f"   Δb {base[fid][1]:+.4f} → {fit[fid][1]:+.4f}")
    worse = [f for f in base if fit[f][0] >= base[f][0]]
    assert not worse, f"留出卷上这些帧没有变好：{worse}"
    assert np.median([v[0] for v in fit.values()]) \
        < np.median([v[0] for v in base.values()])
    # 蓝偏必须**朝零**收（符号不变但幅度变小，或跨过零但幅度变小）
    assert np.mean([abs(v[1]) for v in fit.values()]) \
        < np.mean([abs(v[1]) for v in base.values()])


def test_identity_is_the_negative_control_for_the_direction(inputs):
    """**负控制**：把 blue_ratio 往**反方向**推（>1，即旧学习先验的方向）
    必须让蓝偏变大——否则上面那条"变好"可能只是任何改动都带来的扰动。"""
    base = _gaps(inputs, 1.0, 1.0)
    wrong = _gaps(inputs, 1.0, 1.0556)
    b0 = np.mean([abs(v[1]) for v in base.values()])
    bw = np.mean([abs(v[1]) for v in wrong.values()])
    print(f"\n  反向 blue_ratio=1.0556：|Δb| {b0:.4f} → {bw:.4f}"
          f"（旧先验方向，必须更糟或至少不更好）")
    assert bw >= b0 - 1e-4, (
        f"反向改动把蓝偏也压小了（{b0:.4f} → {bw:.4f}）："
        "说明该量对任意扰动都反应，正锁没有区分力")
