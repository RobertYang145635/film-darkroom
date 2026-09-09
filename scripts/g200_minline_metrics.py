"""G200-1 v4 §12.2 最低线实测。"""
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path("/src/film-darkroom")
sys.path.insert(0, str(ROOT / "src"))
from negative_lab.calibration.annotations import load_annotation
from negative_lab.io.images import load_linear_rgb
from negative_lab.pipeline.service import process_image
from negative_lab.profiles.profile import load_profile
from PIL import Image

rel = "20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-1.NEF"
ann = load_annotation(rel)
t = load_linear_rgb(ROOT / rel).rgb
pf = ROOT / "fixtures/priors/KodakGold200-Z8-led.profile.json"
prior = load_profile(pf).to_initial_params() if pf.exists() else None
pr = process_image(t, rotation_degrees=ann.rotation_degrees, ann=ann, prior=prior)
tech = pr.result.positive_linear
vm = pr.valid_mask
reg = tech[vm] if vm is not None else tech.reshape(-1, 3)
treg = pr.t[vm] if vm is not None else pr.t.reshape(-1, 3)

def stats(px):
    lum = px @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    sat = (px.max(1) - px.min(1)) / np.maximum(px.mean(1), 1e-6)
    return {"lum_p50": round(float(np.quantile(lum, .5)), 4),
            "sat_p50": round(float(np.quantile(sat, .5)), 4),
            "sat_p90": round(float(np.quantile(sat, .9)), 4),
            "clip_low": [round(float(v), 4) for v in (px <= 1e-6).mean(0)]}

nlp = np.asarray(Image.open(str(ROOT / "20260709_SAMPLE-ID(1)/1298_G200/翻拍/NLP/G200-1.jpg"
                              )).convert("RGB"), np.float32) / 255
out = {"status": pr.gate.status, "offset": round(pr.report.params.offset, 4),
       "tech": stats(reg), "input_clip_low": stats(treg)["clip_low"],
       "nlp_ref": stats(nlp.reshape(-1, 3)),
       "targets": {
           "clip<1%": max(stats(reg)["clip_low"]) < 0.01,
           "lum_p50_diff<=0.12": abs(stats(reg)["lum_p50"]
                                     - stats(nlp.reshape(-1, 3))["lum_p50"]) <= 0.12,
           "sat_p90>=75%NLP": stats(reg)["sat_p90"]
           >= 0.75 * stats(nlp.reshape(-1, 3))["sat_p90"]}}
print(json.dumps(out, ensure_ascii=False, indent=1))
