"""锁：`fixtures/references/g200_manual_look.json` 必须等于**参考图本身**。

用户手工去色罩成品是这个项目**第一份外观真值**（此前 48 帧判定一条未签，
调研文档 §0.8 原话"没有任何一张结果被确认为好看"）。
它落在 `fixtures/provided/references/`（.gitignore，227MB 不入库），
所以**必须**有一份入库的元数据把"目标是哪 11 张、量出来是多少"钉住，
否则目标会被悄悄换掉而测试仍然全绿。

本锁只在参考图**在场**时量；不在场时 `skip` 并**明说这是降级环境**
（与文字渲染锁同一套规矩：缺东西要喊，不能装成通过）。
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
MAP = ROOT / "fixtures" / "references" / "g200_manual_look.json"


def _doc() -> dict:
    return json.loads(MAP.read_text(encoding="utf-8"))


def _missing() -> list[str]:
    return [e["reference_file"] for e in _doc()["frames"]
            if not (ROOT / e["reference_file"]).is_file()]


def _stats(path: pathlib.Path, m: dict) -> dict:
    """与建表时**完全相同**的方法：显示域 → 中心裁剪 → 固定缩放。"""
    with Image.open(path) as im:
        d = np.asarray(im.convert("RGB").resize(
            (m["resize"], m["resize"]), Image.BILINEAR), dtype=np.float32) / 255.0
    h, w = d.shape[:2]
    ch, cw = int(h * (1 - m["center_crop_frac"]) / 2), \
        int(w * (1 - m["center_crop_frac"]) / 2)
    d = d[ch:h - ch, cw:w - cw, :]
    luma = 0.2126 * d[..., 0] + 0.7152 * d[..., 1] + 0.0722 * d[..., 2]
    sat = d.max(axis=2) - d.min(axis=2)
    return {
        "luma_mean": round(float(luma.mean()), 4),
        "luma_p05": round(float(np.percentile(luma, 5)), 4),
        "luma_p50": round(float(np.percentile(luma, 50)), 4),
        "luma_p95": round(float(np.percentile(luma, 95)), 4),
        "luma_std": round(float(luma.std()), 4),
        "sat_mean": round(float(sat.mean()), 4),
        "sat_p95": round(float(np.percentile(sat, 95)), 4),
    }


def test_mapping_shape_is_self_consistent():
    """入库那份**不依赖大图**：形状、唯一性、方法参数齐不齐，任何机器都能查。"""
    doc = _doc()
    m = doc["method"]
    assert (m["center_crop_frac"], m["resize"], m["domain"]) == (0.70, 384, "display-sRGB"), (
        f"量法参数被改动过：{m}——改了方法就必须重算统计量并同步本锁")
    frames = doc["frames"]
    assert len(frames) == 11, f"参考应有 11 张，实得 {len(frames)}"
    files = [e["reference_file"] for e in frames]
    assert len(set(files)) == len(files), "同一张图被登记了两次"
    # 11 张图对应 **9 个帧**：G200-2 与 G200-7 各有两个版本（用户自己出的第二版），
    # 所以 frame_id 允许重复，但重复**只能**以 variant_of 的形式出现。
    # （第一版期望写成"frame_id 唯一"，被这条锁当场判红——**错的是期望不是实现**，
    #  按"主帧唯一 + 变体必须指名主帧"改严，而不是放宽成不查。）
    mains = [e for e in frames if not e["variant_of"]]
    assert len(mains) == 9, f"主帧参考应 9 张，实得 {len(mains)}"
    assert len({e["frame_id"] for e in mains}) == len(mains), "两个主帧指向同一帧"
    for e in frames:
        stem = pathlib.Path(e["reference_file"]).stem
        if e["variant_of"]:
            assert e["variant_of"] == e["frame_id"].split("::")[1], (
                f"{stem}: 变体指向的帧与 frame_id 不一致")
            assert stem == f"{e['variant_of']}-2", (
                f"{stem}: 变体文件名与它声称的主帧对不上")
        else:
            assert stem == e["frame_id"].split("::")[1], (
                f"{stem}: 主帧文件名与 frame_id 对不上")
    for e in frames:
        assert len(e["reference_sha256"]) == 64 and e["reference_bytes"] > 1_000_000, (
            f"{e['frame_id']} 的哈希/字节数不像真实成品：{e['reference_sha256'][:12]}")


def test_reference_files_match_their_recorded_hashes():
    """图在场时：sha256 必须逐张相等。**目标被换掉必须在这里红，而不是靠人眼**。"""
    miss = _missing()
    if miss:
        pytest.skip(f"降级环境：{len(miss)} 张用户参考不在本机（大文件不入库）："
                    f"{miss[0]} 等——**这不是通过**，目标未被校验")
    for e in _doc()["frames"]:
        raw = (ROOT / e["reference_file"]).read_bytes()
        assert len(raw) == e["reference_bytes"], (
            f"{e['frame_id']} 字节数变了：{len(raw)} ≠ {e['reference_bytes']}")
        assert hashlib.sha256(raw).hexdigest() == e["reference_sha256"], (
            f"{e['frame_id']} 的参考图被替换过：目标不再等于当初签字的那批图")


def test_recorded_look_stats_still_reproduce():
    """图在场时：按记录的方法重算，统计量必须复现（容差 1e-3，只吸收 PIL 版本差异）。"""
    miss = _missing()
    if miss:
        pytest.skip(f"降级环境：用户参考不在本机，外观统计未校验（{len(miss)} 张缺失）")
    m = _doc()["method"]
    bad = []
    for e in _doc()["frames"]:
        got = _stats(ROOT / e["reference_file"], m)
        for k, v in got.items():
            if abs(v - e["stats"][k]) > 1e-3:
                bad.append(f"{e['frame_id']} {k}: 重算 {v} ≠ 记录 {e['stats'][k]}")
    assert not bad, "参考图的外观统计复现不出来：\n  " + "\n  ".join(bad[:8])


def test_target_is_darker_and_more_contrasty_than_the_current_output():
    """**反空断言**：这批参考不能是"随便什么图"。

    如果哪天映射被生成成一批全白/全黑的图，前两条锁照样绿。
    这里钉住它作为"目标"的最低合理性：阴影确实压得下去、对比确实存在。
    """
    miss = _missing()
    if miss:
        pytest.skip(f"降级环境：用户参考不在本机，合理性未校验（{len(miss)} 张缺失）")
    for e in _doc()["frames"]:
        s = e["stats"]
        assert s["luma_p05"] < 0.40, (
            f"{e['frame_id']} 阴影 p05={s['luma_p05']} 太亮——"
            "这不像是去掉色罩的成品，目标本身可疑")
        assert s["luma_std"] > 0.10, (
            f"{e['frame_id']} 对比 std={s['luma_std']} 过低，同上")
