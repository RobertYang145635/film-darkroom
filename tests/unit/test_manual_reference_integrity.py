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


def test_acceptance_stats_are_rotation_blind_so_they_cannot_certify_orientation(tmp_path):
    """㉛：**这套验收量法对 90° 旋转恒等**——把它钉成事实，防止有人拿"全绿"当"图没问题"。

    ㉛ 修掉"35mm 竖幅规则把用户正方裁切静默转 90°"之后，
    `outputs/signed31.txt` 与 `outputs/signed30b.txt` 的 11 帧四轴数值**逐位相同**
    （四帧的输出旋转确实从 270° 变回 0°）。原因在量法：
    方形 `resize(384,384)` + 方形中心窗 + `mean/percentile/std` 全是**与像素次序无关**的量
    ⇒ 转置画面一个数都不动。

    所以这条锁**不是**在验收方向，而是在**登记验收器的盲区**：
    方向与取景只能靠看图，或靠 `test_output_aspect_is_on_his_side_of_the_wall`
    那一类"看得见的口径"（纵横比）来抓。
    """
    m = _doc()["method"]
    rng = np.random.default_rng(20261013)
    grad = np.linspace(20, 200, 420, dtype=np.float64)[:, None]
    a = np.dstack([np.repeat(grad, 260, axis=1),
                   np.repeat(grad * 0.72, 260, axis=1),
                   np.repeat(grad * 0.45, 260, axis=1)])
    a[30:120, 150:250] = 250.0
    a = np.clip(a + rng.integers(0, 12, a.shape), 0, 255).astype(np.uint8)
    p1, p2 = tmp_path / "as_is.png", tmp_path / "transposed.png"
    Image.fromarray(a).save(p1)
    Image.fromarray(np.ascontiguousarray(a.transpose(1, 0, 2))).save(p2)
    assert Image.open(p1).size != Image.open(p2).size, "两张图的纵横必须真的不同"
    s1, s2 = _stats(p1, m), _stats(p2, m)
    assert s1.keys() == s2.keys()
    for k in s1:
        assert abs(s1[k] - s2[k]) < 1e-3, f"{k} 随 90° 旋转变了：{s1[k]} vs {s2[k]}"


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
    ids = [e["frame_id"] for e in frames]
    assert len(set(ids)) == len(ids), (
        f"两张参考指向同一帧：{[i for i in set(ids) if ids.count(i) > 1]}——"
        "一张手工成品只可能对应一张底片")
    # ⚠ 这里**原来**断言"文件名 stem 必须等于 frame_id 的帧号"，
    # 而那条断言正是错配的来源：用户的文件名是他自己起的，
    # `1298_G200` 与 `1299_G200` 又**同名**（都有 G200-1…G200-11），
    # 按名字对齐就把 1299 的三帧配到了 1298。
    # 配对现在由**内容**决定（肉眼比对，见 test_the_three_cross_roll_mismatches…），
    # 结构上只要求"卷名与 nef 路径一致 + 帧在清单里"，不再要求名字相等。
    mains = [e for e in frames if not e["variant_of"]]
    assert len(mains) == 11, f"11 张各对应一帧，实得主帧 {len(mains)}"
    for e in frames:
        stem = pathlib.Path(e["reference_file"]).stem
        if e["variant_of"]:
            assert stem.startswith(f"{e['variant_of']}-"), (
                f"{stem}: 变体文件名应以它声称的主帧开头")
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


def test_pairing_is_structurally_consistent():
    """**锁**：`frame_id` 的卷名必须出现在 `source_nef` 路径里，且该帧在验收清单里存在。

    由来（2026-10-08 实测）：这张表当初是**按文件名**配的
    （`G200-6.jpg ↔ 1298_G200::G200-6`），而 `1298_G200` 与 `1299_G200`
    两卷的帧**同名**，于是把 1299 的三帧配到了 1298。
    后果不是"少一帧数据"而是**目标被换成另一张照片**：
    1298::G200-6 是河谷，用户的 G200-6 是公路上的白车。
    这条锁挡不住所有错配（同名同卷仍可能错），但挡得住跨卷错配这一整类。
    """
    doc = _doc()
    man = json.loads((ROOT / "fixtures" / "acceptance" / "manifest.json")
                     .read_text(encoding="utf-8"))
    by_id = {f["id"]: f for f in man["frames"]}
    bad = []
    for e in doc["frames"]:
        roll = e["frame_id"].split("::")[0]
        if f"/{roll}/" not in e["source_nef"]:
            bad.append(f"{e['reference_file']} 配到 {e['frame_id']}，"
                       f"但 nef 路径是 {e['source_nef']}")
        if e["frame_id"] not in by_id:
            bad.append(f"{e['frame_id']} 不在验收清单里")
    assert not bad, "参考图配对与底片不一致：\n  " + "\n  ".join(bad)


def test_the_three_cross_roll_mismatches_stay_corrected():
    """钉住已修正的三条：它们是**肉眼比对内容墙**得出的，不是算法。

    自动特征在这里没有分辨力（灰度相关 G200-1 对自己 0.573、对 G200-2 也 0.573；
    边缘布局领先只有 0.00~0.08 还把 G200-10 判成 G200-4），
    所以修正必须显式记录，被改回文件名顺序时要有人负责。
    """
    want = {"G200-6.jpg": "1299_G200::G200-6",
            "G200-2-2.jpg": "1299_G200::G200-2",
            "G200-7-2.jpg": "1299_G200::G200-7"}
    got = {e["reference_file"].split("/")[-1]: e for e in _doc()["frames"]}
    for name, fid in want.items():
        assert got[name]["frame_id"] == fid, (
            f"{name} 又配回 {got[name]['frame_id']} 了；它实际是 {fid}"
            "（两卷帧同名，按文件名配必错）")
        assert got[name]["variant_of"] is None, (
            f"{name} 不是任何帧的第二版本，是另一卷的另一帧")


def test_pairing_lock_catches_a_fabricated_cross_roll_entry():
    """**负控制**：造一条 1298↔1299 混配的条目，结构锁必须抓到。"""
    doc = _doc()
    e = dict(doc["frames"][0])
    e["frame_id"] = "1298_G200::G200-1"
    e["source_nef"] = "20260709_SAMPLE-ID(1)/1299_G200/翻拍/G200-1.NEF"
    roll = e["frame_id"].split("::")[0]
    assert f"/{roll}/" not in e["source_nef"], "构造失败：这条样例本身不矛盾"
