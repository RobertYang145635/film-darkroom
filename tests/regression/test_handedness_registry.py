"""手性登记表（㊲）：**1299 整卷交付的是镜像**，而四项验收统计对镜像恒等 ⇒ 只能靠登记表守住。

## 缺陷与证据链（四路独立，全部当场可复算）

`1299_G200` 的片带在画布里是竖躺的（人工框 `h/w` 1.2201~1.2403），35mm 单格规则
（`geometry.py:SIDEWAYS_FRAME_ROTATION=270`，B-12 追加③）把它补转 270°。转完**正立**，
但**左右与用户的手工样片相反**：

1. **车牌**（最硬的一条）：样片 `G200-6` 车尾牌照顺读 `川G·G004W`
   （`outputs/ref_look_probe/plate_his_1299-6.png`）；同一帧按生产几何（rot=270，
   `mirror_horizontal=False`）出的图，牌照是反字；加上水平镜像后顺读
   （`mirror_proof_1299_G200-6.png` 上下两格）。
2. **片边印字**：样片顶边条 `KODAK` 顺读（`chir_text2_1299-6.png` 第 3 格），
   我方顶边条（= 源的右片边）同一物理边条是反字（`chir4_源右片边_x5563.png`）。
3. **NLP 同手性**：NLP 读同一张 NEF，车在画面左侧、与样片一致
   （`nlp_pairing_check.png` 三格）⇒ 镜像不是"NLP 也这么错"，是**我们这条解码/旋转链**的口径。
4. **逐帧看图**：`handedness_wall_1/2.png` 五列二面体并排，1299 三帧全部"我+水平镜"那一列与样片对齐。

## 为什么锁的是"登记"而不是"检测"

自动判手性这条路本轮**量死**：梯度图在 8 个二面体变换下的相关（`tmp_handedness.py`）
在 11 帧里至少 4 帧判错（1298 的 `rot=180` 组被判"水平镜像最优"，而那一组按看图是恒等），
绝对相关只有 0.04~0.38 ⇒ **不是锁级传感器**。这是 B-12 追加④（相关判 180° 无区分力）
之后同一类尺子的第三次失败。⇒ 与 ㉛ 的"方向/取景只能看图"同一条结论：
**手性只能登记 + 由人翻**，不许写自动翻转。

## 本锁守三件事

1. 1299 的标注**仍然** `mirror_horizontal=False` ⇒ 缺陷仍然在。变红 = 有人已经修了
   （或改动了规则/标注），必须重看 `handedness_wall_*.png` 并同步本表与 `docs/blockers.md` ㊲。
2. **管线绝不自己引入镜像**：35mm 规则只补旋转；用左手系图形逐帧验证，
   并带"镜像确实会改变图形"的非空对照（否则这条断言会在镜像被静默禁用时空转）。
3. `g200_manual_look.json` 的 `nlp_reference` 必须与验收清单一致、且落在**本帧所属卷**的目录里。
   本轮抓到它三张 1299 帧指向 1298 的 NLP 文件（河谷 vs 公路白吉普）——
   2026-10-08 那次"参考配错卷"的修复只改了 `frame_id`/`source_nef`，
   漏了 `tmp_build_ref_map.py:96` 按 1298 建的这一项。**同名帧的坑会以没被复查的字段形式复发。**

⚠ 未决（写在 `docs/blockers.md` ㊲，不进锁）：`1298_G200` 的翻转轴两条证据互相矛盾——
看图说 `rot=180` 那 5 帧恒等就对、`rot=0` 那 3 帧差 180°；梯度相关说整卷都是"源的垂直翻转"。
`G200-5` 的标注本来就 `mirror_horizontal=True`，而看图那张表**没有施加**这个镜像，
所以那一格的读法不能作数。⇒ 1298 不动数据，等用户自己按样片翻。
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from negative_lab.calibration.annotations import load_annotation
from negative_lab.library.models import GeometryRegions
from negative_lab.pipeline.geometry import (OrientTransform, mirror_h, rotate_cw,
                                            sideways_correction)

ROOT = pathlib.Path(__file__).resolve().parents[2]
MIRRORED_ROLL = "1299_G200"
LOOK = json.loads((ROOT / "fixtures" / "references" / "g200_manual_look.json")
                  .read_text(encoding="utf-8"))
MANIFEST = json.loads((ROOT / "fixtures" / "acceptance" / "manifest.json")
                      .read_text(encoding="utf-8"))
MAN = {f["id"]: f for f in MANIFEST["frames"]}
DELIVERY = [e["frame_id"] for e in LOOK["frames"] if not e["variant_of"]]


def _triple(fid: str) -> tuple[int, int, int, bool]:
    """(用户 rot, 规则补角, 生效 rot, 镜像位)——全部现读标注，不抄常量。"""
    ann = load_annotation(MAN[fid]["file"])
    assert ann is not None, f"缺标注：{MAN[fid]['file']}"
    user = int(ann.rotation_degrees or 0)
    add = sideways_correction(ann.image_area_roi, user)
    return user, int(add), (user + add) % 360, bool(ann.mirror_horizontal)


#: 每卷"与用户样片同手性"所需的 (生效旋转, 镜像位)。两条都是**看图 + 读实物文字**定的：
#: 1298 源图既倒置又镜像 ⇒ 转 180 + 水平镜像；1299 片带竖躺（35mm 规则补 270）+ 水平镜像。
#: 旁证：`1298::G200-5` 用户 2026-10-13 就自己设过 `mirror_horizontal=true`，正是这条规则的一半。
EXPECTED_GEOMETRY = {"1298_G200": (180, True), "1299_G200": (270, True)}


def test_every_roll_delivers_the_same_handedness_as_the_users_samples():
    """登记表：两卷每一帧**确认过的**标注，其生效几何必须等于 `EXPECTED_GEOMETRY`。

    2026-10-15 这条锁先按"缺陷仍在"写（1299 缺镜像、车牌反字），当天用户说
    "方向你不能改吗"⇒ 已按卷把 rot/镜像统一改到位，并按锁自己的指示把断言翻过来
    （**没有删锁**）。现在它守的是另一半：别有人改回去、别只改一半帧、
    别把两卷改成同一个数（两卷的胶片朝向不同，规则也不同）。
    """
    seen = {r: 0 for r in EXPECTED_GEOMETRY}
    bad, skipped = [], []
    for path in sorted((ROOT / "fixtures" / "annotations").glob("*.json")):
        roll = path.stem.split("__")[0]
        if roll not in EXPECTED_GEOMETRY:
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("status") != "verified" or d.get("verified_by") != "user":
            skipped.append(path.stem)      # 未确认的标注不进产品，也不进本锁
            continue
        user = int(d.get("rotation_degrees") or 0)
        add = sideways_correction(d.get("image_area_roi") or d.get("frame_roi"), user)
        got = ((user + add) % 360, bool(d.get("mirror_horizontal")))
        seen[roll] += 1
        if got != EXPECTED_GEOMETRY[roll]:
            bad.append(f"{path.stem}: 生效 {got} ≠ 登记 {EXPECTED_GEOMETRY[roll]}"
                       f"（用户 rot={user} 规则补 {add}）")
    print()
    print("手性登记（生效角, 镜像位）：")
    for roll, want in EXPECTED_GEOMETRY.items():
        sk = [s for s in skipped if s.startswith(roll)]
        print(f"  {roll}: 期望 {want}  已确认 {seen[roll]} 份  跳过未确认 {sk or '无'}")
    for line in bad:
        print("  ✗", line)
    assert all(v >= 11 for v in seen.values()), f"登记的标注份数变了，逐卷判据要重估：{seen}"
    assert not bad, "有帧的手性与登记表不一致（照片会左右反）：" + "; ".join(bad)

    # 反空对照（只在内存里）：镜像位若被改掉，同一段判定必须抓到，否则这条锁空转
    user = 0
    got = ((user + sideways_correction([1170, 342, 7146, 5160], user)) % 360, False)
    assert got != EXPECTED_GEOMETRY["1299_G200"], "反空对照失效：镜像位没参与判定"


def test_pipeline_never_mirrors_on_its_own():
    """不变量：**手性只由用户的 `mirror_horizontal` 决定，35mm 规则永远不引入镜像**。

    1299 的镜像缺陷之所以要用户动手，就是因为规则只补旋转；这条锁守的是另一半：
    规则（或以后任何自动几何）不许偷偷把图翻过来。判据用左手系图形的"是否等于某个纯旋转"。
    """
    src = np.arange(24 * 24 * 3, dtype=np.float32).reshape(24, 24, 3)
    pure = [rotate_cw(src, k) for k in (0, 90, 180, 270)]

    def reflected(arr) -> bool:
        return not any(np.array_equal(arr, p) for p in pure)

    for fid in sorted(DELIVERY):
        user, add, eff, mir = _triple(fid)
        with_rule = OrientTransform(src.shape[:2], eff, mir, 0.0).apply(src)
        without_rule = OrientTransform(src.shape[:2], user, mir, 0.0).apply(src)
        # 施加规则前后手性必须一致（规则只转不翻）
        assert reflected(with_rule) == reflected(without_rule) == mir, (
            f"{fid}: 用户镜像={mir} 用户 rot={user} 规则补 {add} ⇒ "
            f"手性 {reflected(with_rule)}/{reflected(without_rule)}；"
            "几何规则不许改变手性（方向由用户管，见 B-12 追加③/㊲）")
        # 组合次序钉死：先转后翻（geometry.py 模块注释的那一条）
        want = mirror_h(rotate_cw(src, eff)) if mir else rotate_cw(src, eff)
        assert np.array_equal(with_rule, want), f"{fid}: 旋转与镜像的合成次序变了"
    # 非空对照：镜像位真的会改变像素，否则上面全部空转
    assert reflected(mirror_h(src)), "mirror_h 没改变手性 ⇒ 对照失效"
    assert sideways_correction((0, 0, 124, 100), 0) in (0, 90, 180, 270)


def test_look_json_nlp_reference_matches_manifest_and_own_roll():
    """`nlp_reference` 必须与清单一致且落在本帧所属卷目录里（同名帧错配的历史坑）。"""

    def check(entries):
        bad = []
        for e in entries:
            rel = e.get("nlp_reference")
            if rel is None:
                continue
            roll = e["frame_id"].split("::")[0]
            if f"/{roll}/" not in rel:
                bad.append(f"{e['frame_id']} → {rel}（不在本卷目录里）")
            elif MAN[e["frame_id"]].get("nlp_reference") != rel:
                bad.append(f"{e['frame_id']} → {rel} 与清单不一致")
        return bad

    bad = check(LOOK["frames"])
    print("\nnlp_reference 配对审计：", "全部一致" if not bad else "\n".join(bad))
    assert not bad, "参考图元数据里的 NLP 路径跨卷/与清单不一致（见 ㊲ 的第三次同名帧事故）"

    # 反空对照：把一张 1299 的 NLP 路径改回 1298（就是本轮修掉的那个状态），必须被抓到
    poisoned = json.loads(json.dumps(LOOK))
    hit = next(e for e in poisoned["frames"] if e["frame_id"] == f"{MIRRORED_ROLL}::G200-6")
    hit["nlp_reference"] = hit["nlp_reference"].replace(f"/{MIRRORED_ROLL}/", "/1298_G200/")
    assert check(poisoned["frames"]), "跨卷 NLP 路径没被抓到 ⇒ 这条锁是空转的"


def test_geometry_regions_round_trip_keeps_mirror():
    """镜像位要能过 `to_dict/from_dict`（sidecar 与草稿回写都靠它）。"""
    g = GeometryRegions(rotation_degrees=270, mirror_horizontal=True,
                        output_crop_roi=(10, 10, 500, 900), image_area_roi=(20, 20, 490, 880))
    back = GeometryRegions.from_dict(g.to_dict())
    assert back.mirror_horizontal is True and back.rotation_degrees == 270
    # 且镜像确实被 `mirror_h` 实现为水平翻转（不是垂直）
    a = np.array([[[0.0], [1.0]]], dtype=np.float32)
    assert np.array_equal(mirror_h(a), np.array([[[1.0], [0.0]]], dtype=np.float32))


def test_manifest_frames_exist_on_disk_or_skip_loudly():
    missing = [fid for fid in DELIVERY if not (ROOT / MAN[fid]["file"]).exists()]
    if missing:
        pytest.skip(f"语料缺失，手性登记未验：{missing}")
    assert all((ROOT / MAN[fid]["file"]).exists() for fid in DELIVERY)
