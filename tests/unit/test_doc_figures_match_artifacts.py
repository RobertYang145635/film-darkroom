"""验收文档里的**手抄数字**必须与真实工件一致（防漂移锁）。

`docs/visual-acceptance.md` 里的帧数/条数/已签数是**手抄**的。本会话已多次
被这类"抄下来就对、改了就漂"的数字咬过，所以把它们变成**机器核对**：
期望值**从工件反算**，文档措辞用**精确锚点**匹配。

**覆盖范围（诚实声明，不假装全覆盖）**：本锁只核对
`metrics 帧数`、`判定条数`、`已签数`、`manifest 唯一 id 数` 四个数字；
文档里其余数字（分层表、置信度占比、剪裁分类计数）**仍未核对**。
"""

from __future__ import annotations

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "visual-acceptance.md"


def _load(rel: str):
    p = ROOT / rel
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _frames_of(obj):
    """metrics 可能是 list，也可能是 {uid: frame} —— 两种形状都要能吃。"""
    if obj is None:
        return []
    fr = obj.get("frames") if isinstance(obj, dict) else obj
    if isinstance(fr, dict):
        return list(fr.values())
    return list(fr or [])


def _metrics_frames() -> int:
    return len(_frames_of(_load("fixtures/verdicts/metrics.json")))


def _decisions() -> list:
    d = _load("fixtures/verdicts/decisions.json") or {}
    return _frames_of(d)


def _signed_count(entries: list) -> int:
    n = 0
    for e in entries:
        if str(e.get("verdict") or "").strip():
            n += 1
    return n


def _manifest_unique_ids() -> int:
    m = _load("fixtures/acceptance/manifest.json") or {}
    if isinstance(m, dict) and "rolls" in m:          # 卷→帧 的两层结构
        frames = [f for r in (m.get("rolls") or [])
                  for f in (r.get("frames") or [])]
    else:
        frames = _frames_of(m)
    return len({f.get("id") for f in frames if f.get("id") is not None})


def test_frame_count_in_doc_matches_metrics_artifact():
    n = _metrics_frames()
    doc = DOC.read_text(encoding="utf-8")
    assert n > 0, "工件读不出来，锁本身失效（不是文档的问题）"
    assert re.search(rf"(?<!\d){n}(?!\d)", doc), \
        f"metrics 实测 {n} 帧，文档里却找不到这个数字（漂移）"


def test_verdict_entry_count_matches_doc_literal():
    """精确锚点：`48 帧已生成` —— 条数变了而文档没改，就红。"""
    n = len(_decisions())
    doc = DOC.read_text(encoding="utf-8")
    assert f"{n} 帧已生成" in doc, \
        f"判定文件实有 {n} 条，文档的『{n} 帧已生成』措辞已对不上（漂移）"


def test_signed_count_matches_doc_literal_and_stays_zero_until_user_signs():
    """已签数是**用户的字段**：只要还没人签，文档就必须写 `0 帧已签`。

    这条同时锁住两件事：文档不漂；**也锁住"脚本偷偷签"的回归**
    ——若哪天有代码往 `verdict` 里写值，工件的已签数变 >0，本锁立刻红。
    """
    signed = _signed_count(_decisions())
    doc = DOC.read_text(encoding="utf-8")
    assert f"{signed} 帧已签" in doc, \
        f"实测已签 {signed} 条，文档写的不一致（漂移，或有人替用户签了）"
    if signed == 0:
        assert "未验收" in doc, "0 签署却把状态写成已验收，是自欺"


def test_manifest_unique_id_claim_matches_artifact():
    """本会话踩过的坑：帧 id **不唯一**；文档必须写清真实的唯一数。"""
    uniq = _manifest_unique_ids()
    total = _metrics_frames()
    doc = DOC.read_text(encoding="utf-8")
    assert uniq > 0, "manifest 形状变了，先修锁再谈文档"
    if uniq != total:
        assert re.search(rf"{total} 帧只有 {uniq} 个", doc), (
            f"实测 {total} 帧 / {uniq} 个唯一 id，文档不再说明这个坑——"
            "新人会以为 id 能当键用（本会话为此重写过键）")


def test_lock_is_not_vacuous_by_checking_it_against_a_mutated_doc():
    """反空判断：把文档里的条数改成别的数字，检测器**必须**看不见它。"""
    entries = len(_decisions())
    real = DOC.read_text(encoding="utf-8")
    broken = real.replace(f"{entries} 帧已生成", f"{entries + 1} 帧已生成", 1)
    assert broken != real, "替换没生效 → 锚点不存在，前面那条锁其实是空的"
    assert f"{entries} 帧已生成" not in broken


def test_low_base_confidence_claim_matches_artifact_and_avoids_the_or_trap():
    """文档"98/169 帧片基置信度低于 0.6"要核对——**并锁死正确的取值写法**。

    本锁的血泪来历：我用 `(f.get("base_confidence") or 1) < 0.6` 算出 **24**，
    兴冲冲去报"文档写错了"；**其实文档是对的（98）**，错的是探针——
    `or` 把**合法的 0.0** 当成缺失吞掉（实测 `min=0.0`、中位数 `0.524`，0.0 大量存在）。
    所以这里**两种写法都算一遍**：把"错写法会得到不同的数"这件事**钉进测试**，
    下次谁再用 `or` 求默认值，会立刻看到差异而不是重蹈覆辙。
    """
    m = json.loads((ROOT / "fixtures/verdicts/metrics.json").read_text(encoding="utf-8"))
    fr = m["frames"]
    fr = fr if isinstance(fr, list) else list(fr.values())
    assert len(fr) > 0

    def count(pred):
        return sum(1 for f in fr if pred(f))

    correct = count(lambda f: (f.get("base_confidence")
                               if f.get("base_confidence") is not None else 1.0) < 0.6)
    buggy = count(lambda f: bool(f.get("base_confidence")) and f["base_confidence"] < 0.6)
    assert correct != buggy, (
        "两种写法算出同一个数 → 本用例失去意义（语料里已无 0.0 置信度？"
        "那就删掉这条锁并在文档里说明，别留一条空锁）")

    doc = DOC.read_text(encoding="utf-8")
    assert f"{correct}/169 帧片基置信度" in doc, (
        f"实测低置信度 {correct}/169，文档措辞对不上（漂移）；"
        f"注意**别用 `or` 取默认值**，那会得到 {buggy}")

