"""写入 P0-A 人工标注（方向经视觉核对；框经渲染验证）。

可重复运行（覆盖写）。verified 仅 G200-1（框+方向都核对过）；
其余样片方向已目检但画面框待 P0-C 检测器改进后核对，verified=false。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negative_lab.calibration.annotations import Annotation, save_annotation  # noqa: E402

# 方向判定依据：负片中"暗=场景亮(天空)"。outputs/ann/G200-1_orient.jpg 四格
# 与 orient_grid.jpg 八格目检记录（2026-08-31，agent 目检，待用户确认）。
ANN = [
    Annotation(
        file="20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-1.NEF",
        roll="1298_G200", rotation_degrees=180, coordinate_space="source",
        frame_roi=(966, 376, 6541, 5166),      # 渲染验证：紧贴画面四边，排除片基与齿孔边
        base_roi=(6721, 66, 7370, 5522),       # 未曝光片基条带（帧底部）
        reference_file="20260709_SAMPLE-ID(1)/1298_G200/翻拍/NLP/G200-1.jpg",
        notes="整改前失败基线：倒置、含片边、偏青蓝、冲淡。天空在场景上方=负片下缘高密度。",
        verified=True, verified_by="agent-inspected", verified_date="2026-08-31",
    ),
    # —— 方向已目检、画面框待核对（P0-C 改进检测器后逐个验证再升 verified）——
    Annotation(file="20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-2.NEF",
               roll="1298_G200", rotation_degrees=180,
               notes="方向目检 180；框待核对", verified=False),
    Annotation(file="20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-5.NEF",
               roll="1298_G200", rotation_degrees=180,
               notes="方向目检 180；框待核对", verified=False),
    Annotation(file="20260709_SAMPLE-ID(1)/1298_G200/翻拍/G200-12.NEF",
               roll="1298_G200", rotation_degrees=180,
               notes="方向目检 180；框待核对", verified=False),
    Annotation(file="20260709_SAMPLE-ID(1)/1299_G200/翻拍/G200-1.NEF",
               roll="1299_G200", rotation_degrees=90,
               notes="holdout 卷；方向目检 90CW（outputs/visual/G200-1_1299_orient.jpg："
                     "草原+山+建筑，天空在上）；画面框 (1584,143,7370,5203) 与片基带"
                     "精确衔接，待用户核对后升 verified",
               verified=False),
    Annotation(file="20260709_SAMPLE-ID(1)/1299_G200/翻拍/G200-5.NEF",
               roll="1299_G200", rotation_degrees=90,
               notes="holdout 卷；方向目检 90CW；检测框 x0=1111 含 KODAK 字样片边带，"
                     "需右移（见 outputs/visual/G200-5_1299_orient.jpg 顶带）",
               verified=False),
    Annotation(file="20260709_SAMPLE-ID(1)/1300_CN400/翻拍/CN400-1.NEF",
               roll="1300_CN400", rotation_degrees=90,
               notes="胶片条带横向放置：天空在左缘 -> 顺时针 90°；框待核对",
               verified=False),
    Annotation(file="20260709_SAMPLE-ID(1)/1300_CN400/翻拍/CN400-13.NEF",
               roll="1300_CN400", rotation_degrees=180,
               notes="方向目检 180；框待核对", verified=False),
    Annotation(file="20260709_SAMPLE-ID(1)/1300_CN400/翻拍2/翻拍/CN400-5.NEF",
               roll="1300_CN400", rotation_degrees=180,
               notes="翻拍2 会话；方向目检 180；框待核对", verified=False),
]


def main() -> int:
    for a in ANN:
        f = save_annotation(a)
        print("saved", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
