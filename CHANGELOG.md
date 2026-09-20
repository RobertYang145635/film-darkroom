# 变更日志

> 说明：v6 / v6.1 / v6.2 此前只存在于规格文档与工作区，未登记进本日志，
> 导致"从 README/CHANGELOG 入口进来会拿到错误的当前范围"（2026-09-07 审查 §4.1）。
> 以下条目按已落地代码补记，不重写历史。

## 把归因送到眼前（开发版，未发布）— 2026-09-09 Round 10

上一轮门禁与片基估计器有了证据（`clip_attribution` / `best_rejected_region` /
分项 `scores`），但**只加字段不加出口，在用户那儿等于没修**
（本仓库死代码审计的先例："只有 emit 没有 connect"）。

- 新增 `src/negative_lab/diagnostics/explain.py`（纯函数、无 Qt、无磁盘）：
  `base_fallback_note()` 说清片基为什么没用上实测——是被**哪一项分项**否的
  （颗粒/背光散布 / 取样区污染 / 样本量不足）、逐通道 MAD 是多少，
  并写明"散布取三通道最大值，**一个通道差就作废整帧测量**"；
  没框到取样区说"没框到"，**不写"污染"**（那是编造）；用上了实测就闭嘴。
  `clip_note()` 说清高光剪裁是采集端丢失还是软件新引入。
- **出口两处**：右栏「技术信息」新增 `evidence` 行（`TechSection.set_info` +
  `MainWindow._update_tech_info` 接线）；判定条标题加短标签
  `[片基被拒:spread]` / `[高光:采集端已丢失]`，完整原话写进
  `fixtures/verdicts/decisions.json` 的 `evidence` 字段（长文案会挤掉图，
  所以图上短、文件里全）。
- **回归锁 8 项**：`tests/unit/test_explain.py`（点名分项与最噪通道、
  没框到≠污染、量到了就闭嘴、**文案与门禁失败信息同调同通道**、
  实例与 `to_dict()` 两种读法同文）；`tests/unit/test_tech_evidence_in_ui.py`
  （证据真的贴进 `lbl_body`；没证据不许凭空多一行"证据："；
  用 duck-typed stub 直接调 `MainWindow._update_tech_info` 锁**接线**，
  不拉起整个窗口）。
- `build_verdict_kit.py --measure` 另记 `clip_attribution` /
  `best_rejected_region`，判定条重渲染后带归因标签。
- 基线：**485 passed**（477 + 6 + 2）· ruff 无豁免全绿。

## 归因轮：谁弄坏了高光，谁弄丢了片基（开发版，未发布）— 2026-09-09 Round 9

本轮只问"这数字是谁造成的"，给 `build_verdict_kit.py --measure` 加了片基侧证据
（`base_dmin` / `base_contamination` / `base_black_level` / `input_clip` / `dmax`，
metrics `schema_version` → 2，重测 169 帧 / 1254 s），量出两个硬结论。

- **被挡导出的 21 帧，软件的高光罪行为零**。21/169 帧因技术正片 B 通道高端剪裁
  >1% 被 §8.2 判 FAIL，但逐帧看 `input_clip.low[B]` 与 `technical_preclip.high[B]`
  **完全相等**（负片越密→正片越亮，对应关系是确定的物理事实）：
  剪裁 100% 继承自采集端，最差 CN400-3 的 11.0% 是相机当场丢的。
  → 门禁新增 `clip_attribution`（逐通道 `high_inherited`/`high_introduced`），
  FAIL 文案改成说得出归因（"其中 4.26% 在输入端已是死数据（软件新引入 0.00%）
  ——采集端就已丢失，重扫/重拍才能根治"）。**判定一个字没改**，>1% 仍 FAIL；
  低端明确**不假装能归因**（其输入端对应量是 above_base，返回 None + 原因）。
  是否该继续硬挡导出记入 `docs/blockers.md` **B-6**（§8.2 两条规则在
  "剪裁全部继承"时互相冲突，属产品决定）。
- **44% 帧退回通用先验，是 B 通道一票否决**。74/169 帧 `generic_prior`，
  报表统一写 `contamination=1.000 / confidence=0.000`，看着像"片基全脏"；
  实测（`scripts/diagnose_base_fallback.py`）取样区**量到了东西**
  （1 万像素、污染 25%、中位透射率 `[0.244, 0.093, 0.026]`），
  而 `rel_spread` 取**三通道 MAD 的最大值**，橙罩下 B 透射率最低、噪声最大
  （逐通道 MAD `[R 0.018, G 0.048, B 0.076]`）→ `spread_score 0.298
  < min_base_confidence 0.45` → 整区作废。**R/G 干净，B 一个通道否掉整帧片基。**
  本轮只修"报不出真相"：`_roi_stats` 写出三项分项 `scores`，
  拒绝路径带出 `best_rejected_region`（真实污染/保留率/散布/中位透射率 +
  `binding_score`，并标注"未参与任何计算"）。
  放宽容差**没有做**：`TOLERANCE` 受 v5 §4.4 冻结约束，且一改就等于改变约 44%
  帧的颜色与反差，必须走版本提升 + 视觉签字 —— 记入 **B-5**。
  注意 `contamination=1.0` 是 **load-bearing** 的（卷级模型靠它排除帧），
  所以本轮测试明确锁住"聚合值不许顺手改成真实值"。
- **回归锁 9 项**：`tests/unit/test_clip_attribution.py`（全继承→文案点名采集端、
  软件引入→文案点名软件、**归因不是放宽门禁的后门**、低端不编造、输入缺失不崩）；
  `tests/unit/test_base_rejection_reported.py`（干净区必须被接受以免样张空转、
  被拒区判定不改但要报真相、`binding_score` 必须等于三项最小值——不硬编码阈值、
  没有取样区时不许凭空造一个被拒区）。
- 基线：**477 passed**（468 + 5 + 4）· ruff 无豁免全绿 ·
  `metrics.json` schema 2 已重测入库。

## 自动中和从不生效的机制修复（开发版，未发布）— 2026-09-09 Round 8

**这是本仓库迄今最像"产品其实没在工作"的发现**：全语料 169 帧里
**默认候选 `preserve`（保留氛围，等于不去色罩）占 162 帧 = 95.9%**，
`neutral` 只被选中 7 次。三候选在实践上是一候选 —— 用户抱怨的
"色罩没除干净"不是调参问题，是判定逻辑本身把治疗锁死了。

- **根因（可复现）**：`build_candidates()` 的 `trusted` 判据用的
  `scene_confidence(ctx.technical_positive, …)`，喂进去的是**还没去色罩的技术正片**。
  橙色片基自己就把 `sat_pen`（高饱和）和 `neutral_pen`（几乎没有中性面，最多 ×0.325）
  打满 —— **用症状证明"不该治疗"**。实测（`scripts/experiment_trust_domain.py`，
  跨 5 卷 20 帧）：同一帧 conf **0.163→0.952 / 0.131→0.955**，
  `neutral_frac` 0.044→0.507，**trusted 4/20 → 14/20**。
- **第二道坎**：`_choose_default` 旧代码是
  `trusted and conf >= 0.72` 或 `roll_confidence >= 0.7 and trusted`。
  `trusted` 本身就是 `conf >= 0.5`，在同一把尺子上再放 0.72；实测有
  **4 帧 `trusted=True` 仍因这道坎被判 `preserve`**。第二支在卷级片基回退
  `generic_prior` 时永远不成立，而回退帧恰恰最需要明确的中性起点。
- **改法**：可信度改判在**校正后**的画面（技术正片 × 本帧自己的 `correction_gain`，
  即 von Kries 增益域，与 neutral 候选施加的是同一个方向与幅度）；
  `_choose_default` 收敛成一条 `trusted -> neutral`；
  校正**前**的统计与置信度**继续写进 diagnostics**
  （`scene_before_correction` / `confidence_before_correction` /
  `confidence_measured_on`），换判据不许把旧证据藏掉，也不许让新旧数字被混着比。
  **保留的防线**：估计器分歧 `spread > MAX_ESTIMATOR_SPREAD` 仍强制 `preserve`；
  真正的夜景/单色/高饱和场景在校正后依然低分（同批 G200-10 0.50、5207-11 0.13
  仍判不可信），v5 §6.6 的降权意图没有丢。
- **回归锁 5 项** `tests/unit/test_candidate_trust_domain.py`：
  合成"本有中性面 + 片基色罩"样张必须默认 neutral；
  **反向对照**锁住"按旧判据这一帧确实会被否掉"（防止样张太好做而自证）；
  `conf=0.50/0.55/0.71 且 trusted=True` 必须 neutral（0.72 那道坎不许回来）；
  分歧过大仍 preserve；`diagnostics` 必须自述置信度口径。
- **全语料实测效果**（`fixtures/verdicts/metrics.json` 重测 169 帧）：
  默认候选 `preserve 162 / neutral 7` → **`preserve 96 / neutral 73`**，
  即 43.2% 的帧开始真正自动中和；`neutral` 置信度中位数 0.456（校正后口径）。
  片基侧不受影响（`generic_prior` 仍 74 帧 —— 那是另一个问题，见 B-5/待办），
  被门禁挡住的帧数不变（21 帧 B 通道高端剪裁 >1%）。
- **⚠ 待用户视觉签字**：本改动改变了**默认导出内容**。判定条已按新默认重新生成
  （`build_verdict_kit.py --sheets`，48 张）；`scripts/verify_verdicts.py`
  现在判**不通过**（0/48 已签）——签字之前，"效果更好"只是机制上的推断，
  不是证据。若用户判下来 neutral 更差，回滚点是 `_choose_default` 一处。
- **自我翻车记录（保留，勿删）**：为了查那次偶发崩溃，我给
  `pyproject.toml` 的 `addopts` 加了 `-X faulthandler` —— **错的**：`-X` 是
  python 解释器参数，pytest 不认，加完 `python -m pytest` 直接
  `unrecognized arguments: -X`，而 CI 会照抄，等于**把测试命令本身弄炸**。
  已回退为 `addopts = "-ra"`，正确姿势写在配置注释里
  （`python -X faulthandler -m pytest -v`）。教训：改测试入口配置后必须
  立刻跑一次真实命令验证，不能只改文件就算完。
- 测试基线：**468 passed**（440 + 23 判定内核 + 5 可信度）· ruff 无豁免全绿。

## 人工视觉判定基准集（开发版，未发布）— 2026-09-09 Round 7

依据：`docs/用户痛点调研与能力差距分析_20260905.md` §6 第一优先级
（"建立可复现的人眼基准"，两份调研都把它列为一切改进的前置条件）。

- **为什么不是又一份联系表**：`fixtures/acceptance/manifest.json` 的 decision 字段
  自 2026-09-02 起全空。原因不是用户懒：旧联系表把 169 帧堆成每张 300px 高的长图，
  判一张要横滚 7 格、纵翻 169 行；而它写的判定落在
  `outputs/quality/<run-id>/rolls/<卷>/decisions.json` —— `outputs/` 被 gitignore
  且 run-id 每次新生成，**签了也不保留**；且 `accepted: bool` 表达不了
  "微调可救 + 花了几步"，调研的验收判据根本算不出来。
- **新增判定内核** `diagnostics/verdicts.py`（23 项回归锁）：
  `frame_strata()` 分层阈值优先**复用管线自己的数字**
  （`neutral_frac<0.12`、`hue_concentration>0.7`、`dark_frac>0.5` 抄自
  `candidates.scene_confidence` 与 `service` 的 scene_flags；剪裁复用
  `quality_gate.CLIP_WARN_FRAC`），只有天空/低反差两个是像素代理且写明适用条件；
  `select_stratified()` 确定性分层抽样（无随机数、平局按 id，同输入必得同输出）；
  `merge_decisions()` 三条硬规则：**已签判定永远保留、绝不覆盖**，
  未签且未抽中的才退休并记入 `retired_ids`；
  `summarize_verdicts()` 汇总成判据，**没签满直接判不通过**，
  分母为 0 的判据标 ⚪「无法判定」而非打勾（0/0 打勾是假绿）。
- **新增两个脚本**：`scripts/build_verdict_kit.py`
  （`--measure` 整卷走生产服务 `process_roll` 测 169 帧 / 23.6 分钟；
  `--select` 抽 48 帧合并进 `fixtures/verdicts/decisions.json`；
  `--sheets` 每帧一张判定条 / 368 秒）与 `scripts/verify_verdicts.py`
  （判据逐项 + 覆盖率 + 退出码 0/1/2，写 `outputs/verdicts/verdict-report.md`）。
  实测：48 张判定条全部生成、非空白（std≈87）、路径写回，且**没有任何判定被填写**。
- **全量量出来的三个事实**（`fixtures/verdicts/metrics.json`，169 帧）：
  1. **98/169 帧片基置信度 <0.60**，其中 74 帧 `base_source=generic_prior`
     （非物理先验）—— 卷内一致的机制性漏洞，1299_G200 是 9/11 帧回退；
  2. **21/169 帧因 B 通道高端剪裁 >1% 被门禁 FAIL 挡住导出**（最差 11.0%，CN400-3）：
     门禁符合 v5 §8.2，但"每 8 帧导不出 1 帧"是真实吞吐问题；
  3. **语料覆盖不了调研点名的场景**：`night_like` 全库 1 帧、`shadow_clip` 0 帧
     → 基准集必须**先补采**（夜景/暗场、明显欠曝薄负片、第二设备）再签。
- **结论校正（就地标注，不重写历史）**：两份调研都写"没有中性点取色器"——
  **已实现**（`pipeline/rescue.py` + 画布拾点 + 撤销 + 可信度警告）；
  09-06 §4"没有一致性数字"部分过时（右栏卷级片基一致性 +
  `roll_consistency.py` 落盘逐帧实际施加值）。
- `docs/visual-acceptance.md` 从 25 行空表重写为可执行流程
  （三步命令、六个判定字段含义、判据表、真实覆盖率表、agent 不得代签）。

## 冷读审查第二轮（开发版，未发布）— 2026-09-09

依据：`质量审查报告_20260909.md`。原计划四路独立冷读，但本会话 subagent 的
spawn/fork **两条路径都返回 `subagent run failed`**（与交接 §7 的故障同形态），
故为单观察者审查——覆盖面不如四路，不应当作"四路都过了"。
基线：审查前 433 passed → 审查后 **440 passed**（+7 回归锁）· ruff 无豁免全绿。

- **导入不再冻住窗口**：`load_files`（「导入文件…」与双击打开）此前在 **GUI
  线程**逐文件 SHA-256——实测真实样片 NEF 55 MB ≈ 80 ms/张、16 位 TIFF
  274 MB ≈ 400 ms/张，选 20 个 TIFF 约 8 秒「未响应」且横幅不重绘。
  现默认经 `_start_import` 走后台线程，`background=False` 保留同步语义给脚本。
- **导入忙碌守卫生效**：`load_folder` 丢弃 `_busy_enter("import")` 返回值，
  连点两次会并发跑两遍导入、后完成的会话**无声覆盖**前一个；现在忙则拒绝并
  横幅说明（不给不成立的承诺，与 L1-② 同口径）。
- **CI 修正（此前从未跑过：`git remote -v` 为空）**：workflow 装了 Qt 的
  系统库却从不装 PySide6（`-e .` 不含 extras）。实测用 `sys.meta_path`
  屏蔽 PySide6 复现该环境：收集阶段 4 个 error，配合旧 `--maxfail=1`
  → **一个测试都没跑**。改为 `-e ".[gui,dev]"` + 新增「GUI 测试层自检」一步
  （extra 再被删时大声失败，而不是 12 个 Qt 测试模块静默 skip 换假绿）+
  去掉 `--maxfail=1` + `timeout-minutes` + pip 缓存。
- **清除区域后分析必须失效（v6.2 F2 补漏）**：F2 的规则只落在键盘删除路径
  （`_on_roi_deleted` 的 `_changes_statistics` 判据），右栏「清除」与
  「重置几何」同样清掉片基取样区/有效画面区却**不重算**，界面继续显示按
  已删除区域算出的 Dmin/来源/置信度。两处统一到同一条判据。
- **光标态**：退出「忙碌」时只认绘制模式，中性点救援的十字准星被吞成箭头
  （拾点模式仍在，用户以为已退出）；改为同时认绘制/拾点/涂抹。
- **死代码与假签名**（逐个 grep 证明 0 引用才动手）：
  `RightPanel.set_fine_rotation`（只给一个从不被读取的字段赋值、从不动滑杆——
  真正生效的是 `AnalysisSection.set_fine_rotation`，属误导性陷阱）、
  `RightPanel.set_tone_intent`、`PreviewView.set_image(flip_y=)` 死形参、
  `process_image` 里 `if ...: pass` 脚手架残留；`load_folder` 的假注解
  `-> ImportSession`（实为 Thread）与 `_import_done` 未用形参改诚实。
- **登记不修**：`docs/blockers.md` 新增 **B-4** —— 可写状态根
  `storage.REPO_ROOT = parents[2]` 只在仓库布局下成立（`pip install .` 算到
  `<venv>/Lib`，PyInstaller onedir 算到安装目录本身，`Program Files` 下
  用户确认标注会写失败）。改道属产品决定，不在审查轮里静默改。
- **自我推翻两条**（沿用交接 §5 的传统）：`slider_fine` 是 `sliderReleased`
  才 emit，不存在"拖动期间几十次 fsync"；`neutral_pick_gain/delta/intent`
  三兄弟是研究面 API（UI 走 `neutral_pick_intent_exact`），不是死代码。

## 结构整改清账（开发版，未发布）— 2026-09-08 Round 6

依据：`docs/交接_20260908.md` §12 未完成清单全部清账；13 笔提交（门禁卫生 →
原子落盘 → 表示归属 → 算法 → UI 线程纪律 → 死模块 → 测试文档 → 上述五项）。

- **导出输入快照冻结（交接 §4-1，最高危）**：整卷导出启动前在主线程一次性
  冻结全部逐帧输入（几何深拷贝、微调、意图查表、批次编码假设、卷统计），
  后台 worker 全程零读零写活对象；状态与采集指纹由主线程按 items 回写
  （`pipeline/roll_export.snapshot_frame / freeze_export_inputs /
  apply_status_writeback`）。回归锁：`test_export_snapshot_20260908.py`
  —— 导出中途改活对象几何/勾选片边，成品仍用启动前输入。
- **process_image 死参数清理（§4-2）**：`film_type` / `calibration`
  （34 形参中 AST 核实从未被读取的两个）随 15 处仪式性调用点一并移除；
  sidecar/RollCalibration/FilmProfile 的同名真形参不受影响。
- **反向依赖清零（§4-3 前半）**：`RenderIntent` 及色温/影调数学整体下沉
  `library/render.py`（`pipeline/render.py` 变兼容重导出壳，io/sidecar
  不再上行）；监督解混拟合下沉叶子 `metrics.py`；`calibration/neutral_stats`
  移入 pipeline（唯一 src 调用方是研发脚本）；冷读新发现 `GENERIC_PRIOR`
  在 roll_analysis 是**独立复制的字面量**而非别名，已改真别名。
  分层门禁 ALLOW 只剩登记在案的 `profiles/profile → pipeline.params`
  （放错目录而非倒置，决定不搬家）。
- **报告 schema 统一（§4-3 后半）**：batch 与 roll 报告共用核心键集
  （schema_version=2 / kind 判别 / summary / summary_complete / items
  最小契约）；roll-report 1→2。
- **MainWindow 拆分（§4-4）**：93 方法 / 1996 行 → 25 方法 / 541 行；
  StateStore（模块级持久状态辅助）、FrameLoader（导入/会话/帧加载/卷分析
  装配）、GeometryBridge（预览管线/即时合成/草稿读写/ROI 事件）、
  IntentController（候选/微调/拾点救援）、ExportCoordinator（整卷导出）
  各自成模块，方法原文搬移；`mw.process_image` monkeypatch 契约经
  `ui_process_image` 保持有效。顺手统一 `_abs` 最后一处 `parents[3]`。
- **C8 实测收口**：`test_c8_hold_orig_guard.py` 证明按住按钮期间
  `isDown()` 为真（守卫前提成立），且按住拒绝、松手恢复。
- **门禁**：分层测试 ALLOW 收紧；ruff `select=["E9","F"]` 无豁免维持；
  全量基线 427 passed（真实样片回归层含内）。

## v6.2 交互与标定闭环修复（开发版，未发布）— 2026-09-08

规格：`负片裁切与去色罩_AI执行规格_v6.2_交互与标定闭环修复.md`；
审查依据：`docs/审查_20260907_v6.2与救援实现核查.md`。

- **写入物理隔离（阶段 A）**：新增 `negative_lab/storage.py`，标注目录与 GUI
  草稿目录统一解析（显式改道 > 环境变量 > 仓库默认）；新增 `tests/conftest.py`
  会话级隔离，测试结束核对真实目录哈希未变，被污染即判失败。
  冒烟脚本此前只改标注目录、把草稿写进真实 `outputs/drafts/`（审计 F6）已封堵。
- **F1 冷启动**：帧加载先在后台解析持久状态，用户标定真正进入算法输入
  （此前 `ann=None` + 空几何，算法跑的是自动检测，界面显示的却是用户标定）。
- **F2 分析失效**：有效画面区编辑与片基取样区一样触发重算；输出裁切保持
  零管线快路径。
- **F3 三个坏按钮**：重新检测 / 候选 / 同步作用域不再调用不存在的
  `_set_banner_status` / `convert_panel`。
- **F4 状态选择**：新增 `calibration/state.py`（确认基线 vs 工作草稿六种组合，
  含内容指纹与草稿修订号）；草稿落盘携带 `base_annotation_sha` +
  `draft_revision`；旧草稿无来源指纹 -> 判冲突、确认基线生效、草稿原地保留。
- **F5 拾点坐标**：`_compose_view` 显式记录显示图在整幅技术正片中的原点
  （`_view_origin` / `_view_space`），拾点先换算再采样；更正研究笔记里
  "场景坐标零换算"的错误断言。
- **F9 阶段门控**：离开调整阶段清除拾点态；分析阶段点击拒绝按正片像素求解。
- **导出计数完整性**：`export_roll_frames` 增加 `cancelled` 计数并返回 `items`
  与 `summary_complete`；取消的帧不再静默消失（gui_smoke 曾因此 KeyError）。
- **一致性度量指向输出**：`roll_consistency.py` 落盘逐帧**实际施加**的
  dmax/temperature/tint 及散布（`applied_consistency`）；离群帧带文件名；
  NaN 一律转 null（此前产出不是合法 JSON）；卷名可显式指定、相对路径按仓库根解析。
- **测试可信度**：新增 `tests/unit/test_v6_2_fixes.py`（27 项，覆盖 F1-F5、F9、
  导出计数、存储优先级）；`test_rescue.py` 增加**整图对照参考**的两项断言——
  此前救援残差只在"被求解的那个 patch"上度量，属于自证（审查 §3.5）；
  新增会话级 QApplication 持有，修掉全量测试在 offscreen 下的非确定性崩溃。

实测证据：`outputs/v6_2/20260908-state-pick-fixes/`（一致性 5 帧实测：
de_mean 19.5、片基漂移 0.0057 decade、**逐帧 dmax 散布 0.241** —— 同卷片基
高度一致而软件逐帧施加的白点差 24%，正是过去没有指标暴露的问题）。

**仍未完成（不许谎报）**：真实 Windows 桌面人工流程验收（§8 阶段 D）、
人眼判定基准集（`decisions.json` 仍为空）、方向根因确认。

## v6.1 质量修复（开发版，未发布）— 2026-09 上旬

规格：`负片裁切与去色罩_AI执行规格_v6.1_质量修复.md`。
审计：`outputs/audit_20260907/v6.1_completion_audit.md`（0 致命 / 6 重要 / 2 轻微）。

- 技术正片与显示层分离：`keep_rebate`（保留片边）贯穿草稿、显示与导出；
- 右栏显示卷级片基一致性（实测帧色度极差，generic_prior 不参与统计）；
- 中性点单点救援（`pipeline/rescue.py`，(温度, 色调) 二自由度族内精确求解）
  接入画布拾点、撤销栈与可信度警告；
- 遗留：F1-F8 状态一致性问题（本轮 v6.2 处理其中 6 项 + 新增 F9）。

## v6 UI 优先（开发版，未发布）— 2026-09 上旬

规格：`负片裁切与去色罩_AI执行规格_v6_UI优先.md`（其 §5.6 部分条款已被用户作废）。

- 取消"裁切页 / 转换页"双页，改单工作区 **分析 / 调整** 两阶段
  （Lightroom 式 Convert/Edit）；分析阶段显示原负片、调整阶段显示缓存正片；
- 三类 ROI 同屏叠加编辑；片基变化才走慢路径重算，裁切与微调零管线调用；
- 三候选（保留氛围 / 中性 / 偏暖）+ 少量微调，风格只存在于 RenderIntent。

## v5 P0（开发版，未发布）— 裁切与无色卡去色罩

规格基线：`负片裁切与去色罩_AI执行规格_v5.md`（唯一生效基线，基线提交
`e795bbbf49a3ac7b782b2f569629adae97c31281`）。v1-v4 规格全部降级为历史资料。

### 产品范围（v5 §0.2 用户决策落地）

- **普通转换不需要色卡**：删除 `needs_calibration`/`needs_regions` 产品状态
  与全部"请提供色卡/必须加载校准/没有中性点"阻断文案（§8.1）；
- 黑白/反转片移出 P0，不再是发布门禁（§0.2.5）；
- C-41 与 ECN-2 同为 P0，且均有真实可解码样片（ECN-2 为 Z8 高效 NEF 的
  16 位 TIFF 导出，capture_type 如实标记 `camera_raw_export_tiff`）；
- 主工作区只有 **裁切 / 转换** 两个一级页面；Dmin/Dmax/矩阵/指数/校准/
  除尘等控件不再暴露给普通用户。

### 新增

- `library/`：`ImportSession`/`RollGroup`/`FrameRecord`/`GeometryRegions`/
  `RollAnalysis`/`CandidateSet` 六个可序列化对象 + 手动分卷（建立/重命名/
  拆分/移动）+ 状态机（§3.3，阻断只剩 decode_error/processing_error）；
- 自动采集指纹（§4.4）：不含文件名/卷名，metadata 不足时只降置信度；
  卷内冲突帧标记 + 一键拆卷建议；跨帧片基漂移检测（容差预先冻结）；
- 三类独立 ROI（§3.2）：`output_crop_roi` / `image_area_roi` /
  `base_sample_rois[]`（可多个）；标注 schema v3，旧 v1/v2 只读迁移并降级
  draft（用途必须重新确认）；
- 裁切页（§5）：图标工具栏、三类 ROI 的八向手柄/越界钳制/多片基区、
  细微旋转 ±10°、水平镜像、归一化相对位置的整卷同步；
- 卷级片基（§6.2）：多取样区密度域稳健估计（q75 中心 + decade 容差）、
  卷级汇总、无片基帧自动继承卷级模型、全卷失败回退**版本化通用先验**
  （全卷共用一个曝光尺度，非物理测量、如实标记）；
- 三个自动候选（§6.5-6.6）：保留氛围 / 中性 / 偏暖，共享同一技术正片；
  多估计器（SoG p=1/2/6、白点、Gray-Edge）方向与幅度比较、高饱和/单色/
  夜景降置信、分歧过大回退 preserve、色温/色调硬上限并记录触发；
- `RenderIntent` 基础微调（§7.2）：色温/色调（Bradford 色适应）、曝光、
  对比（受保护中间调支点）、高光/阴影/白色/黑色（非负斜率累积的单调
  影调曲线）、自然饱和度/饱和度；默认恒等逐比特不变技术正片；
  单项重置/全部重置/复制粘贴/勾选式整卷同步；
- sidecar **schema v4**（§3.4）+ v1-v3 只读兼容迁移；记录三类 ROI、
  采集指纹、片基来源与 provenance、三候选、微调、技术/渲染独立诊断、
  git commit 与规则版本；
- 整卷两遍处理与事务批量导出（§12 P0-7）：TIFF16 + JPEG8 + sidecar +
  `roll-report.json`，唯一命名、失败回滚、取消、源文件只读校验；
- CLI 子命令 `import/invert/batch/roll/profile`（`--base-sample` 可重复）；
  混合文件夹导入会话 JSON。

### 算法修正（真实样片实测教训）

- 片基污染判定改为**密度域**（|log10 t - log10 center|）：相对透射率容差
  在橙罩 B 通道（t≈0.03）上把正常颗粒误判成污染，真实卷 47-93% "污染"
  全是误判，曾导致整卷退化通用先验；
- 片基中心用 **q75**（未曝光片基物理上是条带最亮均匀区）+ 最亮 25% 兜底；
- 自动片基条带**收窄到最亮稳定子带**（`refine_base_roi`），避开齿孔/边码；
- flat-field 只保留**相对梯度**（片基取样区内每通道中位 = 1）：旧实现把
  窄条带二阶外推到全幅，实测把 Dmin 抬高 1.44 倍；
- 技术正片剪裁统计口径移到**软趾之后、最终 clamp 之前**（§8.2 字面），
  并按最终 image_area mask 逐通道统计；输入端剪裁降 warn（采集事实）；
- 通用模型资产（§6.3/§9.4）：无监督密度解混训练（82 帧、相关性 4.14→1.52、
  留出卷验证），资产哈希/单调性/条件数校验，失败自动回退确定性密度模型。

### 已知限制（诚实声明）

- 无真实扫描仪 TIFF / DNG / 第二套设备样片：扫描仪路线、DNG 路线、
  跨设备泛化一律未验证（docs/support-matrix.md）；
- ECN-2 证据来自相机 RAW 的 Lightroom 导出 TIFF（sRGB 显示编码线性化），
  不是扫描仪原生线性数据；
- 全部 169 帧基准的 ROI 仍为自动检测/历史草稿（`rois_user_confirmed=0`），
  需用户在裁切页确认后重生成验收清单；
- 逐卷联系表为预览分辨率（最长边 1200），最终验收以 GUI 全分辨率导出为准；
- 视觉验收（逐张接受/相对 NLP 判定）只能由用户签字，本版本交付时待定。

## v0.1.0（历史版本）

密度域 + 每通道幂指数混合模型；单张 GUI / 批处理 / CLI；Nikon Z8 高效
NEF（AV1）无法解码（LibRaw 0.22.1），需导出 16 位 TIFF——v5 已把该路线
纳入 P0 输入矩阵。
