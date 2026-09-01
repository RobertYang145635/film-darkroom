# 技术反相数学契约（v4 §4.1，唯一权威表述）

状态：生效。任何实现与本文冲突时以本文为准；修改必须提升
sidecar `schema_version` 并更新参数敏感性测试。

## 1. 符号约定（唯一：正密度表述）

生产 API、sidecar、文档只暴露以下符号；旧的负向 z=log10(t/b) 仅作为
等价推导出现在注释中，不得出现在接口命名或 sidecar 字段。

```text
b_c        未曝光片基（或 verified base ROI）的线性透射率，逐通道
d_c        正密度  d_c = -log10( max(t_c, eps) / max(b_c, eps) )   （画面内 d>=0 为常态）
u          密度域混合  u = A d + a      A: 3x3 解混矩阵；a: 密度域偏置（当前恒 0）
v_c        通道响应    v_c = s_c * u_c  s_c>0：逐通道斜率（技术参数）
q_c        技术曝光    q_c = -(wb_high_c / Dmax) * v_c + wb_high_c * offset * wb_low_c
technical  G(v, technical_endpoints) = clip( (1+black) - 10^{q_c}, 0, 1 )
```

与旧公式的等价性：旧 z = -d，旧 q = (wb_high/Dmax)·z + wb_high·offset·wb_low
代入即得上式——逐项等价，仅符号方向约定不同。

## 2. 强制语义

1. `b_c/Dmin` 只来自未曝光片基或 verified base ROI。禁止被场景分位数、
   齿孔、最亮物体改写（v3 F-ALG-3、v4 §4.1）。
2. `A` 只表达密度域混合。运行时应用的 A 必须与拟合/序列化的 A 完全一致；
   **禁止 backend 内部做未记录的行归一化、斜率重置等隐藏重参数化**
   （v4 §1.2.11 实锤）。若可辨识性需要把 A 分解为单位对角 × 通道尺度，
   分解后的 A 与 s_c 全部序列化，并有测试证明重组 = 拟合变换。
3. `f_c`（通道响应）必须单调。路线 C 的曲线节点受限并正则化；
   完整 3x3 A 与非线性曲线**不能**从单一片基或一条中性密度射线估出
   （可辨识性：中性样本在密度空间只张成一维射线）。
4. `technical_endpoints`（Dmax、offset、black）只做技术范围映射。
   曝光、对比度、高光、阴影、白色、黑色调性、饱和度、审美曲线一律属于
   RenderIntent（v4 §4.4）。
5. 每个保留参数必须有敏感性测试：改变参数 → 输出按预期变化。
   无效果或语义重叠的参数删除或迁移，禁止假旋钮（v4 §1.2.9/10 实锤：
   旧 power_reciprocal 计算了 k 却不用；exposure 全链路无效）。

## 3. Backend 路线（同一注册表、同一生产接口）

| 路线 | backend_name | 模型 | 用途 |
|---|---|---|---|
| A | `power-reciprocal`（旧兼容名） | A=I，s_c=逐通道斜率 | 可解释基线 |
| B | `density-affine` | verified 3x3 A + 明确 s_c | 一阶串扰 |
| C | `density-affine-curves` | B + 受约束单调通道曲线 | 染料非线性（优先候选） |

- `invert()` 与 `invert_chunked()` 必须经同一注册表执行；chunked 只负责
  分块与诊断聚合，每块调用与非分块相同的核心。
- bake-off（§4.2）完成前，任何路线不得自称"彩色负片默认"；
  `power-reciprocal` 只是明确选择的旧兼容对照。
- 未知 backend、非 3x3 / 非有限 / 病态矩阵必须在处理前失败（可读错误）。

## 4. 已废止的隐藏行为（回归锁）

- backend 内 `k` 计算后不使用（red/blue_ratio 无效）→ 已重写，s_c 真实生效。
- `exposure` 在技术链路无效 → 已迁移 RenderIntent.exposure。
- density_affine 行归一化 + 斜率重置的隐藏重参数化 → 已删除。
- Lab 计算先加 TRC 再乘线性矩阵（L*[0.5]=88.70）→ colorimetry.py 唯一实现，
  L*[0.5]=76.069 测试锁定。
