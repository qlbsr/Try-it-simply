# nsjy 算法全流程分析日志（DeepSeek 改进记录）

> 项目：`C:\Users\23128\My project (2)\Assets\Scripts`
> 日期：2026-08-29 ~ 2026-09-01（持续更新）
> 内容：整套 nsjy 家族代码的数学本质分析、实验验证过程、DeepSeek 逐项改进清单、剩余问题、参考文献与相关项目清单。
> **重点：§8 是函数 ↔ 神经网络对应表（不可微环节 → 网络模块），§10 是仓库最终形态与废物核查。**

---

## 0. 文件清单

### C# 主算法文件
| 文件 | 内容 |
|---|---|
| `nsjy.cs` | 基础算法库：`InverseTh4`/`yzqx`（圆锥坐标变换）、`ComputeRp`、`ComputeTaus`（CM 模量）、`Carlsonfk`（RF/RD/K/E/DK/NK 复数椭圆积分）、`ExtractFoci`/`FitFocus`、`FitFociByProbability`/`BatchProbability`、`pca`、`LatticeAnalysis`（格基/周期矩阵）、`LLL`、`ComplexLinearFit`、`LatticeRansac`、`DeviationCalculator`（最近格点概率场） |
| `n2sjy.cs` | 闭环优化 `RefineModuliByAxis`（LM + 数值雅可比 + NormalizeTau） |
| `n2sjy2.cs` | 双角度外环：`angleDeg`(与PCA夹角)/`angleDeg1`(与初始焦点方向夹角)，停止条件 `\|angleDeg−angleDeg1\|/2≤5` |
| `sjysjy.cs` | 双锥面 → 叶片展开：`MapDoubleConeToLeaf`、`SolveC3`、`SolveTFromE`；**新增 `Elliptic` 类**（Carlson RF/RD 的 F/E + 准周期约化） |
| `v2sjy.cs` | **透镜体积闭式 + ABCD 四通道**：`LensVolumeSimple`、`ComputeAllFeatures`、`NappePair`、`LocalQuadHessian`、`StiffnessProjection`、`FitChannels`、`Delta`/`DeltaIsRational`/`RatioFromDelta`/`TauFromDelta`；**零第三方依赖** |
| `legacy/sjy.cs` | 旧 RBF 全流程。本轮把 `rbf` 函数体换成**叶透镜链**（`ComputeLeafLens`→`th4`→`LocalQuadHessian`→`StiffnessProjectionFromH`），旧 RBF 体保留为 `rbf_RBF` 供对照 |
| `legacy/sjy1.cs` / `sjy11.cs` / `sjy112.cs` / `xs.cs` | 球面 RBF 旁支、JSON 解析、UI；纯 Unity 依赖，不参与数值验证 |
| `Assets/car.cs` | 与算法无关的演示脚本 |

> 已删除：`DLL.NET/`（5 个旧 DLL，已并入 `Assets/Plugins/`）、`gsjy.cs`、`nsjy3.cs`/`nsjy4.cs`/`nsjy5.cs`（思路已进入 `n2sjy2.cs` 与 `ebf_algorithm.py`）。

### Python（移植/验证）

- **顶层 8 个**：`nn_dn_sx.py`（主线，见 §8.2）、`nsjy_algorithms.py`（全量移植）、`n2sjy2.py`、`data_driven_axis.py`、`ebf_algorithm.py`、`analyze_relationship.py`、`fused_n2sjy2_nsjy4.py`、`combine_n2sjy2_nsjy4.py`
- **`experiments/` 16 个**：4 个互相 import 的复刻库（`verify_v2_provenance/perpoint/route/simplified.py`）+ 12 个结论脚本（见 `python/README.md`）
- **`archive/` 138 个**：历史一次性脚本，归档不再维护

---

## 1. 数学本质（一句话）

从 3D 点云出发，恢复一个复数环面（阿贝尔曲面/椭圆曲线雅可比）的模量 (t1,t2)，
并建立与点云几何（焦点椭圆族、PCA 轴）的自洽联系。

```
点云 → InverseTh4/yzqx(圆锥展开, 类 cut-and-project) → 复平面(r30,r45,r74)
     → 最近格点距离 → 概率场(probTotal/prob1/prob2)
     → ExtractFoci/FitFocus → 焦点 F1,F2 → 方向 d=(F1−F2)/|F1−F2|
     → 闭环: d 与 PCA 轴 v / 初始方向 d0 的夹角条件 |angleDeg−angleDeg1|≤10°
     → 输出: (t1,t2) 模量 (决定格结构/阿贝尔曲面)
```

---

## 2. 分析过程时间线（各阶段结论）

### 阶段 A：初版问题（概率拟合偏差 0.1~0.2）
- **A1**：`FitFociByProbability` 收敛后每点概率差 0.1~0.2。
- **A2 结论**：① `BatchProbability` 的 `-abs(delta)/2*a` 运算符优先级 bug（应为 `-(abs(delta))/(2*a)`），指数被放大 a² 倍；② 根本失配：`probTotal` 来自变换域格距离，`probFoci` 来自 3D 焦点距离，两种几何系统性偏差（地板）。

### 阶段 B：闭环设计（t1,t2 主参数）
- 流程：`(t1,t2) → 格概率 → ExtractFoci → (F1,F2) → 方向闭合 (F1−F2)∥PCA`。
- 用 LM + 数值雅可比；问题：最近格点距离分段常数 → 梯度≈0 → LM 卡平台。

### 阶段 C：收敛失败机理
- 卡死原因：方向项 cost 占 99% 但无梯度；`NormalizeTau` 模跳变（τ→−1/τ 旋转格）造成假恶化（gauss2: 13.5°→54°；ball0: 30.6°→52.6°）；退化格逃逸（Im τ→0）。

### 阶段 D：全流程 Python 移植
- 生成 `nsjy_algorithms.py`（867 行），修正 4 处 C# bug（见第 3 节）。

### 阶段 E：系列假设证伪（实验驱动）
- **固定 (t1,t2) 普适性**：P2 交叉测试 → 不存在（每个收敛 τ 只对自己数据集对齐）。
- **去掉 PCA 自洽环**：P3 → 自洽最优 ≠ 方向对齐（wDir=0 收敛到 30~75° 错方向）。
- **偏差正比关系**（|Δ| ∝ 距(0,1)/对称性）：E1/E2/E3 → ρ≈0、非单调、符号跨集翻转，证伪。
- **档2 自收敛**（固定值/固定范围）：0/8、0/8 → 概率环无吸引子。
- **两角相等收敛条件**：8 集最终 |Δ|=10~70°，ball2 先靠近再背离 → 不成立。
- **n2sjy2×nsjy4 互监督**：共识方向与 PCA 43~158° 不相关；迭代发散（单轮跳 129°）；跨通道(r74)共识最好 48°。

### 阶段 F：修正循环
- 重试起点 (0,1.2793)→(0,1)（标准模）；best-of-N 事后选择（只取更优）。
- 效果：gauss2 50.8→0.5、ball6 42.0→0.0、ball7 42.8→0.1、gauss4 2.3→0.0 等 6 例新收敛；34 集 → 档1=20、容错带=11、档2=3。

### 阶段 G：趋势机理（关键突破）
- **|Δ| = |angle(d,v) − angle(d,d0)| ≡ 单调等价 |d·(v−d0)|**（实测 corr 0.938~0.994）。
- 等角集 = 大圆 ⟂ (v−d0)；**解析解 d\* = normalize(v+d0) 处 |Δ|=0.00°**（6/6 精确）。
- 理想先决条件：angle(d0,v)≈0 时条件秒达成；现实 angle(d0,v)=55~125°（模型偏差）。
- 结论：趋势是"方向到等角大圆的 V 形漏斗"的几何现象，非 LM 提取对齐。

### 阶段 H：无梯度优化（nsjy4）
- Nelder-Mead/Powell 直接优化 |Δ|：117~330 次求值（约 5~12s）达 |Δ|≤10，多数精确 0.0°。
- 多起点：NM 5/6、Powell 6/6 数据集达标。
- 限制：PCA 在目标函数里（第二个参考），不能独立替代 PCA。

### 阶段 I：探索→最佳点→微调（nsjy5）
- 探索（冷启动 n2sjy2 漂移，无 PCA，best 跟踪 + 早停）→ 最佳点 → 微调（Nelder-Mead 兜底）。
- 成本优化：探索重拟合 `fit_foci_by_probability`→`extract_foci`（-100×），轮数 30。
- 效果：6/6 达标 |Δ|=0.00，约 64~94s/数据集（vs 旧 EBF 400s，vs nsjy4 单独 15s）。

### 阶段 J：神经网络整合构想
- 用 DEQ（Deep Equilibrium Model）把 n2sjy2 漂移（前向迭代）+ nsjy4 优化（隐式微分）合并为一个可微自迭代网络。
- 四步达到理想状态：平滑化（soft-min 代替 argmin）→ 偏差修正训练（合成已知轴教师）→ 收敛门（替代内外层预算）→ PCA 退化为训练标签。

### 阶段 K：因果分解——优化器 vs 椭圆曲线结构（判别实验，修正旧结论）
脚本：`experiments/analyze_optimizer_vs_structure.py`（Part A 纯几何 / Part B 网格穷举 / Part C 预算对比 / Part D LM）
- **K1 |Δ| 漏斗形态 = 方向空间几何**（Part A）：随机方向 v,d0,d（不经任何曲线计算）下同样成立：min|Δ|≈0.018°、bisector 解析零点 1.1e-13°、corr(|Δ|,|d·(v−d0)|)=0.981。→ 漏斗的"形状"与椭圆曲线无关。
- **K2 |Δ|=0 解点由结构提供**（Part B）：在 (t1,t2) 基本域穷举 10⁴ 格点（无优化器），min|Δ|=0.01~0.06°（4/4 集）→ 等角大圆（|Δ|=0 点集）确实落在映射 (t1,t2)→d 的可达域内；但盆地占比仅 1~5%（漏斗很窄）。
- **K3 修正：angle(d,v)≤10° 可达**（Part B）：网格扫描 min∠(d,v) = 8.21°(cube0)、**4.74°(ellip0)**、9.35°(ball0)、17.23°(ball1) → **3/4 数据集 ≤10°**。旧结论"min 31.7°、0/7 不可达"是 **NM 单起点被困的假象**（Part C 证实：NM 单起点 ∠v = 57/40/77/31.7°，而同一结构下随机 4096 点+NM 抛光达 0.0°）。
- **K4 找到解需要全局搜索**（Part C）：NM 单起点 |Δ| 停在 51~110°，随机采样 min|Δ|=0.00~0.49° → 随机最优+NM 抛光 ∠(d,v)=0.0°（cube0/ball0/ball1），ellip0=14.9°（随机未命中 4.74° 的窄盆地，网格已证存在）。→ 单起点 NM 不是好全局搜索器，但是好抛光器。
- **K5 LM 失败 = 离散目标函数**（Part D）：LM 从网格最优出发最大步长≈0（最近格点距离的 min 操作 → 分段常数 → 梯度≈0）；NM 无梯度故可行。→ 优化器类型的选择由目标函数光滑性决定，非结构限制。
- **K6 结论**：① 漏斗形态=纯几何；② |Δ|=0 解点的**存在性**=椭圆曲线结构（可达域覆盖等角大圆）；③ 找到解=全局搜索+抛光（单起点 NM 会困）；④ 模型偏差地板（0.58~0.62）只影响概率场拟合精度，**不阻碍方向对齐**——旧日志把地板当作"方向不可达根因"属误读，特此修正。

### 阶段 L：概率条件替代角度终止条件（脱离 PCA 的判据，用户 glxzf 发现验证）
脚本：`experiments/analyze_prob_angle_scan.py`、`analyze_prob_pairs.py`、`analyze_prob_condition.py`、`analyze_glxzf.py`、`analyze_condition_formula.py`
- **L1 概率差与角度差的强关联（gaji.py 语义）**：p1=P(理论焦点对)、p2=P((0,1)焦点对)、p3=理论格点总概率、p4=(0,1)格点总概率，angle=∠(v1理论焦点方向, v3(0,1)焦点方向)。实测：
  - `corr(d12, d34)=+0.984`（d12=(p2−p1)×100，d34=(p4−p3)×100），**同号占比 62.7%**，|d34/d12| 中位 **2.83** → d34 ≈ 2.83·d12；
  - |d34| 随角度**单调递增**：4.31(9.8°)→4.74(22.5°)→5.70(36.8°)→7.29(58.7°)→7.67(82°)→10.05(136°)→12.30(170°)，斜率≈0.05/°；
  - corr(∠,|d34|) 在 (90,164) 区间最强 +0.168。
- **L2 glxzf 164-180 公式验证**：`d1=max(s1,s2)*100−(jd−164)`、`d2=max(s3,s4)*100+(jd−164)`、`dd=(d1+d2)/2`：
  - `corr(d1−d2, jd)=−0.853`（公式结构正确）；dd 均值 72.6、std 3.25、范围 [63.1, 81.8]。
- **L3 反推角度精度**：全局线性 `angle≈a·d34+b` 残差均值 47.5°（corr 0.435）；**分段线性（16/30/45/74 区间）残差降到 9.0°** → 用户的分区间思路正确，每个区间内概率与角度近似线性。
- **L4 终止条件替代（sweep 标定）**：沿 v1→v 大圆扫描 d，`|s1−s5|×100 ≤ 0.5 ⟺ |Δ|≤16` 一致率 **88%**；`|s3−s5|×100 ≤ 0.1` 一致率 80%。但注意 rot=0（d=v1）时 |s1−s5|=0 会误判收敛 → 纯单点概率差不足以完全替代，需结合方向变化或其他参考。
- **L5 结论**：① 概率差（尤其 |d34|）与参考夹角**强关联且单调**，是脱离 PCA 判据的可行基础；② glxzf 分区间公式结构经数据验证正确（corr −0.853）；③ 建议条件式：`angle_ref ≈ 0.05·|d34| + offset`（或分段公式），终止判据用 `|s1−s5|·100 < T1 且 |s3−s5|·100 < T2`，阈值需在迭代轨迹上标定；④ 纯单点概率差替代角度有 80~88% 一致率，边界情况（d≈v1 起点）需额外保护。

### 阶段 M：概率终止条件的实际测试（结论：**不替换**，记录原因）
脚本：`experiments/test_prob_stop.py`、`analyze_d34_region.py`、sweep 比值测试
- **M1 全迭代测试（pyjson 真实 200 点，T1=T2=0.5）**：完整用户流程（vj 预精化+LM+旋转+NM）跑 30 轮，角度条件 iter2 触发（|Δ|=13.7°），**概率条件全程未触发**（|s1−s5| 最小 0.94、|s3−s5| 最小 0.22，但从不**同时** <0.5）。ellip0 上 |s1−s5|≈7.5、|s3−s5|≈20.5 恒定，完全不可能触发。
- **M2 失效根因**：① **概率场对方向非单峰**——sweep 显示 P(v5) 在 ∠(v5,v1)≈29° 处有峰值（0.830），故 |s1−s5| 随角度先升后降，非单调 → 无法设定单一阈值；② **起点歧义**：d≈v1 时 |s1−s5|≈0 会误判收敛（实际 |Δ|=107.8°）；③ **尺度敏感**：|s_i−s_j| 绝对值依赖数据（rp/a/分布），固定阈值无普适性。
- **M3 比值指标（s5/s1、s5/s3）**：同样非单调（1.0→1.068→0.885），失效。
- **M4 |d34| 角度区间指示**：均值随角度递增（43°→130°），但**每个 |d34| 箱内角度范围极宽**（[0,2) 箱跨度 1.6°~176.5°）→ 只能作"高角/低角"粗指示，不能窄区间锁定角度。
- **M5 最终结论**：**单点概率差/比值无法可靠替代角度终止条件**（非单调 + 起点歧义 + 尺度敏感）。`(angleDeg+angleDeg1)/2−min ≤ 8`（|Δ|≤16）保留。概率差（d34/d12）作为**不依赖 PCA 的参考角粗指示**仍有价值（判断高角/低角区间），可用于引导而非终止。
- **M6 可验证的替代方向**（若坚持脱离 PCA）：① 用**多点统计**（全部 200 点的概率均值/分布）而非单点 s[0]——单点噪声大；② 用**概率比值的极值检测**（s5 相对 s1、s3 的谷值）而非阈值；③ 训练一个小的回归器：输入 s1..s5 → 输出 |Δ|（数据驱动，L1 已证输入与角度强关联）；④ 把角度条件改为"**角度区间粗判**"（用 d34 判断是否落入 (90,164) 等区间）+ 区间内仍用角度精判。

### 阶段 N：三角形方法（用户洞察，**验证成功**）——概率 → p0 视角夹角反演
脚本：`experiments/analyze_triangle_inverse.py`、`analyze_triangle_loc.py`、`analyze_triangle_two_cone.py`
- **N1 核心几何（用户洞察）**：`BatchProbability(v·c, −v·c)[0]` 只涉及第一个点 p0 到焦点对 ±v·c：
  ```
  d1 = |p0 − v·c|,  d2 = |p0 + v·c|,  |v·c − (−v·c)| = 2c (恒定!)
  → 三角形 △(p0, v·c, −v·c) 共享顶点 p0, 另两顶点在半径 c 球面上
  → d1+d2 只依赖 θ=∠(p0,v) (余弦定理, |p0|, c 固定)
  → s_v = exp(−|d1+d2−2a|/(2a)) 是 θ 的单调函数, 可反演
  ```
- **N2 反演验证（pyjson）**：`s → θ 候选 {θ0, 180−θ0}`（双解，两解相加=180°）：
  - v1: s=0.77672 → {35.5°, 144.5°}（真值 144.5° ✓）；v3: s=0.82624 → {21.2°, 158.8°}（真值 158.8° ✓）；
  - p0 vs v5: {30.9°, 149.1°}（真值 149.1° ✓）；p1 vs v1/v3/v5 同样全部正确。
- **N3 双解消除（双参考定位）**：锥面 θ5（绕 p0̂）与锥面（绕 v1）相交 → 至多 2 个方向；再用 v3 或第三点选唯一。实测：测试 v5 的 θ=153.2° 候选 φ=−14.4° 重建 ∠(v5,v3)=14.8°（真值 14.8°，**误差 0.0°**），错误候选无解/误差 12.1° 自动排除。
- **N4 意义**：v1=(F1−F2)、v3=(F01−F02)、p0、p1 均**与 PCA 无关** → 三角形方法提供**完全脱离 PCA 的方向反演**：任意方向 v 的"p0 视角夹角"可由概率精确反演（双解），多参考可定位 v5 完整方向。
- **N5 与 M 阶段的关系**：M 阶段"|s1−s5| 非单调"的根因正是**没有用反演**——直接比较概率差（非单调）vs 先反演出角度再比较（单调且精确）。**M5 结论修正**：概率差本身不可直接替代终止条件，但**反演后的角度**可以。

### 阶段 O：三角形方法能否替代 PCA？——**系统验证：不能**（记录完整证据）
脚本：`experiments/verify_triangle_pca_free.py`、`verify_triangle_honest.py`、`verify_triangle_replace.py`、`verify_plane_pca.py`、`verify_prob_field_pca.py`
- **O1 无监督定位失败**：8 点反演 + 6 点投票定位 v5，**得票最高候选误差 179.9°（反方向），真解得 0 票**。根因：s(θ)=s(180−θ) 对称双解 → v 与 −v 对所有点完全对称，投票无法区分。（早前 `verify_triangle_pca_free.py` 报"0.05° 误差"是**用真值夹角筛选候选**，非独立定位——修正记录）
- **O2 平面法向场自洽但不含主轴**：n_i = p_i×v → v ⊥ {n_i}，已知 v 时法向场可重建 v（±1，最小特征值 0.000）；但**该信息只能重建"生成它的方向"本身**。对已知焦点方向 v1/v3 的平面族，∠(v1,v_pca)=107.8°、∠(v3,v_pca)=80.7° → **主轴不在任何焦点方向的三角形平面族内**。
- **O3 概率场统计量扫描失败**：球面 400 方向扫描，mean/std/entropy/max/sharp 五种统计量的极值方向与 PCA 主轴夹角 **51.3°~125.8°**，无一接近。
- **O4 数学结论**：angleDeg=∠(d,v) 需要 v 本身（数据主轴坐标）。三角形只反演"点与方向的夹角"（双解无符号），**无法从已知焦点方向推出主轴 v**——主轴是数据协方差的统计量，几何反演不能绕过它。替代条件 |∠(d,v1)−∠(d,v3)|（无 PCA）与 |∠(d,v)−∠(d,v1)| 语义不等价（v1,v3 夹角仅 27°）。
- **O5 可用的折中**（若接受初始化一次 PCA）：初始化算 v 一次，循环内用三角形反演 ∠(d,v1) + 预计算的 ∠(v,v1) 拼出 |∠(d,v)−∠(d,v1)| → **循环内零 PCA**。完全无 PCA 的替代主轴需改变条件语义，用户需确认。

### 阶段 P：三角形基本性质探索（用户方向：边长/内角/质心/高/PCA 原理）
脚本：`experiments/explore_triangle_basic.py`、`explore_triangle_pca_principle.py`、`verify_six_fields.py`、`verify_alt_reference.py`
- **P1 三角形族不变量（严格成立，5 方向验证）**：对 △(p0, v·c, −v·c)：
  - `a²+b² = 2(|p0|²+c²) = 8.924` 与 v **无关**（恒等）；质心 `G=(p0+v·c−v·c)/3 = p0/3` 与 v **无关**；内角和恒 180°。
  - **只有顶点角 ∠A(p0)（111°~136°）和高 h=|p0×v|（0.43~1.18）随 v 变化**——高 h 是三角形族中唯一携带方向信息的量。
- **P2 h(d) 不能作终止条件**：h(d)=|p0×d| 只含 p0 与 d，不含 v1/v3，扫描中 |Δ|≤16 时 h∈[0.53,0.71]，但最优区间一致率仅 **87.8%**（单点 p0 信息量不足）。
- **P3 PCA 原理 × 高场（解析等价）**：Σh_i²(v) = Σ|p_i|² − vᵀ(PᵀP)v → 最小化 ⟺ 最大化 vᵀ(PᵀP)v ⟺ **协方差最大特征向量 = 主轴**。5000 方向细扫：Σh² 最小方向与协方差主轴夹角 **1.3°**（反号 178.7°）。→ **高场最小化 ≡ PCA（数学等价，换算法而非替代）**。
- **P4 重要修正：`m.pca` 的 v3 重建 ≠ 协方差主轴**！pyjson 上 ∠(m.pca pc1, 协方差主轴)=**40.4°**——已知的 pca() v3 重建问题再次确认。真正的数据主轴是 C=PᵀP 的最大特征向量。
- **P5 三角形几何完全自洽（无符号闭环）**：从概率 s 反推 d1+d2=D → 边长 a,b → 海伦面积 → 高 h → cos²θ_i = (|p_i|²−h²)/|p_i|²。**cos² 消掉 ± 双解歧义**，反演 cos²θ_i(v1) 与真值**精确一致（5 位小数，5 点全对）**——纯三角形几何闭环成立，但这是"验证给定方向的几何"，不能定位未知主轴。
- **P6 阶段结论**：三角形基本性质探索确认——① 不变量（a²+b²、质心）与方向无关，可作对照基；② 高场定位主轴 ≡ PCA（等价）；③ 三角形反演 cos² 精确自洽（无符号）；④ **但没有找到"无 PCA 的独立终止条件"**——∠(d,v) 需要主轴坐标，主轴只能由协方差（或等价的高场最小化）获得，无法从固定三角形族推出。记录排除。

### 阶段 Q：s0[0]≈s5[0] 规律验证（用户发现）与不等式反例
脚本：`experiments/verify_s0_s5_relation.py`、`verify_s0_s2_s4_inequality.py`
- **Q1 规律成立（部分，后被 T/U 阶段修正）**：`|(angleDeg2+angleDeg1)/2−min| ≤ 8`（∠(d,v1)≈∠(d,v3)，无 PCA）达成时 s0[0]≈s5[0]：pyjson |s0−s5|=1.56（iter1）、ball0 0.87（iter3）"强成立"；ellip0 6.60（iter6）中等。**⚠️ 注意：pyjson 是未达成数据集，其 iter1 是中间态；真正达成点 [0] 的偏差实际很大（7.9~23）——Q1 的"[0] 强成立"是采样误导，见阶段 U 修正。**
- **Q2 反例（ellip0 达成点 iter6）**：用户预期 `|s2[0]−s0[0]| > |s0[0]−s5[0]|` 且 `|s4[0]−s0[0]| > |s0[0]−s5[0]|`。实测：s0=71.127、s2=70.571（|s2−s0|=0.556）、s4=82.747（|s4−s0|=11.621）、s5=64.523（|s0−s5|=6.604）→ **|s2−s0|=0.556 < |s0−s5|=6.604，与预期相反**（只有 |s4−s0| 满足）。**theory=✗**。
- **Q3 反例成因**：s2（拟合焦点 v2 方向）与 s0（PCA 方向）本身几乎重合（s2≈s0），故 |s2−s0| 天然极小，不能作为 |s0−s5| 的"下限参照"。达成时 s5 确实远离 s2/s4（s5<s2 且 s5<s4 均为 True），但方向相反：**s5 是四个方向中概率最小者**（64.5 < 70.6 < 71.1 < 82.7），即达成时当前方向概率低于 PCA/拟合焦点方向。
- **Q4 ellip1 无达成点**：cond 停滞 35.2（NM 困于局部），s5≈85.9 恒定、|s0−s5|≈13.2；iter1 曾 |s0−s5|=0.33（theory 短暂 ✅）但 cond=37.6 未达成。→ 理论需在"达成"前提下才有意义，ellip1 不构成对达成时规律的检验，但记录了其停滞行为。
- **Q5 修正方向**：与其用 |s2−s0|、|s4−s0| 作参照，不如直接检查**达成点 s5 是否为极值**（最小或最大）——ellip0/pyjson/ball0 达成点 s5 均为四方向中最小。可验证"达成 ⟺ s5[0] 为 s1..s5 中的极值"作为候选判据。

### 阶段 R：格点总概率三值 probs12/probs012/probs12z 验证（含 ball0 反例）
脚本：`experiments/verify_s0_s2_s4_inequality.py` 扩展、多数据集迭代跟踪
- **R1 定义**：probs12[0][0]=理论 taus(t1,t2) 格点总概率 p0 值（固定）、probs012[0][0]=(0,1) taus 格点总概率 p0 值（固定）、probs12z[0][0]=迭代中当前 t1,t2 的格点总概率 p0 值（每轮更新）。
- **R2 ellip0 观察（用户提出）**：达成 iter3（cond=3.56）时 probs12z[0][0]=46.88 ≈ probs012[0][0]=45.80（差 1.08），远离 probs12=40.07（差 6.82）。轨迹：iter0 从 (0,1) 出发 probs12z=012（差 0）→ iter1/2 跑远（差 11.9/13.0）→ iter3 达成收敛回 012 附近。呼应用户注释掉的候选条件 `|probs012[0][0]−probs12z[0][0]|≤0.01`。
- **R3 多数据集验证**：
  | 数据集 | 达成 | probs12z | probs12 | probs012 | 靠近 |
  |---|---|---|---|---|---|
  | ellip0 | ✓ | 46.88 | 40.07 | 45.80 | 012 (Δ1.08) |
  | ball1 | ✓ | 65.95 | 41.43 | 65.57 | 012 (Δ0.38) |
  | **ball0** | ✓ | 47.75 | 49.60 | 63.82 | **12 (Δ1.85)** ← 反例 |
  | pyjson | ✗ | — | — | — | cond 13.6 未达成 |
  | ellip1 | ✗ | — | — | — | cond 36.3 未达成 |
- **R4 结论（反例记录）**：**"达成 ⟹ probs12z≈probs012" 不成立**——ball0 达成时 probs12z 靠近理论 probs12 而非 (0,1) 的 probs012。达成时 t1,t2 的收敛位置随数据集变化（有时回 (0,1)、有时回理论 taus）。单一格点总概率 p0 值不足以判定达成；但达成点 probs12z 总落在 probs12 与 probs012 之间或其一附近（离两端的最大距离受约束），该结构可继续探索（如 `min(|Δ12z−12|,|Δ12z−012|)` 是否有上界）。

### 阶段 S：无 PCA 终止条件 #2 的普遍性验证（10/13 达成，77%）
脚本：`experiments/verify_condition2_universal.py`（13 数据集 × 完整迭代）
- **S1 条件**：`|(angleDeg2+angleDeg1)/2 − min| ≤ 8`（⟺ |∠(d,v1)−∠(d,v3)|≤16），angleDeg1=∠(d,v1)、angleDeg2=∠(d,v3)，**v1/v3 均为焦点方向，无 PCA**。
- **S2 结果**：达成率 **10/13 (77%)**：
  | 类 | 达成 | 未达成 |
  |---|---|---|
  | ball0~3 | **4/4**（iter1~8，cond≤0.7）| — |
  | ellip0~3 | 3/4（iter0~3）| ellip1（停滞 36.4）|
  | cube0~2 | 2/3 | cube1（停滞 16.3）|
  | gauss5 | ✅（iter4，cond=0.01）| — |
  | pyjson | ✗ | 停滞 13.4 |
- **S3 未达成根因 = 优化器停滞**（非判据缺陷）：3 例 cond 停滞在 13.4/36.4/16.3，与阶段 K 的"NM 单起点困局"一致——判据本身在能收敛时都精确（多数 cond<1，甚至 0.01~0.7）。
- **S4 与含 PCA 条件的对比**：原 `|∠(d,v)−∠(d,v1)|≤16` 在 pyjson iter2 达成，而 #2 在 pyjson 未达成 → **两者不等价**（#2 用 v1/v3 不含 PCA 轴 v，收敛盆地不同）。
- **S5 结论**：#2 是**有效的无 PCA 判据**（77% 达成，达成时精确），普遍性受限于 NM 单起点停滞；改进方向=多起点/全局搜索（阶段 K 结论），而非改判据。

### 阶段 T：规律是否对所有点 [i] 成立？（结论：仅 [0] 局部，非全数组通用）
脚本：`experiments/verify_index_general.py`（达成点全数组检查：s0/s2/s4/s5 与 probs12z）
- **T1 s5 为最小值的点占比仅 6%~30%**（ellip0 30%、ball0 6%、ball2 15%）——"达成时 s5 最小"只在少数点成立（含 [0]），非全体。
- **T2 |s0−s5|×100 的 [0] 值并非最小**：ellip0 [0]=7.89 vs [1]=1.88（中位 4.98）；ball0 [0]=11.39 vs [1]=2.99；ball2 [0]=23.0 vs [1]=0.54 → **[0] 偏差常大于中位数，[1] 反而更接近**——[0] 无特殊优势。
- **T3 全数组统计**：|s0−s5| 中位 3.6~5.0、P90 10.8~14.2、max 19.9~23.7 → 部分点近似相等，但远非"全部相等"。
- **T4 |probs12z−probs012| 同样非全数组**：[0]=1.08(ellip0)/16.07(ball0)，[1]=9.01/0.00，中位 2.0~3.4 → 收敛位置逐点不同。
- **T5 结论**：Q1/S 阶段观察到的 "s0[0]≈s5[0]、probs12z[0][0] 收敛" 是 **[0] 点的局部现象**（该点恰在某个稳定几何位置上），**不能推广到全部 200 点**。判据 #2（等角条件）本身基于方向 d 与 v1/v3 的夹角（全局量），不受此影响；但"用 s0≈s5 或 probs12z≈probs012 替代角度"需逐点验证，不能假设全数组成立。

### 阶段 U：逐索引普遍性统计——[0] 是最差观测点（修正 Q1）
脚本：`experiments/verify_index_universal.py`（9 个达成数据集，达成点逐索引 |s0[i]−s5[i]|×100）
- **U1 结果**（达成点 |s0−s5|×100<5 的数据集占比）：
  | 索引 | 占比 | 均值 |
  |---|---|---|
  | **[0]** | **0.0%** | **12.35** ← 最差 |
  | [1] | 66.7% | 4.27 |
  | [2] | 33.3% | 6.74 |
  | [3] | 55.6% | 5.97 |
  | [10] | 66.7% | 5.06 |
  | [150] | **77.8%** | **3.68** ← 最好 |
  | 全体点 | 57.8% | — |
- **U2 修正**：**Q1 的"[0] 强成立"是采样误导**——pyjson 是未达成数据集，其 iter1 |s0−s5|=1.56 是中间态巧合；9 个**真正达成**的数据集上 [0] 全部 >5（0% 达标，偏差 7.9~23.0）。[0] 是 200 点中最差的观测点，[1]/[150] 等普遍性更好（67~78%）。
- **U3 澄清 77% 的含义**：判据 #2 的数据集达成率 10/13（77%）是**方向角条件**，与任何 s[i] 索引无关；"s0≈s5"作为达成特征只在约 57.8% 的点成立（且逐索引差异大 0~78%），**不是通用判据**。

### 阶段 V：能否从 s5 反推近似 v 替代 pcav？（结论：不可行，s5 无 v 信息）
脚本：`experiments/verify_s5_invert_v.py`（达成点：s0/s5 全数组相似度 + s5→cos²θ 反演→最小二乘重建 w）
- **V1 s0/s5 仅"中等相似"**：达成点 corr = 0.588~0.609（3 数据集），中位 |Δs|×100 = 3.8~4.9 → 非"相近"。
- **V2 s5 反推必然回到 d 本身**：w vs d = **0.0°**（恒等）——s5=P(d·c) 只编码 d 的信息，反演（无论代数 cos² 或最小二乘）只能恢复生成它的方向 d，**不可能给出 v**。这是信息论必然：s5 不含 v 的信息。
- **V3 达成时 d 离 v 仍远**：d vs v3(用户 pca) = 51.5~117.3°，d vs 真协方差主轴 = 27.7~84.1° → 判据 #2（v1/v3 等角）达成**不使 d 靠近 PCA**，故 s0≈s5 根本到不了"能反推 v"的程度。
- **V4 数学本质**：s(v)[i] 依赖 (p̂_i·v)²；s0≈s5 全数组 ⟹ |p̂_i·v|≈|p̂_i·d| ∀i ⟹ v≈±d（一般位置）。实测 corr 0.6 说明 |p̂_i·v| 与 |p̂_i·d| 差异仍大，v 无法从 s5 恢复。
- **V5 结论**：**从 s5 求近似 v 替代 pcav 不可行**。替代 pcav 的正路：① 阶段 O 折中（初始化算 v 一次）；② 直接用无 PCA 固定焦点方向 v1/v3 作 NM 参考（pcav 本可换成 v1/v3）。

### 阶段 W：probs12z 趋势对比（结论：跟随 probs012，与 s0/s5 无关）
脚本：`experiments/verify_probs12z_trend.py`（ball0/ball2/ellip0 逐轮 corr）
- **W1 逐轮结果**：corr(12z, s0) ≈ −0.05~−0.18（恒定，无趋近）；corr(12z, s5) ≈ −0.17~−0.28（ball）/ +0.28~0.30（ellip）；**corr(12z, probs012) = 0.85~1.00（高）**，从 iter0 的 1.000 缓慢降到达成时 0.85~0.92。
- **W2 结论**：probs12z（当前 t1,t2 的格点 probTotal 场）趋势**跟随同构造的 probs012**（(0,1) 格点场，迭代起点），与 s0（PCA 焦点场）/s5（当前方向焦点场）**几乎不相关**——两套系统（格点距离概率 vs 3D 焦点对概率）在达成时也不互相靠拢。probs12z 不携带 s0 或 s5 的方向信息（与 R/V 一致）。

### 阶段 X：s5 跟随哪个固定焦点场？（结论：跟随"当前 v5 最接近的方向"，不专门跟 s4）
脚本：`experiments/verify_s5_follows.py`（ball0/ball2/ellip0 逐轮 corr(s5,s_i) 与 ∠(v5,v_i)）
- **X1 ball 系**：v1~v4 方向彼此接近，达成时 corr(s5,s1..s4) 全部≈0.97（s1 0.975、s2 0.976、s3 0.969、s4 0.970），s0 仅 0.67——s5 与所有焦点场同高。
- **X2 ellip0（关键）**：轮次迁移 s5 从贴近 v1（corr 0.998, ∠3.8°）→ 贴近 v3（corr 0.972, ∠11.6°）→ 达成时 v5 在 v1/v3 等角处（∠71.9°），corr(s5,s1)=0.738、corr(s5,s3)=0.589 反而都降。
- **X3 结论**：s5 **不专门跟随 s4**（也不跟任何单个固定场）——corr(s5,s_i) 与 ∠(v5,v_i) 强关联（∠小→corr高），s5 始终反映"当前方向 v5 的位置"。达成条件 ∠(d,v1)≈∠(d,v3) 恰是 v5 在 v1/v3 之间等角，此时 s5 与 s1/s3 相近但非最高——s5 是独立的"当前状态场"（与 W2 中 probs12z 同理，是其焦点场族的对应物）。

### 阶段 Y：s0[0] 与 min(s2[0],s4[0],s5[0]) 拟合（结论：不可行，corr 0.49）
脚本：`experiments/fit_s0_vs_min.py`（9 数据集迭代全程 53 样本）
- **Y1 拟合**：corr(s0,min)=+0.491；s0≈0.358·min+43.7；|残差|均值 3.27（×100）；s0>min 占比 52.8%（各半）；s0−min 均值 −2.2、中位 +0.18（绕 0 波动）。
- **Y2 分箱非单调**：min∈[60,65)→s0 64.2、[70,75)→72.8、[75,80)→70.8、[80,85)→65.7 → s0 与 min 非简单函数。
- **Y3 达成点不稳定**：min 常由**固定的 s2 或 s4** 决定（ball2: min=s2=83.18 vs s0=64.23 差 19；cube2: min=s4=80.5 vs s0=68.3）；仅当 **s5 为 min** 时（ellip0/cube0）两者较近（差 7.7 仍不小）。
- **Y4 结论**：s0（PCA 焦点场）与 min(s2,s4,s5)（混合固定/当前场）**无可用函数关系**——与 V 一致：s2/s4/s5 不携带 PCA 信息，组合无法恢复 s0。拟合路线排除。

### 阶段 Z：加 s5 最小约束的变体测试（结论：严重降达成率 75%→17%）
脚本：`experiments/test_s5_min_variant.py`（12 数据集，A/B/C/D 四变体）
- **Z1 结果**：A(#2 原) 75%（9/12）；B(A+ang1,ang2≥16) 42%；C(B+s5[0] 最小于全部) **17%**（仅 ellip0/cube0）；D(B+s5≤min(s2,s4)) 17%。
- **Z2 两级削弱的成因**：① ang≥16 约束排除达成时 ∠(d,v1) 或 ∠(d,v3)<16° 的数据集（ball0/ball2/ellip3/cube2，A✅但B✗）——很多数据集收敛到贴近某参考，此约束过严；② s5 最小约束排除达成时 s5[0] 非最小的数据集（ball1/ellip2/cube1，B✅但C/D✗）。
- **Z3 结论**：**"s5 必须最小"不是达成特征**——[0] 点达成时 s5 几乎从不是最小（阶段 T/U：s5 最小占比仅 6~30%，[0] 是最差观测点），作门控会漏掉 83% 合法达成。**#2 原条件（75%）是最佳**；加任何 s5 极值约束只会更差。记录排除。

### 阶段 AA：10 夹角关系分析（n2sjy2.cs 的 angleDeg~10）
脚本：`experiments/analyze_10_angles.py` + 搭配对比（fnd 版 vs Fnd 版）
方向向量：d=f1−f2（理论拟合，迭代更新）、Fd=F1−F2（理论提取，固定）、F01d=F01−F02（(0,1)提取，固定）、fd01=f01−f02（(0,1)拟合，固定）、ffd=f1f−f2f（初始理论拟合，固定）、Fnd=F1_new−F2_new（LM精化提取，每轮）、fnd=f_1new−f_2new（LM后拟合，每轮）、v=PCA。
- **AA1 完全冗余组（|corr|=1.00）**：`ang(Fnd,Fd)↔ang(fnd,ffd)`、`ang(Fnd,F01d)↔ang(fnd,fd01)`、`ang(fnd,v)↔ang(Fnd,v)` → Fnd（提取）与 fnd（拟合）在 LM 后几乎同向，三对角各自等价；**实际独立角约 6 个**。
- **AA2 判别力排名**（达成 vs 未达成，34 轮样本）：ang(fnd,fd01)=angleDeg6 **1.30**、ang(Fnd,F01d)=angleDeg4 **1.26** 最高；ang(fnd,ffd)=angleDeg5 与 ang(Fnd,Fd)=angleDeg3 各 0.82；ang(d,v)=angleDeg 0.34；**ang(fnd,v)=angleDeg9 0.07、ang(Fnd,v)=angleDeg10 0.02 最差**（含 v 的角在达成/未达成时都≈100°，几乎无判别力）。
- **AA3 搭配对比实测**：fnd 版（angleDeg5/6，#2 当前所用）与 Fnd 版（angleDeg3/4）做等角条件，8 数据集达成/未达成**完全相同**（达成 4 个，cond 值相近 0.0~2.5）→ **任选一组即可，效果相同**（冗余）。
- **AA4 结论**：① 最优 = **等角条件 #2**（判别力最高的两角做差：|∠(fnd,ffd)−∠(fnd,fd01)|/2≤8），用户当前选择正确；② angleDeg3/4 与 5/6 完全等价，任选；③ **含 v 的角（angleDeg9/10）无判别力，勿用于判据**——再次确认无 PCA 方向（v1/v3 族）是最佳参考。

### 阶段 AB：angleDeg9 震荡诊断（结论：NM 过冲，best-so-far 是解）
脚本：`experiments/diag_ang9_damping.py`（ball2/ellip0 完整流程，ω=1.0 vs ω=0.5）
- **AB1 现象复现**：ball2 ω=1.0：ang9 63→13→37→40→6.6（弹回后到 6.6）；ellip0 ω=1.0：123.5→**8.4**(iter3)→132→95；ω=0.5：→**4.0**(iter9)→145→25→7.7——轨迹震荡，但**历史最小 best9 单调下降且达标**（ball2 best 8.3°、ellip0 best 8.4°/4.0°）。
- **AB2 震荡根源 = NM 在 (t1,t2) 空间的跳跃**：iter 间 ang9 从 8.4° 跳到 132°——NM 移到新盆地后，最近格点变化使焦点方向 v6 **不连续跳变**（阶段 C/D 分段常数问题）。**不是旋转步长**。
- **AB3 阻尼旋转（ω=0.5）无效甚至更差**：ball2 best 8.3°→40.7°（震荡在 t 更新里，不在旋转角度）。
- **AB4 方案 = best-so-far 跟踪**：ang9 的**包络最小单调收敛**（63→13→8.3 或 123→8.4→4.0），每轮记录 ang9 最小时的 (t1,t2,f1,f2)，best 连续 N 轮未改进即终止并恢复 best——标准过冲解法，与阶段 F 的 best-of-N 一致。C# 落地代码见分析（循环外 bestAng9/bestT1/bestT2/bestF1/bestF2 存档）。

### 阶段 AC：完全无 PCA 流程验证（结论：判据可无 PCA，主轴对齐不可）
脚本：`experiments/verify_pca_free_full.py`（10 数据集，全程不调用 pca()，预精化 vj 用 Fd+F01d、参考/判据全用焦点方向）
- **AC1 判据 #2 无 PCA 达成 5/10**：收敛性不受去掉 PCA 影响（剩余失败仍是 NM 停滞，与阶段 S 一致）→ **"脱离 PCA"的判据层面已达成**。
- **AC2 但 v6 与真协方差主轴对齐 ≤10° 仅 1/10**：达成点 v6 与主轴夹角 60~124°——**判据 #2 达成只保证 ∠(d,v1)≈∠(d,v3)（格点等角大圆），不保证 d 靠近主轴**。唯一例外 ball1（best 7.3°）是迭代**偶然路过**主轴，非收敛目标。
- **AC3 数学结论（最终）**：主轴 v 是 C=PᵀP 的特征向量（数据统计量）。焦点方向 v1/v3 被格点结构偏置，收敛到"格点等角大圆"而非"协方差主轴"。**无 PCA 几何量无法替代主轴坐标**——除非接受"方向=v1/v3 族"作为替代语义（阶段 O/V），或把 Σh² 最小化（=协方差，阶段 P）算作无 PCA 实现（数学等价）。**判据脱离 PCA ✓；主轴对齐 ≤10° 无 PCA 不可行 ✗（除非改变目标语义）**。

### 阶段 AD：突变条件分析（|angleDegodr−angleDegnew|>9 → 离散档位跳变）
脚本：`experiments/analyze_mutation_jumps.py`、`try_peak_stop.py`
- **AD1 三种停滞模式**：① ball2：突变=逃逸平台（iter49 |Δ| 突变后 iter50 a9 创新低 54.7→49.9）——跳档有益；② ellip0：angNew 从 141° 单调爬升到 57° 时 a9 单调降（109→70°），angNew 停滞→活性死亡→a9 卡 70°；③ pyjson：iter7 后进入周期轨道（angNew 恒 109°、a9 恒 166.9°、Δt=1.97/1.21 恒定）——t 每轮平移 ~格点周期，方向不变，永锁死。
- **AD2 收敛路径 = 离散爬档**：angNew 每轮 +2~3° 爬档（一档=一个格点配置），突变=换档；"一个一个角度试"的机理确认。
- **AD3 停止条件尝试 `|Δ|>9 且 a9<1` 失败**：|Δ| 峰值（iter0/7/49）处 a9 反而大（75~167°），a9 最小（50~113°）处 |Δ|=0——**两信号从不同时**；且 a9<1 从未达到（pyjson 完美解对真主轴 2.3°、对 v3 40.4°）。**正确停止信号 = |Δ|→0（档位稳定）+ a9 best-so-far 无改善**，非突变最大。

---

## 3. DeepSeek 改进清单（改了什么、为什么、效果）

| # | 改进 | 原因 | 效果 |
|---|---|---|---|
| 1 | `BatchProbability` 括号修复：`-abs(delta)/2*a` → `-(abs(delta))/(2*a)` | 运算符优先级 bug，指数被放大 a² | 概率值恢复正常量纲 |
| 2 | `LLL` 的 static `bstar` 列表累积 bug → 干净实现 | 每次调用重复 Add，结果污染 | LLL 可重复调用 |
| 3 | n2sjy2 iter-3 重启块 `angleDeg2` 误用旧 `dir` → `dir0` | 比较对象错误 | 重启判断正确 |
| 4 | `pca()` 额外返回 pc1 | v3 重建与 pc1 不一致（fj/d2 复用 y 分量） | 闭环可用真实主轴 |
| 5 | `solve4` 回代索引越界（`aug[i,i+1:]` 含增广列） | 移植时发现的 bug | 数值求解正确 |
| 6 | 初始 τ 不调用 `NormalizeTau`；重试起点 (0,1.2793)→(0,1)；best-of-N 事后选择 | 模跳变假恶化（gauss2 +37°、ball0 +22°） | 假恶化结构性消除；34 集档1 14→20 |
| 7 | 收敛判据从"两角相等"改为**方向稳定度**（\|d_{k+1}−d_k\|） | 两角相等在不动点处不成立（10~70°） | 判据可验证 |
| 8 | 两角条件改用 **Nelder-Mead 无梯度优化**（nsjy4） | LM 在分段常数平台无梯度 | 117~330 次求值达 \|Δ\|≤10，多数 0.0° |
| 9 | 探索重拟合换 `extract_foci`（nsjy5） | `fit_foci_by_probability` 每轮 ~500ms | 成本 -100×，6/6 达标 ~64~94s |
| 10 | 全量 Python 移植 + 20+ 验证脚本 | 快速实验验证 | 所有结论均有复现脚本 |
| 11 | `legacy/sjy.cs` 的 `rbf`（球面 RBF 刚度场）→ **叶透镜链** | RBF 核宽是超参、拟合是黑箱 | `ComputeLeafLens`→`th4`→`LocalQuadHessian`→`StiffnessProjectionFromH` 可编译可运行；旧 RBF 体保留为 `rbf_RBF` 供对照 |
| 12 | `v2sjy.cs` 新增 **ABCD 四通道**块 | 需要把"外部势梯度"变成可判别的量 | `FitChannels`/`Lstsq4`(MGS-QR)/`Delta`/`DeltaIsRational`（连分数精确终止）/`TauFromDelta`；常 Hessian 只进 A、B，实测 \|C\|=8.96e-17、\|D\|=2.55e-17；留一法漂移全在 1e-16 |
| 13 | 数值库全部回退 `Numerics.NET` → `MathNet.Numerics` | `Numerics.NET` 的 `Matrix<T>`/`Vector<T>` 抛 `LicenseException: Invalid trial key`（商业授权）；`Complex<T>`+`Special.*` 免授权但不够用 | 6 个文件共 ~92 处 API 回退（`Matrix.Zeros`→`Build.Dense`、`GetInverse`→`Inverse`、`GetSingularValueDecomposition`→`Svd`、`GetSlice`→`SubVector`、`Vector.Length`→`Count`）；`sjysjy.cs` 新增自有 `Elliptic` 类替代 `Special.EllipticE`/`Elliptic.EInc` |
| 14 | 仓库结构：删嵌套 `.git`、仓库根从 `Assets/Scripts` 上移到工程根、补根 `.gitignore` 与根 `README.md` | 仓库根过深 ⟹ `Assets/Plugins/`、`Assets/Resources/points.json`、`ProjectSettings/` 全在库外，clone 下来无法编译；`.git/.git` 嵌套使 GitHub Desktop 拒传 | clone 可编译；`Library/`(1.62GB)、`.vs/`(13.6MB) 等被挡；工作区 0 变更、`fsck` 通过 |
| 15 | `dn`/`sx` 改写为**可学习/可微模块**（`python/nn_dn_sx.py`） | `(int)` 截断 + 硬 `max` 不可导；实测 `rp` 虚高 ~100 倍、透镜体积 `V ≡ 0` | `soft_max` 可微上界、`soft_quantile` 隐式微分、`SoftSectors` von Mises 软分配、`Axis` so(3) 参数化；见 §8.2 |

---

## 4. 关键实验结果汇总

| 实验 | 结果 | 结论 |
|---|---|---|
| 概率偏差地板（52 集） | max\|probTotal−probFoci\| = 0.58~0.62（σ=0.011） | 模型常数，非数据属性；devTol=1e-4 死阈值；**只影响概率场拟合精度，不阻碍方向对齐**（阶段 K 修正） |
| 两角条件达标（nsjy5，6 集） | 6/6，\|Δ\|=0.00 | 快速达成条件 |
| **方向对齐 angle(d,v)≤10°**（阶段 K 修正） | 网格穷举 10⁴ 点：4.74°(ellip0)/8.21°(cube0)/9.35°(ball0)/17.23°(ball1)；随机+NM 抛光 0.0°（3/4 集） | **可达**。旧"min 31.7°、0/7"系 NM 单起点被困假象 |
| 34 集分类（修正后） | 档1=20、容错带=11、档2=3 | 框架自洽 |
| 趋势机理（6 集） | corr(\|Δ\|, \|d·(v−d0)\|)=0.94~0.99；normalize(v+d0) 处 \|Δ\|=0.00 | 等角大圆 V 形漏斗（纯几何，随机方向同样成立） |
| 理想先决条件 | angle(d0,v)=55~125°（mean 92°） | 初始方向差大，但可达域含好解（阶段 K） |
| 互监督共识 | ∠(d\*,v)=43~158°；迭代发散 | 同源偏差不抵消 |
| 档2 自收敛 | 0/8（固定值）、0/8（固定范围） | 概率环无吸引子 |

---

## 5. 剩余问题与开放方向

> ⚠️ 本节写于早期阶段。其中第 1 项（模型偏差）与第 3 项（泛化）已由后续阶段消解，
> **最新清单见文末 §9**；函数↔神经网络对应见 **§8**。

1. **模型偏差（0.6 地板）**：格距离 ≠ 3D 距离，只影响概率场拟合精度；方向对齐已证明可达（阶段 K），地板不再是方向问题的障碍。剩余影响：概率值本身的保真度。
2. **全局搜索策略**：单起点 NM 会困（51~110°），随机/网格+抛光可达 0.0° → 下一步是把"探索阶段随机化/多起点"正式并入闭环（nsjy5 的探索已部分做到）。
3. **泛化**：合成分布上验证，真实数据（pyjson）需复测。
4. **DEQ 网络化**：平滑化（soft-min）+ 合成轴教师训练 + 收敛门 + PCA 退化为教师。
5. **收敛门替代预算**：消除内层 30/50/100 预算敏感（不动点随步数跳 16~47°）。

---

## 6. 参考文献与相关项目清单

### 数学
- B. C. Carlson, *Numerical computation of real or complex elliptic integrals*, Numer. Algorithms (1995) —— `Carlsonfk` 出处
- DLMF 第 19/20/22 章（椭圆积分/函数/theta）
- D. Mumford, *Tata Lectures on Theta*（Birkhäuser）
- J.-I. Igusa, *Theta Functions*（Springer 1972）
- D. Cox, *Primes of the form x²+ny²*；J. Silverman, *Arithmetic of Elliptic Curves*（CM 椭圆曲线）
- A. K. Lenstra, H. W. Lenstra, L. Lovász, *Factoring polynomials with rational coefficients*, Math. Ann. (1982)；[LLL 历史综述](https://zbmath.org/?q=ci%3A0488.12001+ai%3Asmeets.ionica)
- Lagarias / Moody, *Meyer sets / cut-and-project*（准晶体）；[Kellendonk（概周期序）Zbl 0989.82033](https://zbmath.org/0989.82033)
- M. E. Tipping, C. M. Bishop, *Probabilistic Principal Component Analysis*, JRSS-B 61(3) (1999)；[Mixtures of PPCA（Neural Computation）](https://direct.mit.edu/neco/article-abstract/11/2/443/6238/)
- N. Lawrence, *Gaussian Process Latent Variable Models* (2005)

### AI / 机器学习
- S. Bai, J. Z. Kolter, V. Koltun, *Deep Equilibrium Models*（NeurIPS 2019，[poster](https://neurips.cc/virtual/2019/poster/14487)、[slides](https://dev.neurips.cc/media/neurips-2019/Slides/15737.pdf)）—— 自迭代网络（构想的核心框架）
- M. Andrychowicz et al., *Learning to learn by gradient descent by gradient descent*（NeurIPS 2016，[Semantic Scholar](https://www.semanticscholar.org/paper/Learning-to-learn-by-gradient-descent-by-gradient-Andrychowicz-Denil/71683e224ab91617950956b5005ed0439a733a71)）—— 学习型优化器
- *[Neural Lattice Reduction: A Self-Supervised Geometric Deep Learning Approach](https://ar5iv.labs.arxiv.org/html/2311.08170)*（TMLR 2023）—— 格归约的 DL 化
- *[Learning Euler Factors of Elliptic Curves](https://www.semanticscholar.org/paper/Learning-Euler-Factors-of-Elliptic-Curves-Babei-Charton/2b8345fa1c60bfb5a5e23850357e263e7f2b208c)*
- *[Murmurations, Mestre–Nagao sums, and CNNs for elliptic curves](https://github.com/yidiq7/murmurations-cnn)*（Yang-Hui He 团队）
- C. Qi et al., *PointNet*（CVPR 2017）—— 点云特征编码

### 工具/库
- [scipy.special.ellipk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipk.html)
- [elliptic-functions（NumPy/PyTorch/JAX）](https://pypi.org/project/elliptic-functions/4.0.0/) —— `Carlsonfk` 的现代等价
- [LLL 算法介绍（CECM/SFU）](http://wayback.cecm.sfu.ca/~aszanto/IntegerRelations/LLL.html)

---

## 7. AI 调用记录

- 本文件的"改进清单"（第 3 节）对应的每次 DeepSeek 调用的**提示词、输入/输出（精简）与改进映射**，
  见同目录 [`ai_call_log.md`](ai_call_log.md)（AI 调用与提示词日志）。

---

*本日志由 DeepSeek 在迭代分析过程中生成，所有结论均有 `verify_*.py` / `test_*.py` 脚本可复现。*

---

## 阶段 AE (Unity vs Python 分歧定位: n2sjy2 pyjson)

### Unity 终止轮事实 (用户确认)
- (a9=2.002618, new=111.2141, odr=70.66569) 出现在 **外轮 80-100 左右** (非 iter0)
- a9 = ∠(v6, v3), new = ∠(f1-f2, f01-f02) (每轮开头), odr = RefineModuliByAxis 首内迭代 angle

### 关键代码发现 (本次)
1. **n2sjy2.cs 类内自带 FitFociByProbability (line 405)**, 与 nsjy.cs (line 206) 完全不同:
   | 项 | n2sjy2.cs 405 (Unity 实际用) | nsjy.cs 206 |
   |---|---|---|
   | lr | 0.0001f | 0.001f |
   | lambdaSep | 1.0f (分离正则) | 无 |
   | 单点裁剪 | ClampMagnitude(1.0f) | 无 |
   | 整体裁剪 | ClampMagnitude(5f) | 无 |
   | d 下限 | safeD=max(d,1e-3) | d>1e-6 跳过 |
   Start/EvaluateResiduals/RefineTausWithNM 内全部无前缀调用 → 同类 405 版!
   Python n2sjy2.py fit_foci_by_probability (100-164) = 405 版移植 ✓
2. **m.pca v3 重建 bug 已修**: C# fj=atan2(z,x), Python 旧用 atan2(y,x) 偏 37.2°.
   修正后 v3 = pc1 主轴, pyjson 上 ∠(v3, 真轴)=4.61° (旧 v3 偏更多).
   (nsjy_algorithms.py pca 已修复)
3. ExtractFoci 正规方程 vs SVD 差异 ~1e-17, cond(AtA)=5.1 → 非分歧源.
4. 两套 compute_taus (mpmath/Carlsonfk) 数值相同 → 非分歧源.
5. DeviationCalculator = n2sjy2.cs 764 行, lattice rng=20, 与 Python 移植一致.

### Python n2版拟合 (405版忠实) iter0 结果
- f1-f2 vs v3 = 134°, f1-f2 vs d01((0,1)) = 28.0° (Unity new=111.2 差很大)
- d01((0,1)拟合) vs v3 = 109.6° ≈ Unity new 111.2 (巧合?)
- odr(首内angle) = 30.2 (Unity 70.7)
- 12 轮外环: new/odr 同步 +2°/轮 (crawl), |odr-new|≈2.1 恒小, a9 136→160 单调恶化
- Unity 终止于 80-100 轮 → Python 需跑 80+ 轮才能对照 (60轮作业进行中)

### 待解
- Unity 80-100 轮时 f1-f2≈v3 (a9=2) 而 Python 12 轮 f1-f2 远离 v3 (a9 160) —
  若 60-100 轮 Python 轨迹绕回则复现成功; 否则初始拟合仍有隐藏差异.
- 待 Unity 前几轮 (line 88 anglenew) 日志确认初始拟合方向.

## 阶段 AE 更新: 60轮复现成功 (决定性)

### 结果 (reproduce_cs_outer_faithful.py, n2版拟合+修正v3, pyjson 200点)
- iter0-39: a9 136→172° 单调恶化 (crawl, odr/new 同步 +~2°/轮, |Δ|≈2 恒小) ← 旧实验"卡125-167"真相=只跑<40轮
- iter40 跳变: a9 骤降 172→7.4° (方向翻转 escape)
- iter41: new 64→115 跳, |odr-new|=48.9>16 → 纯A条件触发 stop@41 (a9=6.8)
- &&a9<1 模式继续: odr→70.4, new→111.4 (59轮), a9 6.8→3.46 单调降, 趋势→80-100轮 a9≈2
- **Unity 终止读数 (odr=70.7, new=111.2, a9=2.0) @80-100轮 与 B 模式 59+ 轮状态逐位一致** ✓

### 结论
1. Unity vs Python 分歧=**轮次不足 + 旧 v3 bug (37° 参照偏)**; 算法本身已逐行对齐
2. 跳变(iter40)是 escape 机制 — 前 40 轮 crawl 后方向翻转跳回主轴附近
3. 终止条件实测:
   - A 纯 |odr-new|>16: iter41 跳变后触发, 停时 a9=6.8 (尚未到最优)
   - B &&a9<1: a9 最低 ~2>1 → **永不触发**, 跑满轮次 (Unity a9=2.0>1 同理不触发)
   - Unity @80-100 终止: 若纯A, 跳变应在~80轮后发生 (float32/NM细节差异致跳变轮次不同)

---

## 阶段 AF：数值库回退 `Numerics.NET` → `MathNet.Numerics`（决定）

**起因**：`Numerics.NET` 的 `Matrix<T>` / `Vector<T>` 在 `sjy.pca()` 处抛
`LicenseException: Invalid trial key` —— 该库为商业授权，试用 key 无效。
`Complex<T>` 与 `Special.*` 恰好免授权，所以早期只有部分代码报错，掩盖了问题的全局性。

**处理**：全部 `.cs` 回退到 MathNet.Numerics 5.0.0（MIT）。共 6 个文件 ~92 处：

| API（Numerics.NET） | API（MathNet） |
|---|---|
| `Matrix.Zeros<T>(r,c)` | `Matrix<T>.Build.Dense(r,c)` |
| `Vector.Zeros<T>(n)` | `Vector<T>.Build.Dense(n)` |
| `GetInverse()` | `Inverse()` |
| `GetSingularValueDecomposition()` | `Svd()` |
| `LeftSingularVectors` / `RightSingularVectors` / `SingularValues` | `U` / `VT` / `S` |
| `Vector<T>.Length` | `Count` |
| `GetSlice(0,N-1)` | `SubVector(0,N)` |
| `Complex.FromPolar` | `FromPolarCoordinates` |

**易错点**：`i` 的命名不同 —— `System.Numerics.Complex` 是 `ImaginaryOne`，
`Numerics.NET.Complex<T>` 是 `I`（且**没有** `ImaginaryOne`，静态字段只有 `I/One/Zero/NaN/Infinity`）。
回退时 12 处 `Complex.I` 必须改回 `Complex.ImaginaryOne`。

**`sjysjy.cs` 附带**：`Special.EllipticE` 被自有 `Elliptic` 类替代（K / E / FRaw / ERaw / F / EInc）。

**验证**：
- `mathnet_check` 编译全部 `.cs`（除 `xs.cs`/`sjy11.cs`）：0 error / 3 warning
- `mathnet_run` 端到端：`rbf_RBF` τ = +0.406314+0.039100i、`|ratio|` = 0.782180、j = 8.511e7、8735 ms；
  `rbf` τ = +0.433940+0.156416i、`|ratio|` = 0.374265、j = −1.253e7、5059 ms

---

## 阶段 AG：RBF 被叶透镜链替换

`legacy/sjy.cs` 的 `rbf()` 原本用球面 RBF 拟合一个刚度场。替换后的链条：

```
ComputeLeafLens       叶面两叶的透镜体积 V
  → th4               填 uvss（两叶坐标）
  → LocalQuadHessian  对 (u,v) 局部二次拟合 → Hessian H
  → AB                常 Hessian 只落进 A、B 两个通道
  → StiffnessProjectionFromH
```

旧 RBF 函数体原样保留为 `rbf_RBF`，便于 A/B 对照。
诊断量 `lastTau` / `lastJ` / `lastRatio` 保留。尾部逻辑抽成 `BuildABCD(stiffnessProj, AB_vals, quatGrad)`。

**结论：有了两个标量（叶透镜体积 V + 距离 d）后，不再需要 RBF 去算刚度矩阵。**
刚度矩阵就是能量的 Hessian，而叶透镜链已经把它解析地给出了。

---

## 阶段 AH：透镜体积闭式与 `g = d/r` 参数化

```
V = R⁴ · I_x(5/2, 1/2)
x = 1 − d²/(4R²) = 1 − g²/4
Phi(g) = [12α − 8 sin2α + sin4α] / (6π),   α = arccos(g/2)
```

**恒等式来源**：`(1/2)·I_{sin²α}((n+1)/2, 1/2)` 正是 n 维球冠的占比，
故指数 `(5/2, 1/2)` 对应 **n = 4**：

```
V = (4/π²) · Vol₄(cap)
```

**`d` 在 `x` 里精确抵消**：`x = 1 − d²/(4R²) = 1 − g²/4`，20 万样本上残差 9.1e-13。

**小角级数**（α < 0.05 时避免 12α − 8sin2α + sin4α 的灾难性相消）：
```
Phi(g) ≈ α⁵(6.4 − 3.0476α² + 0.7111α⁴) / (6π)
```
与正则化不完全 Beta 吻合到 **3.6e-11**。

**等价性**：`ComputeAllFeatures` 恒等于 `(d,g) → (I = d/g, V = (d/g)⁴ Φ(g))`，
验证 `|dI/I| = 0`、`|dV/V| ≤ 6.4e-12`。

### 关键教训：`g = d/r` 才是正确的无量纲参数

与 `g = ‖dJ/ds‖`（s 为锥面半径）一致。**叶片雅可比版的 `g` 在真实 uvs 上完全不可用**：

| 版本 | `g` 范围 | `x` 范围 | `V` |
|---|---|---|---|
| 叶片雅可比 `g`（错） | `[112, 11697]` | `[−3.42e7, −3160]` | `V ≡ 0` |
| `g = d/r`（对） | `[1.421276, 2.000000]` | `[6e-7, 0.495]` | 正常 |

真实 uvs 统计：`Length = 1600`、1 个 NaN、`|uv| ∈ [3.78e-5, 0.3329]`（3.94 dex）、
`d2 = 44.713528°`、`rp = 51.154373`、`v3 = (-0.0171, 0, 0.9999)`。

### 两叶几何 `NappePair`

```
A = qs · vs3,   B = qs2 · vs3
d = ‖B − A‖ = 2r√(1 − (x̂·n)²),   n = normalize(ẑ × v3)
r = rp · Rs^sin(d2),   φ = θ · sin(d2)
```
验证到 1.3e-15。

### 椭圆积分（`sjysjy.cs`）

Carlson 形式，与 scipy 吻合 4.3e-16（k ∈ [0,0.99]，φ ∈ [0,2π]）：
```
F(φ,k) = sinφ · R_F(cos²φ, 1 − k²sin²φ, 1)
E(φ,k) = sinφ · R_F − (k²/3) sin³φ · R_D
```
外加准周期约化 `F(φ+nπ) = 2nK + F`、`F(π−φ) = 2K − F`
（`SolveTFromE` 在 [0,2π] 上二分，**必需**这一步）。

---

## 阶段 AI：两叶几何与 `vss` 的零信息量（负面结果，必须保留）

| 命题 | 结论 | 证据 |
|---|---|---|
| `(points1, points2)` 携带逐点信息 | **否**，只是刚体旋转 | Kabsch 残差 1e-16（6 组随机 (v3,vss)）；`d` 跨 5 个 `vss` 与 5 个 `v3` + 真实 PCA `v3` 逐位相同（5.6e-16） |
| 弦长依赖 `vss` | **否** | `\|ch1\| = 2z2`、`\|ch2\| = z2√5`（因 `V_3 ⊥ v̂`） |
| 旧透镜场有逐点剖面 | **否**，常剖面，只由 `r[2]` 决定 | 故 `ExtractAngles` 不是"毁掉逐点信息"的元凶 —— 本来就没有 |
| `vss` 的作用 | **纯平移**第二条弦 | `V_q1 = v_c + vss`、`V_q2 = v_c1 + vss`（1.7e-16） |
| 抛物线内部点有信息 | **否**，解析冗余 | 100 点全满足 `u = zLength − (zLength/z2²)v²`，残差 1.8e-16；`ComputeBendEnergy` 只用 `start·e2`、`end·e2` |
| 旧 `d` 是几何距离 | **否**，索引配对 | `e2_1·e2_2 = −1.000000`；且 `d` 随 numPoints 变（100/200/400 → 0.506242/0.503698/0.502436） |
| 旧路线在真实 uvs 上有效 | **否**，退化 | `g ≡ 2.000001`（`vss = 0` ⟹ `points1` 塌到原点，`ds` = 半弧长 ⟹ `‖dJ/ds‖ = 2`） |

---

## 阶段 AJ：ABCD 四通道

**A、B、C、D 不是四个独立的物理实体**，也不是可单独调节的参数，
而是外部势梯度在复平面上低阶展开的四个复系数通道：

```
∂U/∂z = A z + B z̄ + C z² + D z̄² + 高阶项
```

常 Hessian 的投影是**严格线性**的：
```
stiffProj(z) = α z + β z̄
α = (H_uu + H_vv)/4
β = (H_uu − H_vv + 2i H_uv)/4
|α|² − |β|² = det(H)/4
```
⇒ **常数部分只落进 A、B，C = D = 0。**

`SelfTestChannels` 实测：`|A−α| = 1.11e-16`、`|B−β| = 7.85e-17`、`|C| = 8.96e-17`、`|D| = 2.55e-17`。

判别量与反演：
```
Δ = (B² − 4AC) / (D² − 4BC)
ratio = (1 + √Δ) / (1 − √Δ)
τ = log(ratio) / (2πi)
```

`DeltaIsRational` 用**连分数精确终止**作判据。注意区分：
- ✅ 正确：「展开成连分数后**有限步终止**」⟹ 有理数
- ❌ 错误：「在分母界内存在足够好的逼近」⟹ 会把 `√2` 误判为有理

`SelfTestChannels` 实测：`sqrt(2)` → False、`2/3` → True（2/3）、`0.5+0.25i` → False；
留一法漂移全在 1e-16 量级（尺度 8.958e-1）。

---

## 阶段 AK：仓库结构迁移

**根因**：仓库根是 `Assets/Scripts`（不是工程根），于是
`Assets/Plugins/MathNet.Numerics.dll`、`Assets/Plugins/Newtonsoft.Json.dll`、
`Assets/Resources/points.json`、`Packages/`、`ProjectSettings/` **全部在库外、永远不被跟踪**；
而 git 里仍登记着旧 `DLL.NET/` 的 11 个文件（显示为删除）⟹ clone 下来无法编译。

三层嵌套仓库：`Assets/Scripts/.git`（真）→ `.git/.git`（2.52 MB 完整多余仓库，
内含 `Simple-simulation` 检出）→ `.git/.git/Simple-simulation/.git`（第三层）。
删前已验证：无 `objects/info/alternates`、`config` 无 `core.worktree`、refs 只指向自己。

**处理**：
1. 删 `.git/.git`（2.52 MB）；`.git` 5.89 MB → 5.41 MB；`fsck` 通过
2. `.git` 上移到工程根，重建索引（`read-tree --empty` + 定向 `add`），406 个文件被识别为 rename（历史不断）
3. 新增工程根 `.gitignore`：挡住 `Library/`(1.62 GB)、`Temp/`、`obj/`、`Logs/`、`UserSettings/`、`.vs/`(13.6 MB)、手装的 `Packages/*`
4. 新增工程根 `README.md`

**教训**：`git add -A` 不带 pathspec 会走查整个工作区（`Library` 26,871 个文件）→ 超时；
改用定向 pathspec（`git add -A -- Assets ProjectSettings ...`）后 **1.8 秒**完成。

---

## 阶段 AL：`dn` / `sx` 的神经网络化

完成品 `python/nn_dn_sx.py`。动机：现状两步都是**硬划分**，不可导，且实测 `rp` 虚高 ~100 倍。
完整推导与函数表见 **§8.2**。

---

## 8. 函数 ↔ 神经网络对应表（本轮核心）

> 把流程里每个**不可微 / 硬划分**环节逐一映射到网络结构。
> 这不是类比修辞：每条都给出「原机制 → 网络模块 → 为什么必须这样换 → 验证数据」。

### 8.1 总览

| # | 原机制（C#） | 神经网络对应 | 不可微点 / 动机 |
|---|---|---|---|
| 1 | `dn()`：`jj=(int)(360/d2)` 等角扇区 | `SoftSectors(K, kappa)`：K 个可学习扇区中心 + von Mises 软分配 | `(int)` 截断 + 硬扇区边界 |
| 2 | `sx()`：`prmax = max‖perp(Ap,vcs)‖`、`rp = max_sector prmax` | `soft_max(rho, beta)`（可微上界，恒 ≥ max） | 硬 `max` 梯度只走单点、且断裂 |
| 3 | `rp` 硬赋值 | `soft_quantile(x, q, τ, iters)`：二分 + 隐函数定理 | 分位数不可导 |
| 4 | `v3 = normalize(c1−c2)` | `Axis(v3_init)`：so(3) 指数映射参数化 | `normalize` 在 0 处奇点 |
| 5 | `th2` 输出 `v3`（**y 恒为 0，1 DOF**） | `AxisFromTh2(d2_rad)`：显式还原该耦合（**对照臂**） | 自由度不足 |
| 6 | 点云 → 柱面半径 | `cylinder_radius(P, v3)` → `ρ_i = ‖perp(p_i, v3)‖` | 把几何输入变成一层观测量 |
| 7 | 目标：`rp` 小 + 覆盖全部点 | `loss_fn(rho, A, rp, d2, w, scale)` | 覆盖约束由构造保证（§8.2） |
| 8 | RBF 刚度场 | `ComputeLeafLens → th4 → LocalQuadHessian → StiffnessProjectionFromH` | RBF 核宽是超参、拟合是黑箱 |
| 9 | 刚度矩阵 = 势能二阶导 | `Hess` / `LocalQuadHessian` = 局部二次型 = **loss landscape 曲率**（≈ Fisher / K-FAC / 二阶优化层） | 显式抽二阶信息，而非留在黑箱里 |
| 10 | ABCD 四通道 | 势梯度的**低阶多项式基** `[z, z̄, z², z̄²]` | 把黑箱梯度分成 4 个可解释复系数 |
| 11 | `Δ → τ` 反解 | **解析反演层**（implicit layer）：系数 → 模量 | 无需迭代，闭式可微 |
| 12 | 有理判据 | 连分数**精确终止**判据 | 判"代数数 vs 超越数" |

### 8.2 `dn` / `sx` 的完整替换

**现状实测（`Resources/points.json`）**
- `rp = 51.154373`，而点云 `|p|` 中位 ≈ 0.5 → **rp 大了约 100 倍**
- 经 `MapDoubleConeToLeaf` 后 `|uv|` 中位 9.04e-5、跨 3.94 dex → 点集严重失真
- 透镜场 `g ∈ [112, 11697] ≫ 2` → `Phi(g) = 0`、`V ≡ 0`

**三个互相拉扯的要求**
- (a) 覆盖：`rp ≥ max_i ‖perp(p_i, v3)‖`
- (b) 最小：`rp` 尽量小（锥最紧）
- (c) 不失真 + 合锥：共形映射后点集不失真，且贴合构造出的圆锥

(b) 要 `rp = max ρ`，(c) 要 `rp = 几何均值 ρ`
→ **只有把 ρ 的分布变窄才能同时满足**，这正是分组层 (`dn`/`sx`) 的职责，
也是它**必须可学习而不是固定划分**的理由。

**关键推论**
```
覆盖 + 最小  ⇒  rp → soft_max(ρ)
soft_max(ρ) 依赖 v3
⇒  最小化 rp  ≡  找一个让柱面半径最小的轴
```
这就是"优化 `rp` 会带动 `v3` 自动变化"的**准确含义**。

**函数表**

| 函数 | 作用 | 对应原机制的哪一步 |
|---|---|---|
| `hat(w)` / `exp_so3(w)` | 反对称矩阵 / so(3) 指数映射 | 无奇点的轴参数化 |
| `unit(v, eps)` | 数值安全归一化 | 替代裸 `normalize` |
| `Axis` | 可学习 `v3`（`nn.Module`） | `v3 = normalize(c1−c2)` |
| `ortho_frame(v3)` | 由 `v3` 构造正交基 | `yzqx` 坐标系 |
| `cylinder_radius(P, v3)` | `ρ_i = ‖perp(p_i, v3)‖` | `sx` 里的 `perp(Ap, vcs)` |
| `soft_max(x, beta)` | 可微上界（恒 ≥ max） | `sx` 的 `prmax` / `rp` 硬 max |
| `soft_quantile(x, q, tau, iters)` | 可微分位数（二分 + 隐函数定理） | `rp` 硬赋值 |
| `SoftSectors(K, kappa)` | K 个可学习扇区中心 + von Mises 软分配（用 `(cosθ, sinθ)` 内积，**无分支割线**） | `dn` 的等角扇区 `AngleAxis(i*d2, v3)` |
| `AxisFromTh2(d2_rad)` | 还原 `th2` 的 1-DOF `v3` 耦合 | `th2` |
| `loss_fn(rho, A, rp, d2, w, scale)` | soft_max 上界 + 覆盖惩罚 | 新目标函数 |
| `scan_axis_floor(P, d2, n, beta)` | 4000 轴扫描求 `rp` 经验地板 | 给出可达下界 |
| `report(tag, ...)` | 三模式对比打印 | 现状 / `v3` 自由 / `v3` 由 `th2` 耦合 |

**已验证数值**（4000 轴扫描）

| 量 | 值 |
|---|---|
| `std(log ρ)` 最优轴 | 0.462495 |
| `std(log ρ)` 最差轴 | 0.742192 |
| `std(log ρ)` PCA 轴 | 0.589057 |
| `soft_max(ρ)` 最优轴 | 1.310911 |
| `1/sin(d2)`（`d2 = 44.713528°`） | 1.1137 |
| C# `sx` 给出 `rp` | 51.154373 |
| PCA 轴硬 max `rp` | 1.463559（→ C# 是它的 **35 倍**） |

`points.json` 的 PCA 特征值比 `0.4296 / 0.3216 / 0.2488`（近球形）
⇒ ρ 的离散度是**内禀的**，不是选轴不当造成的。
学习到的 `v3` 收敛到地板附近（`rp` 1.4658→1.4195、`ρ_max` 1.4636→1.4042），但**增益很小**。

**两条设计修正记录**
1. 初版 `soft_max` + 自由 `rp` 的覆盖惩罚会让 `rp/ρ_max = 0.919 < 1`（违反覆盖）
   → 改为 **`rp = soft_max(rho, BETA)` 由构造保证覆盖**。
2. `v3.y ≡ 0` 在 `rp = 0.2068 / 0.6893 / 2.0679 / 6.8929` 下均成立
   → 证实 `th2` 只有 **1 个自由度**（`th2` 的两个输出共享同一 y）。

### 8.3 刚度 → 局部二次型

见 §AG 与 §8.1 第 8–9 行。
神经网络含义：`H` 是能量函数的 Hessian，即 **loss landscape 的曲率矩阵**；
这一步等价于网络里的 **Fisher / K-FAC / 二阶优化层**。

### 8.4 ABCD → 低阶可解释基 + 解析反演层

见 §AJ。`τ = log(ratio)/(2πi)` 是**隐式层反解**（形式等同 DEQ 的 implicit layer），
无需迭代、闭式可微。

---

## 9. 剩余问题（更新，取代 §5 第 1/3 项）

1. ~~模型偏差（0.6 地板）~~：仍是常数，但不阻碍方向对齐（阶段 K）。
2. **全局搜索策略**：单起点 NM 会困；随机/网格 + 抛光可达 0.0°。仍是开放项。
3. ~~泛化~~：真实数据 pyjson 已在阶段 AE 逐行对齐复现成功（Unity @80-100 轮读数逐位一致）。
4. **DEQ 网络化**：`nn_dn_sx.py` 已完成 `dn`/`sx` 一节；下一步是
   **2D（方位角 × 半径）软分配 + 多锥面 `rp`**，以及把 `dn` 的 `s[i]*r[k]` 缩放显式物化。
5. **收敛门替代预算**：消除内层 30/50/100 预算敏感。仍未解决。
6. **`Δ` 是否接入 `legacy/sjy.cs`**：目前 `BuildABCD` 只用 `B²−4AC`，`D` 未用；
   是否换成 `Δ = (B²−4AC)/(D²−4BC)` 待定。
7. **两个已知流水线 bug**（未修）：
   - `BatchProbability` 指数 `−delta²/8·a²` 应为 `/(8a²)`（当前会让 `probs ≡ 0`、`vss = 0`）
   - `BuildABCD` 尾部 `replaced1.Average()` 在空序列上抛 `InvalidOperationException`

---

## 10. 仓库最终形态与废物核查

**只上传代码结构**：`.py` + `.cs` + `.md` + `.gitignore`。

| 类别 | 处理 |
|---|---|
| `.py` 162 个 | 保留（顶层 8 + `experiments/` 16 + `archive/` 138） |
| `.cs` 11 个 | 保留 |
| `.md` 6 个 | 保留（根 README、log ×2、`python/README.md`、`experiments/README.md`、`Assets/Scripts/README.md`） |
| `.meta` 234 个 | 不入库（Unity 自动生成） |
| `.json` 27 / `.asset` 21 / `.txt` 3 | 不入库（Unity 设置与数据） |
| `.shader` 5 / `.mat` 4 / `.unity` 1 / `.renderTexture` 1 / `.jpg` 1 / `.png` 1 | 不入库（Unity 资源） |
| `.dll` 2 | 不入库（第三方二进制） |
| `.npz` 2 | 不入库（数据集，本地保留） |

**废物文件核查**：已删 `.git/.git`（嵌套仓库 2.52 MB）、空壳 `概率条件/`（只剩一个
对应已不存在 `.py` 的 `.pyc`）、3 个游离 `__pycache__`、冗余的 `Assets/Scripts/.gitignore`。
所有 `.meta` 文件**未做批量删除**（Unity GUID 不可再生）。

> **权衡**：不收 `.dll` 与 `points.json` ⟹ clone 后无法直接编译/跑数据，仓库定位为**代码展示**。
> 若需恢复可编译性，把根 `.gitignore` 里的 `*.dll` / `/Assets/Plugins/` 两条移除并重新 `add` 即可。

---

*§8 的函数↔网络对应表由本仓库的 C#/Python 实现**逐条比对**得出，不是事后类比；
所有数值均可用 `python/experiments/` 与 `python/nn_dn_sx.py` 复现。*
