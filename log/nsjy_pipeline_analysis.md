# nsjy 算法全流程分析日志（DeepSeek 改进记录）

> 项目：`C:\Users\23128\My project (2)\Assets\Scripts`
> 日期：2026-08-29
> 内容：整套 nsjy 家族代码的数学本质分析、实验验证过程、DeepSeek 逐项改进清单、剩余问题、参考文献与相关项目清单。

---

## 0. 文件清单

### C# 主算法文件
| 文件 | 内容 |
|---|---|
| `nsjy.cs` | 基础算法库：`InverseTh4`/`yzqx`（圆锥坐标变换）、`ComputeRp`、`ComputeTaus`（CM 模量）、`Carlsonfk`（RF/RD/K/E/DK/NK 复数椭圆积分）、`ExtractFoci`/`FitFocus`、`FitFociByProbability`/`BatchProbability`、`pca`、`LatticeAnalysis`（格基/周期矩阵）、`LLL`、`ComplexLinearFit`、`LatticeRansac`、`DeviationCalculator`（最近格点概率场） |
| `n2sjy.cs` | 闭环优化 `RefineModuliByAxis`（LM + 数值雅可比 + NormalizeTau） |
| `n2sjy2.cs` | 双角度外环：`angleDeg`(与PCA夹角)/`angleDeg1`(与初始焦点方向夹角)，停止条件 `\|angleDeg−angleDeg1\|/2≤5` |
| `nsjy3.cs` | 两套判断闭环：档1(θ<0.5°) / 容错带 / 档2(自迭代)，含 best-of-N 重试 |
| `nsjy4.cs` | Nelder-Mead 无梯度求解器，快速达成两角条件 \|Δ\|≤10° |
| `nsjy5.cs` | 改进版 EBF（探索→最佳点→微调），自包含私有 Nelder-Mead |

### Python（移植/验证）
`nsjy_algorithms.py`（全量移植）、`n2sjy2.py`、`test_convergence.py`、`test_axis_hypothesis.py`、`verify_*.py`（deviation/falsify/fixedpoint/tier2/range/final/orig_angles/corrected/correspondence/all_sets）、`measure_deviation_floor.py`、`test_nelder_mead.py`、`data_driven_axis.py`、`combine_n2sjy2_nsjy4.py`、`ebf_algorithm.py`、`verify_nsjy5.py` 等。

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
- **Q1 规律成立（部分）**：`|(angleDeg2+angleDeg1)/2−min| ≤ 8`（∠(d,v1)≈∠(d,v3)，无 PCA）达成时 s0[0]≈s5[0]：pyjson |s0−s5|=1.56（iter1）、ball0 0.87（iter3）强成立；ellip0 6.60（iter6）中等。几何链：等角(d,v1,v3) ⟹ P(PCA方向)≈P(d) ⟹ ∠(p0,v)≈∠(p0,d)（三角形反演 s→θ）。
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
