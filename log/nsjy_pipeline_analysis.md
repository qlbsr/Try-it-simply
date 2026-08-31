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
| 概率偏差地板（52 集） | max\|probTotal−probFoci\| = 0.58~0.62（σ=0.011） | 模型常数，非数据属性；devTol=1e-4 死阈值 |
| 两角条件达标（nsjy5，6 集） | 6/6，\|Δ\|=0.00 | 快速达成条件 |
| 34 集分类（修正后） | 档1=20、容错带=11、档2=3 | 框架自洽 |
| 趋势机理（6 集） | corr(\|Δ\|, \|d·(v−d0)\|)=0.94~0.99；normalize(v+d0) 处 \|Δ\|=0.00 | 等角大圆 V 形漏斗 |
| 理想先决条件 | angle(d0,v)=55~125°（mean 92°） | 模型偏差大 → 条件需搜索 |
| 互监督共识 | ∠(d\*,v)=43~158°；迭代发散 | 同源偏差不抵消 |
| 档2 自收敛 | 0/8（固定值）、0/8（固定范围） | 概率环无吸引子 |

---

## 5. 剩余问题与开放方向

1. **模型偏差（0.6 地板）**：格距离 ≠ 3D 距离，唯一出路是"模不变距离"（SL(2,Z) 归约后算最短格距）或"神经网络偏差修正"。
2. **泛化**：合成分布上验证，真实数据（pyjson）需复测。
3. **DEQ 网络化**：平滑化（soft-min）+ 合成轴教师训练 + 收敛门 + PCA 退化为教师。
4. **收敛门替代预算**：消除内层 30/50/100 预算敏感（不动点随步数跳 16~47°）。

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

*本日志由 DeepSeek 在迭代分析过程中生成，所有结论均有 `verify_*.py` / `test_*.py` 脚本可复现。*
