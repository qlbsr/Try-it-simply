# Assets/Scripts/python

> 注意：本目录在 `Assets/` 内，Unity 会为每个文件生成 `.meta`，属正常现象。

## 目录结构

| 目录 | 内容 |
|---|---|
| `*.py`（顶层） | 与 C# 主链并行的 Python 实现 / 新实验入口 |
| `experiments/` | **当前活跃**的验证脚本（互相 import 成一簇，见下） |
| `archive/` | 历史一次性验证脚本（138 个），保留可追溯，不再维护 |
| `experiments/records/` | 不收敛点集记录（`bad_ws25_*.json`） |

顶层：

| 文件 | 作用 |
|---|---|
| `nn_dn_sx.py` | **可学习/可微的 dn+sx 替换 + 让 rp 有效**（当前主线） |
| `nsjy_algorithms.py` | C# `nsjy` 链的 Python 复刻（PCA / yzqx / 叶映射） |
| `n2sjy2.py` | C# `n2sjy2` 主闭环的 Python 复刻 |
| `data_driven_axis.py` | 数据驱动主轴 |
| `ebf_algorithm.py` | EBF 算法 |
| `analyze_relationship.py` / `fused_n2sjy2_nsjy4.py` / `combine_n2sjy2_nsjy4.py` | 通道关系与融合分析 |

## nn_dn_sx.py —— dn/sx 的神经网络替换（主线）

把 C# 里**不可微 / 硬划分**的两步换成可学习、可微的模块。

### 原机制与不可微点

| 原机制（`C#/legacy/sjy.cs`） | 为什么必须换 |
|---|---|
| `dn()`：`jj = (int)(360/d2)`，等角扇区 `vcs[i] = AngleAxis(i*d2, v3) @ cs` | `(int)` 截断 + 硬扇区边界 → 不可导；扇区中心固定不可学 |
| `sx()`：每扇区 `prmax = max‖perp(Ap, vcs)‖`，`rp = max_sector prmax` | 硬 `max` 梯度只走单点且断裂 |

实测（`Resources/points.json`）：
- `rp = 51.154373`，点云 `|p|` 中位 ≈ 0.5 → **rp 虚高约 100 倍**
- 经 `MapDoubleConeToLeaf` 后 `|uv|` 中位 9.04e-5、跨 3.94 dex → 点集严重失真
- 透镜场 `g ∈ [112, 11697] ≫ 2` → `Phi(g)=0`、`V ≡ 0`

### 三个互相拉扯的要求

- **(a) 覆盖**：`rp ≥ max_i ‖perp(p_i, v3)‖`
- **(b) 最小**：`rp` 尽量小（锥最紧）
- **(c) 不失真 + 合锥**：共形映射后点集不失真，且贴合构造出的圆锥

(b) 要 `rp = max ρ`，(c) 要 `rp = 几何均值 ρ` → **只有把 ρ 的分布变窄才能同时满足**。
这正是分组层（`dn`/`sx`）的职责，也是它必须可学习而非固定划分的理由。

**关键推论**：覆盖 + 最小 ⇒ `rp → soft_max(ρ)`，而 `soft_max(ρ)` 依赖 `v3`
⇒ 最小化 `rp` 等价于"找一个让柱面半径最小的轴"。这就是"优化 rp 会带动 v3 自动变化"的准确含义。

### 函数表

| 函数 | 作用 | 对应原机制的哪一步 |
|---|---|---|
| `hat(w)` / `exp_so3(w)` | 反对称矩阵 / so(3) 指数映射 | 无奇点的轴参数化 |
| `unit(v, eps)` | 数值安全归一化 | 替代裸 `normalize` |
| `Axis` | 可学习 `v3`（`nn.Module`） | `v3 = normalize(c1−c2)` |
| `ortho_frame(v3)` | 由 `v3` 构造正交基 | `yzqx` 坐标系 |
| `cylinder_radius(P, v3)` | `ρ_i = ‖perp(p_i, v3)‖` | `sx` 里的 `perp(Ap, vcs)` |
| `soft_max(x, beta)` | 可微上界（恒 ≥ max） | `sx` 的 `prmax` / `rp` 硬 max |
| `soft_quantile(x, q, tau, iters)` | 可微分位数（二分 + 隐函数定理） | `rp` 硬赋值 |
| `SoftSectors(K, kappa)` | K 个可学习扇区中心 + von Mises 软分配（用 `(cosθ, sinθ)` 内积，无分支割线） | `dn` 的等角扇区 `AngleAxis(i*d2, v3)` |
| `AxisFromTh2(d2_rad)` | 还原 `th2` 的 1-DOF `v3` 耦合 | `th2` |
| `loss_fn(rho, A, rp, d2, w, scale)` | soft_max 上界 + 覆盖惩罚 | 新目标函数 |
| `scan_axis_floor(P, d2, n, beta)` | 4000 轴扫描求 `rp` 经验地板 | 给出可达下界 |
| `report(tag, ...)` | 三模式对比打印 | 现状 / `v3` 自由 / `v3` 由 `th2` 耦合 |

### 已验证数值

| 量 | 值 |
|---|---|
| `std(log ρ)` 最优轴 / 最差轴 / PCA 轴 | 0.462495 / 0.742192 / 0.589057 |
| `soft_max(ρ)` 最优轴 | 1.310911 |
| `1/sin(d2)`（`d2 = 44.713528°`） | 1.1137 |
| C# `sx` 给出 `rp` | 51.154373 |
| PCA 轴硬 max `rp` | 1.463559（→ C# 是它的 **35 倍**） |

`points.json` 的 PCA 特征值比 `0.4296 / 0.3216 / 0.2488`（近球形）⇒ ρ 的离散度是**内禀的**，
不是选轴不当造成的。学习到的 `v3` 收敛到地板附近（`rp` 1.4658→1.4195、`ρ_max` 1.4636→1.4042），但增益很小。

两条设计修正记录：
1. 初版 `soft_max` + 自由 `rp` 的覆盖惩罚会让 `rp/ρ_max = 0.919 < 1`（违反覆盖）
   → 改为 **`rp = soft_max(rho, BETA)` 由构造保证覆盖**。
2. `v3.y ≡ 0` 在 `rp = 0.2068 / 0.6893 / 2.0679 / 6.8929` 下均成立
   → 证实 `th2` 只有 **1 个自由度**（`th2` 的两个输出共享同一 y）。

## experiments/ 的模块簇

这 16 个文件互相依赖，**不能拆开**：

```
复刻库（被其它脚本 import）
  verify_v2_provenance.py    Vss1 + ComputeAllFeatures 复刻
  verify_v2_perpoint.py      th4 两叶复刻
  verify_v2_route.py         MapUVToS3 / ExtractAngles / SphericalRBF / NumericalGradient
  verify_v2_simplified.py    Phi(g) 闭式 + 等价性 + fit_polynomial + tau_from_ABCD

结论脚本
  verify_lens_volume_identity.py    4 维球冠恒等式 / x=1-g²/4 / trace-multiplier
  verify_v2_replaces_rbf.py         (d,g)->(I,V)；RBF 替换前后对照
  verify_v2_jacobian.py             雅可比面积元版
  verify_leaf_two_sheets.py         两叶几何 / d 闭式 / 依赖表
  verify_parabola_vs_sheets.py      抛物线 vs 双叶：两个向量到底编码了什么
  verify_d_vs_vss.py                d 对 vss 的依赖扫描
  verify_vss1_is_rotation_only.py   Vss1 输出只差一个刚体旋转（Kabsch）
  verify_constant_part.py           常数 Hessian 只进 A,B 通道
  verify_sjysjy_map.py              sjysjy.MapDoubleConeToLeaf 全文复刻
  verify_sjysjy_relations.py        SolveC3 关联式 + SolveTFromE

数据导出
  dump_points_ab.py     导出 Vss1 的两条抛物线（给 C# 测试工程）
  dump_uvcloud.py       导出 uv 点云 + 合成概率场
```

## 运行

```powershell
cd "C:\Users\23128\My project (2)\Assets\Scripts\python"
python nn_dn_sx.py                    # 当前主线
python experiments\verify_v2_simplified.py
```

依赖：`numpy` / `scipy` / `torch`（CPU 足够）
