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
