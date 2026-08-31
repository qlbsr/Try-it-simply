# 阿贝尔曲面 · 格点概率 · 焦点方向闭环

点云 → 锥面展开（`InverseTh4`）→ 复平面格点概率场（r30/r45/r74）→ 焦点提取 → 方向向量 `d`，
目标是在**不依赖 PCA** 的前提下，让焦点方向与参考方向形成可验证的几何闭环。

## 目录结构

```
C#/                  Unity 主实现（MonoBehaviour + 静态算法库）
  nsjy.cs            核心算法库：Carlson 椭圆积分、InverseTh4、ComputeTaus、
                     ExtractFoci / FitFocus、pca、LatticeAnalysis、LLL、
                     ComplexLinearFit、LatticeRansac、DeviationCalculator
  n2sjy.cs           LM 闭环（Levenberg-Marquardt 迭代）
  n2sjy2.cs          双角外循环：方向角条件 + Nelder-Mead 扩张放大
  legacy/            早期/旁支脚本（sjy、sjy1 球面RBF、xs UI 等，仅供参考）
python/              研究与验证（可独立运行，无 Unity 依赖）
  nsjy_algorithms.py 核心算法库的完整 Python 移植
  n2sjy2.py          双角闭环 Python 版
  data_driven_axis.py  数据驱动轴方向实验
  ebf_algorithm.py   Explore → Best → Fine-tune（EBF）策略
  fused_n2sjy2_nsjy4.py / combine_n2sjy2_nsjy4.py  融合方案
  analyze_relationship.py  各量关系分析
  experiments/       历史验证/实验脚本（一次性，归档保留）
log/                 nsjy_pipeline_analysis.md 完整分析日志
DLL.NET/             第三方依赖（MathNet.Numerics 等，仅 C# 侧使用）
```

## 运行

- **C# 侧**：Unity 中把对应脚本挂到场景对象（依赖 `DLL.NET/` 下的程序集）。
- **Python 侧**：在 `python/` 目录下运行，例如

  ```bash
  cd python
  python experiments/verify_swing_center.py
  ```

  实验脚本已内置仓库根路径引导，`import nsjy_algorithms` 等可直接使用。

## 核心结论（如实记录）

| 结论 | 说明 |
|---|---|
| 双角条件 `|Δ| = |angle(d,v) − angle(d,d0)| ≤ 10°` | 几何上等价于 `|2θ − θ0|`，可实现闭环（V 形漏斗，`normalize(v+d0)` 为其一个解析驻点） |
| `angle(d,v) ≤ 10°`（方向对齐 PCA）**不可达** | 模型偏差下限 `max|probTotal − probFoci| ≈ 0.58–0.62`（52 组数据）：格点距离 ≠ 三维距离，导致方向最多逼近 PCA 至 ~31.7° |
| `σ` 无关 | 在距离反演 `d̂ = σ√(−2 ln p) = d` 中 σ 可消去 |
| 全旋转设计自指 | 把参考精确映射到 `d_new` 会使 `|Δ|` 恒为 0，属自证，无信息量 |
| 数据参考（d0、d_i） | 自洽但偏离 PCA 62–128°，只能作指标不能作真值 |

> 详细推导与验证过程见 `log/nsjy_pipeline_analysis.md`；`python/experiments/` 内每个脚本文件名对应一次假设验证。

## 备注

- 仓库只包含 `Assets/Scripts`（算法与实验代码），不含整个 Unity 工程（Library/Temp 等 1.6GB 缓存）。
- `__pycache__/`、`*.pyc`、`*.log` 已在 `.gitignore` 中忽略。
