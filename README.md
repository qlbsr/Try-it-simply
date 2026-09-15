# 阿贝尔曲面 · 格点概率 · 焦点方向闭环

点云 → 锥面展开（`InverseTh4`）→ 复平面格点概率场（r30/r45/r74）→ 焦点提取 → 方向向量 `d`，
目标是在**不依赖 PCA** 的前提下，让焦点方向与参考方向形成可验证的几何闭环。

---

## 仓库结构

仓库根 = **Unity 工程根**，包含完整可复现所需的全部源文件：

```
Assets/                        Unity 资源（全部纳入版本控制）
  Plugins/                     第三方程序集
    MathNet.Numerics.dll         数值线性代数（MIT，唯一数值依赖）
    Newtonsoft.Json.dll          JSON 解析（MIT）
  Resources/
    points.json                  实验点云数据（Resources.Load 读取）
  Scripts/                     算法与实验代码
    C#/                          Unity 主实现（MonoBehaviour + 静态算法库）
      nsjy.cs                      核心算法库：Carlson 椭圆积分、InverseTh4、ComputeTaus、
                                   ExtractFoci / FitFocus、pca、LatticeAnalysis、LLL、
                                   ComplexLinearFit、LatticeRansac、DeviationCalculator
      n2sjy.cs                     LM 闭环（Levenberg-Marquardt 迭代）
      n2sjy2.cs                    双角外循环：方向角条件 + Nelder-Mead 扩张放大
      nsjy.cs / n2sjy.cs 等
      sjysjy.cs                    双锥面 → 叶片展开映射 MapDoubleConeToLeaf
      v2sjy.cs                     透镜体积闭式 + 复平面 ABCD 四通道拟合（无库依赖）
      legacy/                      早期/旁支脚本（sjy、sjy1 球面 RBF、xs UI 等，仅供考古）
    python/                     研究与验证（可独立运行，无 Unity 依赖）
      nsjy_algorithms.py           核心算法库的完整 Python 移植
      n2sjy2.py                    双角闭环 Python 版
      nn_dn_sx.py                  将 dn / sx 改写为可学习模块（so(3) 轴 + 软极大/软分位）
      data_driven_axis.py          数据驱动轴方向实验
      ebf_algorithm.py             Explore → Best → Fine-tune（EBF）策略
      fused_n2sjy2_nsjy4.py        融合方案
      experiments/                 当前有效的验证脚本（每个文件名对应一次假设验证）
                                  + 4 份移植副本 + 12 份结论脚本
      archive/                     138 个历史一次性实验脚本（归档，不再维护）
      README.md                    python 侧目录与模块簇说明
    log/                         nsjy_pipeline_analysis.md  技术分析日志（数学本质/阶段结论/改进清单）
                                 ai_call_log.md             AI 调用与提示词日志
Packages/
  manifest.json                 Unity 包依赖清单（唯一被跟踪的 Packages 内容）
  packages-lock.json            依赖锁定
ProjectSettings/                Unity 工程设置
```

> `Packages/` 下的第三方解包目录（Numerics.NET 等 179 MB 本地包）、`Library/`（1.6 GB）、
> `Temp/`、`obj/`、`Logs/`、`UserSettings/`、`.vs/` 均已在根 `.gitignore` 中忽略，不会上传。

## 运行

- **C# 侧**：用 Unity 打开本工程（2022.3.62f3c1）。数值依赖来自 `Assets/Plugins/`，
  点云数据来自 `Assets/Resources/points.json`，clone 后无需额外配置即可编译。
- **Python 侧**：在 `Scripts/python/` 目录下运行，例如

  ```bash
  cd Assets/Scripts/python
  python experiments/verify_lens_volume_identity.py
  ```

  实验脚本已内置仓库根路径引导，`import nsjy_algorithms` 等可直接使用。

## 核心结论（如实记录）

| 结论 | 说明 |
|---|---|
| 双角条件 `|Δ| = |angle(d,v) − angle(d,d0)| ≤ 10°` | 几何上等价于 `|2θ − θ0|`，可实现闭环（V 形漏斗，`normalize(v+d0)` 为其一个解析驻点） |
| `angle(d,v) ≤ 10°`（方向对齐 PCA）**可达** | 网格穷举 10⁴ 点得 4.74°(ellip0)/8.21°(cube0)/9.35°(ball0)；随机+NM 抛光 0.0°。旧"~31.7° 不可达"系 NM 单起点被困假象（阶段 K 修正） |
| \|Δ\| 漏斗形态 = 纯几何 | 随机方向（无曲线计算）下同样成立：bisector 解析零点、corr 0.98；\|Δ\|=0 解点的**存在性**由椭圆曲线结构提供（可达域覆盖等角大圆），优化器只是执行者 |
| `σ` 无关 | 在距离反演 `d̂ = σ√(−2 ln p) = d` 中 σ 可消去 |
| 全旋转设计自指 | 把参考精确映射到 `d_new` 会使 `|Δ|` 恒为 0，属自证，无信息量 |
| 数据参考（d0、d_i） | 自洽但偏离 PCA 62–128°，只能作指标不能作真值 |
| 透镜体积闭式 | `V = R⁴·I_x(5/2,1/2) = (4/π²)·Vol₄(cap)`，`x = 1 − g²/4`，`Phi(g) = [12α − 8sin2α + sin4α]/(6π)`；与正则化不完全 Beta 吻合到 3.6e-11 |
| `g = d/r` 是正确的无量纲参数 | 与 `g = ‖dJ/ds‖`（s = 锥面半径）一致；叶片雅可比版 `g` 在真实 uvs 上是 `[112, 11697]`，不可用 |
| ABCD 不是四个独立物理量 | 是外部势梯度在复平面低阶展开的四个复系数通道 `∂U/∂z = Az + Bz̄ + Cz² + Dz̄² + …`；常 Hessian 部分只落进 A、B，C=D=0 到机器精度 |

> 详细推导与验证过程见 `Assets/Scripts/log/nsjy_pipeline_analysis.md`。
