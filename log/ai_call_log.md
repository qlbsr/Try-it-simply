# AI 调用与提示词日志（DeepSeek）

> 项目：`C:\Users\23128\My project (2)\Assets\Scripts`
> 日期：2026-08-29 ~ 2026-08-31
> 记录方式：本项目所有 AI 辅助开发均通过与 **DeepSeek** 的对话式调用完成
> （DeepSeek Harness / DSH，模型 `deepseek-v4-flash`）。一次对话轮次 = 一次调用。
> 本文件记录每次调用的**提示词（精简）**、**输入/输出要点**与**落地改进**，
> 与 `nsjy_pipeline_analysis.md`（技术分析、改进清单）互为补充、可交叉索引。

---

## 0. 调用约定

| 术语 | 含义 |
|---|---|
| 提示词 | 用户在对话中提出的请求（原文精简，完整原文见会话记录） |
| 输入 | AI 读取的代码 / 日志 / 验证脚本（本轮上下文） |
| 输出 | AI 的结论与修改（**已精简**为要点，全部可由脚本复现） |
| 落地改进 | 该轮实际写入代码 / 仓库的变更，编号对应 `nsjy_pipeline_analysis.md` 第 3 节改进清单 |

---

## 1. 调用记录总表（精简对话）

| # | 日期 | 提示词（精简） | 输入要点 | 输出要点（精简） | 落地改进 |
|---|---|---|---|---|---|
| 1 | 08-29 | 分析 `FitFociByProbability` 收敛后每点概率偏差 0.1~0.2 | `nsjy.cs` 的 `BatchProbability` / `FitFociByProbability` | ① 运算符优先级 bug：`-abs(delta)/2*a` 应为 `-(abs(delta))/(2*a)`，指数被放大 a²；② 根本失配：`probTotal`（变换域格距离）与 `probFoci`（3D 焦点距离）系统性偏差 | 改进 1（括号修复） |
| 2 | 08-29 | 设计闭环：`(t1,t2) → 格概率 → 焦点 → 方向 ∥ PCA` | `n2sjy.cs` 闭环雏形 | LM + 数值雅可比在最近格点距离（分段常数）处梯度≈0 → 卡平台 | 建立闭环框架 |
| 3 | 08-29 | 收敛失败机理分析 | 各数据集收敛轨迹 | `NormalizeTau` 模跳变（τ→−1/τ 旋转格）造成假恶化；退化格逃逸（Im τ→0） | 改进 6（不调 NormalizeTau、重试起点、best-of-N） |
| 4 | 08-29 | 全流程 Python 移植以便快速实验 | `nsjy.cs` 全部算法 | 移植出 `nsjy_algorithms.py`（867 行），移植中发现 4 处 C# bug | 改进 2/3/4/5（LLL、angleDeg2、pca pc1、solve4） |
| 5 | 08-29 | 系列假设证伪（P2/P3/E1–E3/档2/两角相等/互监督） | `verify_*.py` / `test_*.py` 脚本 | 各假设均证伪：无普适 (t1,t2)、自洽≠方向对齐、偏差非正比、概率环无吸引子、互监督发散 | 方向收敛判据改为"方向稳定度"（改进 7） |
| 6 | 08-29 | 修正循环：起点与选择策略 | 34 集结果 | 重试起点 (0,1.2793)→(0,1)；best-of-N 事后选择；档1 14→20 集 | 改进 6 收尾 |
| 7 | 08-30 | 趋势机理：`|Δ| = |angle(d,v) − angle(d,d0)|` 为何总是下降 | 6 集轨迹数据 | 几何等价 `|2θ−θ0|`（V 形漏斗）；等角集=大圆；解析驻点 `d\* = normalize(v+d0)` 处 `|Δ|=0.00°`（6/6）；corr 0.94~0.99 | 理论突破，写入分析日志阶段 G |
| 8 | 08-30 | 无梯度快速达成两角条件 | `data_driven_axis.py` | Nelder-Mead / Powell 直接优化 `|Δ|`：117~330 次求值达 `|Δ|≤10`，多数 0.0°；PCA 仍在目标函数内（第二参考） | 改进 8（nsjy4 思路） |
| 9 | 08-30 | 探索→最佳点→微调（EBF） | `ebf_algorithm.py` | 探索用 `extract_foci` 替代重拟合（成本 −100×），6/6 达标 `|Δ|=0.00`，约 64~94s/集 | 改进 9（nsjy5 思路） |
| 10 | 08-30 | 神经网络整合构想 | 前述全部结论 | DEQ 合并漂移+优化为可微自迭代网络；四步路线（soft-min → 偏差教师训练 → 收敛门 → PCA 退化为标签） | 构想入日志阶段 J（未实现） |
| 11 | 08-30 | 完全脱离 PCA 的可行性验证 | `verify_pca_free.py`、`verify_angle_v_target.py` 等 | **`angle(d,v) ≤ 10°` 不可达**：模型偏差地板 max\|probTotal−probFoci\|=0.58~0.62（52 集）阻断，方向仅能逼近 PCA 至 min 31.7°（0/7 集达标）；全旋转设计自指、无信息量 | 明确结论：PCA 不可删除，只能作指标/教师 |
| 12 | 08-31 | 已装 GitHub Desktop，把代码整洁简化后上传仓库 | 仓库现状（120+ 文件混杂） | 归档 36 个实验脚本→`python/experiments/`、旧 C#→`C#/legacy/`、补 README、清理 `__pycache__`/日志/PAT 助手；提交 `b33bef4`，121 文件上传 `qlbsr/Try-it-simply` | 仓库整理（改进 10） |
| 13 | 08-31 | 审查：AI 调用、提示词、输入输出是否明确写入 log | 审查 `log/` | 发现只有技术改进清单、无调用记录 → 创建本文件（`ai_call_log.md`），提示词/输入输出/改进三栏对齐 | **本文件** |

---

## 2. 提示词原文（精简归档）

> 按调用约定，以下为各轮提示词的精简版；完整原文保留在会话记录中（不随仓库分发）。

1. "分析 `FitFociByProbability` 收敛后每点概率差 0.1~0.2 的原因。"
2. "设计一个 (t1,t2) 主参数的闭环，让焦点方向闭合到 PCA 轴。"
3. "分析为什么收敛失败 / 卡死。"
4. "把整套 C# 算法移植成 Python 以便快速实验验证。"
5. "用实验证伪/证实下列假设：……"（P2/P3/E1–E3/档2/两角相等/互监督）
6. "修正循环：换起点、best-of-N 事后选择，看 34 集分类效果。"
7. "`|Δ|` 为什么总是单调下降？机理是什么？"
8. "能不能不用梯度、快速达成两角条件？"
9. "探索→最佳点→微调（EBF）方案，成本优化。"
10. "能不能把漂移+优化合并成一个神经网络（DEQ）？"
11. "完全脱离 PCA 的可行性：`angle(d,v) ≤ 10°` 能否达到？"
12. "把代码整洁和简化，然后上传仓库。"
13. "审查 log：AI 调用、提示词、输入输出要明确写入；输出精简。"

---

## 3. 输入/输出示例（1 轮完整还原）

以 #7（趋势机理）为例：

- **输入**：`data_driven_axis.py` 生成的 6 数据集轨迹 `(angle(d,v), angle(d,d0))` 随时间步变化。
- **输出（精简）**：
  1. `|Δ| = |angle(d,v) − angle(d,d0)|` 恒等于 `|2θ − θ0|`（θ 为 d 与等分面的夹角）——**V 形漏斗**；
  2. 等角集是球面上垂直于 `(v−d0)` 的大圆；
  3. 解析驻点 `d* = normalize(v+d0)` 处 `|Δ| = 0.00°`（6/6 数据集精确）；
  4. 实测 `corr(|Δ|, |d·(v−d0)|) = 0.938 ~ 0.994`；
  5. 理想先决条件 `angle(d0,v)≈0`，现实 55~125°（模型偏差）。
- **落地**：写入 `nsjy_pipeline_analysis.md` 阶段 G；成为 nsjy4/nsjy5 的目标函数设计依据。

---

## 4. AI 调用 → 改进 映射

| 调用轮次 | 对应改进（`nsjy_pipeline_analysis.md` §3） | 验证脚本 |
|---|---|---|
| 1 | 1 | `verify_corrected.py` |
| 3 | 6 | `test_convergence.py`、`test_best_of_drift*.py` |
| 4 | 2, 3, 4, 5 | `nsjy_algorithms.py` 全量移植 |
| 5 | 7 | `verify_falsify.py`、`verify_fixedpoint.py`、`verify_tier2.py` |
| 7 | （理论） | `verify_data_bisector.py`、`verify_trend_cause.py` |
| 8 | 8 | `test_nelder_mead.py`、`verify_swing_center.py` |
| 9 | 9 | `verify_nsjy5.py` |
| 11 | （负面结论） | `verify_pca_free.py`、`verify_angle_v_target.py`、`measure_deviation_floor.py` |
| 12 | 10（仓库整理） | `README.md`、`git log` |

---

*本日志由 DeepSeek 在协作开发过程中生成，与 `nsjy_pipeline_analysis.md` 同步维护。*
