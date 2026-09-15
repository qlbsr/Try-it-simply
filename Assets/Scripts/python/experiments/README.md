# experiments — 历史验证 / 实验脚本

本目录存放开发过程中的一次性验证脚本（`verify_*`、`test_*`），**仅归档保留，非正式接口**。

- 每个脚本对应一次假设验证或实验，结论汇总见 `log/nsjy_pipeline_analysis.md` 与仓库根 `README.md`。
- 运行方式：在 `python/` 目录下执行

  ```bash
  cd python
  python experiments/verify_swing_center.py
  ```

- 脚本头部已注入仓库根路径引导（`sys.path`），可正常 `import nsjy_algorithms`、`n2sjy2`、`data_driven_axis` 等顶层模块。
- 各脚本依赖 `numpy`、`scipy`、`matplotlib`。

| 脚本 | 验证内容 |
|---|---|
| `verify_pca_free.py` | 完全脱离 PCA 的方向闭环可行性 |
| `verify_angle_v_target.py` | angle(d,v) ≤ 10° 可达性（结论：不可达，下限 ~31.7°） |
| `verify_swing_center.py` | 摆动中心 / 二分面几何 |
| `verify_data_bisector.py` | 数据参考 `normalize(v+d0)` 驻点性质 |
| `verify_full_loop.py` | 全旋转闭环（结论：自指，无信息量） |
| `verify_final.py` / `verify_all_sets.py` | 52 组数据整体偏差 |
| `measure_deviation_floor.py` | 模型偏差下限 ~0.58–0.62 |
| `verify_trend_cause.py` | 趋势机制归因（n2sjy2 漂移 + NM 扩张放大） |
| `find_ideal_factors.py` | 理想因子搜索 |
| 其余 `test_*` | 各阶段候选策略的快速试验 |

如需清理历史，可直接删除本目录（不影响 `python/` 顶层正式脚本）。
