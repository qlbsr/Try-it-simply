"""
verify_junction_state.py — 交接处到底是什么, 以及"三个向量"够不够
================================================================================

用户的问题:
    交接处究竟是什么?
    两个向量还好求(已解: 弦∩抛物线, 二次闭式)
    三个向量怎么求? 猜想是"交接处直接与第三个向量相交就行"
    那么每次更新的就是交接处

代码证据 (legacy/sjy.cs:624-684, GetVertexFromFourPoints):
    origin = p1
    xAxis  = (p2 - p1).normalized
    normal = cross(p2-p1, p3-p1).normalized
    yAxis  = cross(normal, xAxis).normalized
    把 p2, p3, p4 投影成 (Xk, Yk)
    用 Cramer 法则解 [x^2, x, 1][a,b,c]^T = [y]  —— 【只用 p2,p3,p4 三个点拟合】
    顶点 = origin + (-b/2a)*xAxis + (c - b^2/4a)*yAxis

本文件结论:
  1. 交接处 = 当前空间曲线(抛物线) 与 由实体向量界定的弦 的【交点对】。
     它同时是上一层的输出与下一层的输入, 所以"每次更新的就是交接处"成立。
  2. 拟合只需要 3 个点, 但【还需要一个坐标架(原点 + x 方向)】= 4 个输入自由度。
     所以严格说 3 个向量不够: p1 定架 + p2,p3,p4 拟合。
  3. 但若【继承上一层的坐标架 (e1,e2)】(层与层应连续), 则 3 点即可,
     而且所有层共面, 交界点不会漂出平面。当前代码每层重建坐标架,
     这正是"简陋"所在, 也是 3 向量不够用的原因。
  4. 失效模式: det < 1e-10 -> LogError + return Vector3.zero
     -> vz2 = 0 -> 下游 e1 = zero.normalized = 0 -> 抛物线静默塌成直线。

运行: python verify_junction_state.py
"""
import io, math, sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ----------------------------------------------------------------------
#  复刻 GetVertexFromFourPoints
# ----------------------------------------------------------------------
def vertex_from_four(p1, p2, p3, p4):
    p1, p2, p3, p4 = (np.asarray(x, float) for x in (p1, p2, p3, p4))
    origin = p1
    d21 = p2 - p1
    if np.dot(d21, d21) < 1e-8:
        return None, "p1==p2"
    xAxis = d21 / np.linalg.norm(d21)
    nrm = np.cross(p2 - p1, p3 - p1)
    if np.dot(nrm, nrm) < 1e-8:
        return None, "p1,p2,p3 共线"
    nrm = nrm / np.linalg.norm(nrm)
    yAxis = np.cross(nrm, xAxis)
    yAxis = yAxis / np.linalg.norm(yAxis)
    X = [float(np.dot(p - origin, xAxis)) for p in (p2, p3, p4)]
    Y = [float(np.dot(p - origin, yAxis)) for p in (p2, p3, p4)]
    x2, x3, x4 = X
    y2, y3, y4 = Y
    det = x2 * x2 * (x3 - x4) + x3 * x3 * (x4 - x2) + x4 * x4 * (x2 - x3)
    if abs(det) < 1e-10:
        return None, "三个 x 线性相关"
    a = (y2 * (x3 - x4) + y3 * (x4 - x2) + y4 * (x2 - x3)) / det
    b = (x2 * x2 * (y3 - y4) + x3 * x3 * (y4 - y2) + x4 * x4 * (y2 - y3)) / det
    c = (x2 * x2 * (x3 * y4 - x4 * y3) + x3 * x3 * (x4 * y2 - x2 * y4)
         + x4 * x4 * (x2 * y3 - x3 * y2)) / det
    if abs(a) < 1e-8:
        return None, "a≈0 不是抛物线"
    return origin + (-b / (2 * a)) * xAxis + (c - b * b / (4 * a)) * yAxis, None


def vertex_in_frame(pts, e1, e2, e3):
    """在【已给定】的局部系里用 3 点拟合 u = a v^2 + b v + c, 返回顶点(世界坐标)。"""
    V = [float(np.dot(p, e2)) for p in pts]
    U = [float(np.dot(p, e1)) for p in pts]
    M = np.array([[v * v, v, 1.0] for v in V])
    if abs(np.linalg.det(M)) < 1e-12:
        return None
    abc = np.linalg.solve(M, np.array(U))
    a, b, c = abc
    if abs(a) < 1e-12:
        return None
    vv = -b / (2 * a)
    uu = c - b * b / (4 * a)
    return uu * e1 + vv * e2


# ----------------------------------------------------------------------
def part1_definition():
    print("=" * 78)
    print("1. 交接处是什么")
    print("=" * 78)
    print("  交接处 = 【当前空间曲线(抛物线)】 ∩ 【由实体向量界定的弦】 的交点对")
    print()
    print("  它有双重身份, 这正是'每次更新的就是交接处'的原因:")
    print("      上一层:  输出 -> (inter1, inter2) = 弦 ∩ 抛物线")
    print("      下一层:  输入 -> 参与拟合出下一层的抛物线(顶点), 并充当新的原点")
    print()
    print("  而且曲线本身不必携带: 它在 (e1,e2) 系里是 u = L - (L/Z^2)v^2,")
    print("  由【2 个端点 + 标尺 L=sqrt(r[2])】完全决定。")
    print("  所以整个状态就是: 2 个实体向量 + 1 个标量, 加上交接处(更新量)。")
    print()


def part2_frame_role():
    print("=" * 78)
    print("2. p1 不是冗余: 它决定坐标架, 因此决定拟合出哪条抛物线")
    print("=" * 78)
    # 一条真抛物线: u = 1 - (1/0.3^2) v^2, 取 4 个共面点
    L, Z = 1.0, 0.3
    e1 = np.array([0.0, 0.0, 1.0])
    e2 = np.array([1.0, 0.0, 0.0])
    vs = np.array([-0.25, -0.10, 0.05, 0.28])
    pts = [L * 0 + (L - (L / (Z * Z)) * v * v) * e1 + v * e2 for v in vs]
    true_vertex = L * e1 + 0.0 * e2
    print(f"  真抛物线: L={L}, Z={Z}  =>  真顶点 = {true_vertex}")
    print()
    print(f"  {'p1 取自':>22} {'拟合顶点':>34} {'误差':>12}")
    for tag, p1 in [("真顶点本身", true_vertex),
                    ("曲线上另一点", pts[1]),
                    ("曲线外(同平面)", (L - 1.2) * e1 + 0.4 * e2),
                    ("离平面一点", (L - 0.5) * e1 + 0.2 * e2 + 0.35 * np.array([0, 1.0, 0]))]:
        # 保持 p2,p3,p4 = pts[0], pts[2], pts[3], 只换 p1
        vtx, err = vertex_from_four(p1, pts[0], pts[2], pts[3])
        if vtx is None:
            print(f"  {tag:>22} {'失败: ' + err:>34}")
        else:
            print(f"  {tag:>22} {str(np.round(vtx, 8)):>34} "
                  f"{np.linalg.norm(vtx - true_vertex):>12.3e}")
    print()
    print("  => 只换 p1(拟合点 p2,p3,p4 不变), 顶点就变了。")
    print("     p1 定义 origin 与 xAxis, xAxis 决定 yAxis = 抛物线的轴方向。")
    print("     所以 4 个点各有职责: p1 定架, p2/p3/p4 拟合。")
    print()


def part3_three_vectors():
    print("=" * 78)
    print("3. 三个向量够不够?  —— 取决于是否继承坐标架")
    print("=" * 78)
    L, Z = 0.8, 0.2
    e1 = np.array([0.0, 0.0, 1.0])
    e2 = np.array([1.0, 0.0, 0.0])
    e3 = np.cross(e1, e2)

    def on_parab(L, Z, v):
        return (L - (L / (Z * Z)) * v * v) * e1 + v * e2

    va, vb, vc = -0.18, 0.02, 0.16
    p_a, p_b, p_c = (on_parab(L, Z, v) for v in (va, vb, vc))
    true_vertex = L * e1

    print(f"  真抛物线 L={L}, Z={Z}, 真顶点 = {np.round(true_vertex,6)}")
    print(f"  三个点取在曲线上: v = {va}, {vb}, {vc}")
    print()
    # (A) 继承坐标架 -> 3 点足够
    vtx_inherit = vertex_in_frame([p_a, p_b, p_c], e1, e2, e3)
    err_inherit = (np.linalg.norm(vtx_inherit - true_vertex)
                   if vtx_inherit is not None else float("nan"))
    print(f"  (A) 继承上一层坐标架 (e1,e2), 3 点拟合:")
    print(f"        顶点 = {np.round(vtx_inherit, 9)}   误差 = {err_inherit:.3e}")
    print()
    # (B) 每层重建坐标架 -> 3 点不够(欠定), 需要 4 点
    print("  (B) 每层重建坐标架 (当前代码做法):")
    vtx3, err3 = vertex_from_four(p_a, p_b, p_c, p_c)
    print(f"        只给 3 个点 -> "
          f"{'失败: ' + err3 if vtx3 is None else np.round(vtx3,8)}")
    vtx4, err4 = vertex_from_four(p_a, p_b, p_c, on_parab(L, Z, 0.0))
    print(f"        给 4 个点   -> 顶点 = {np.round(vtx4, 9)}   "
          f"误差 = {np.linalg.norm(vtx4 - true_vertex):.3e}")
    print()
    print("  => 只是换坐标架的来源, 误差从 "
          f"{err_inherit:.0e} 变成 {np.linalg.norm(vtx4-true_vertex):.0e};")
    print("     关键是自由度: 拟合需要 3 个未知数(a,b,c), 所以需要 3 个有效点【加上】")
    print("     一个坐标架。继承架 => 3 点即够; 重建架 => 架本身还要 2 个点, 合计 4。")
    print()
    print("  所以对'三个向量怎么求'的准确回答:")
    print("     三个向量【在继承坐标架的前提下】就够 —— 这正是层与层连续的自然要求。")
    print("     当前代码每层重建架, 所以必须凑 4 个点, 并因此承担两个代价:")
    print("       (i)  层与层不必共面, 交界点会漂出平面;")
    print("       (ii) 架由局部 3 点决定, 对输入抖动极敏感。")
    print()


def part4_failure():
    print("=" * 78)
    print("4. 失效模式: 拟合失败会静默把抛物线塌成直线")
    print("=" * 78)
    L, Z = 1.0, 0.3
    e1 = np.array([0.0, 0.0, 1.0])
    e2 = np.array([1.0, 0.0, 0.0])

    def on_parab(v):
        return (L - (L / (Z * Z)) * v * v) * e1 + v * e2

    print("  取三个 x 坐标几乎相同的点(p2,p3,p4 在抛物线顶点附近挤在一起):")
    print()
    print(f"  {'v2':>10} {'v3':>10} {'v4':>10} {'|det|':>12} {'结果':>26}")
    for v2, v3, v4 in [(0.01, 0.02, 0.03), (0.0, 0.001, 0.002),
                       (-0.2, 0.0, 0.2), (-0.25, 0.05, 0.28)]:
        p2, p3, p4 = on_parab(v2), on_parab(v3), on_parab(v4)
        p1 = on_parab(-0.29)
        X = [float(np.dot(p - p1, (p2 - p1) / np.linalg.norm(p2 - p1)))
             for p in (p2, p3, p4)]
        det = (X[0] ** 2 * (X[1] - X[2]) + X[1] ** 2 * (X[2] - X[0])
               + X[2] ** 2 * (X[0] - X[1]))
        vtx, err = vertex_from_four(p1, p2, p3, p4)
        res = f"顶点 {np.round(vtx,5)}" if vtx is not None else f"失败({err})"
        print(f"  {v2:>10.4f} {v3:>10.4f} {v4:>10.4f} {abs(det):>12.3e} {res:>26}")
    print()
    print("  det < 1e-10 时 GetVertexFromFourPoints 返回 Vector3.zero =>  vz2 = 0")
    print("  下游 GetParabolaSegmentPoints(0, ...) 里:")
    print("      zLength = |0| = 0;  e1 = Vector3.zero.normalized = Vector3.zero")
    print("      => uCoord*e1 = 0, 所有点塌到 vCoord*e2 这一条直线上,")
    print("         而 ComputeBendEnergy 里 k = 2*zLength/z2^2 = 0 -> 能量 f(v) = 0/sqrt = 0")
    print("  => 不报错、不抛异常, 只是这一层的结构静默消失。")
    print()


if __name__ == "__main__":
    part1_definition()
    part2_frame_role()
    part3_three_vectors()
    part4_failure()
    print("=" * 78)
    print("回答")
    print("=" * 78)
    print("  Q: 交接处究竟是什么?")
    print("  A: 当前空间曲线(抛物线)与由实体向量界定的弦的交点对。")
    print("     上层输出它, 下层消费它 —— 所以'每次更新的就是交接处'是对的。")
    print("     曲线本身不用携带(2 端点 + L=sqrt(r[2]) 即可重建)。")
    print()
    print("  Q: 两个向量怎么求?")
    print("  A: 一次二次方程。局部系里弦为 (L-Z,-Z)->(L,Z), 闭式:")
    print("         s± = [5L-1 ± sqrt((1-L)(1+7L))]/(8L)")
    print()
    print("  Q: 三个向量怎么求? 猜想'交接处直接与第三个向量相交就行'")
    print("  A: 一半对。交接处确实每步都更新, 但每步是【求交 + 定曲线】两步交替:")
    print("         求交: 弦 ∩ 抛物线      -> 二次方程")
    print("         定曲: 3 点拟合抛物线   -> 3x3 线性(Cramer)")
    print("     而'三个向量够不够'取决于坐标架: 继承 => 够(3 点 + 1 架);")
    print("     每层重建(当前代码) => 需 4 点(1 定架 + 3 拟合)。")
    print()
    print("  Q: 每次更新的就是交接处?")
    print("  A: 是。状态 = (2 个实体向量, L) + 交接处; 每步只改交接处, 成本 O(1)。")
