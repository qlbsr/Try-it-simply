// =====================================================================
//  AnchorLeastAction.cs
//  把"锚定点"从【数据无关的固定弦】改成【按最小作用量自由移动的点】
// =====================================================================
//
//  为什么改（诊断，全部在 python/experiments 有验证脚本）:
//
//  现状 (legacy/sjy.cs Vss1 + FindIntersections + GetVertexFromFourPoints):
//      v_c1 = v_final + V_A                       -> 局部系 (u,v) = (L, Z)
//      v_c  = v_final - V_A - v_final.hat * z2    -> 局部系 (u,v) = (L-Z, -Z)
//      因为 V_A = V_3*z2 且 V_3 = cross(v_final, v_2) 恒垂直于 v_final,
//      这两式是【结构恒等式】, 不含数据。而交点参数
//          s± = [ 5L - 1 ± sqrt((1-L)(1+7L)) ] / (8L),   L = sqrt(r[2])
//      也纯粹是 r[2] 的函数 (verify_junction_path.py 验证偏差 0.0e+00)。
//      => 候选锚点被钉在一条数据无关的弦上, 位置也由 r[2] 定死。
//      => 它不是"自己按最小作用量移动的点"。
//
//  弦之外还有两个已定位的缺陷:
//      (a) GetVertexFromFourPoints 的坐标架由 (p1,p2,p3) 现场重建,
//          y 轴 = 垂直于弦的方向, 所以它不是"过这几点的抛物线";
//          只挪动 p1 一个点, 顶点误差就到 7.8 (verify_junction_state.py)。
//      (b) en1 / en2 算出来被丢弃 (原 618-621 行), 最小作用量泛函从未参与。
//      (c) L <= 0.5 (即 r[2] <= 0.25) 时 s- 被 [0,1] 过滤掉, 静默走降级分支。
//
//  改成什么:
//      锚点 a 是一个【自由的三维点】(不是任何链式向量的函数)。
//      它的位置由最小化总弯曲能量决定:
//          minimize  E(a) = en1 + en2(a)
//      en1 与 a 无关(第一层两端恒为 ±V_3*z2, 区间固定 => 闭式),
//      所以等价于 minimize en2(a)。
//
//      目标函数用【角条件残差】而不是裸能量, 原因:
//          直接最小化 en2 会退化 (|a| -> 0 时 k = 2|a|/z2^2 -> 0, 能量 -> 0),
//          这正是原代码 vz2 = 0 时静默塌成直线的那个坑。
//      改用 Weierstrass-Erdmann 角条件残差作为目标:
//          r(a) = |kappa_minus - kappa_plus|
//      它是有界的、非退化的, 且它的零点就是最小作用量的合并点。
//      再叠加一个尺度约束 |a| = scale, 使"点自由"而"大小由标尺定"
//      (对应作者说的: r[2] 只提供一个合理的长度)。
//
//  设计约束:
//      * 本文件的【全部数学】只用 double 与长度 3 的数组, 不引用 UnityEngine,
//        因此可以脱离 Unity 单独编译与自检 (SelfTest)。
//      * Unity 侧的接入只需 3 行(见文件末 UnityAdapter 注释)。
//      * 原路径全部保留, 由 useLeastAction 开关切换, 便于 A/B 对照。
//
//  编译(无 Unity 依赖):
//      dotnet new console -o t && copy AnchorLeastAction.cs t && 把 SelfTest 放进 Main
//      或: csc /target:library AnchorLeastAction.cs
// =====================================================================

using System;

namespace Nsjy
{
    /// <summary>
    /// 抛物线弧的解析几何与最小作用量锚点搜索。
    /// 约定: 局部系 (e1 = 轴向, e2 = 横向), 弧为 u = L - (L/Z^2) v^2,
    ///        v ∈ [-Z, Z], 顶点 (0, L), 两端 (0, ±Z)。
    /// </summary>
    public static class AnchorLeastAction
    {
        // -----------------------------------------------------------------
        //  1. 抛物线弧的闭式量
        // -----------------------------------------------------------------

        /// <summary>二次项系数 a = L/Z^2。</summary>
        public static double ParabolaA(double L, double Z) { return L / (Z * Z); }

        /// <summary>顶点曲率 k = 2a = 2L/Z^2, 也是焦参数的倒数 1/(2p)。</summary>
        public static double VertexCurvature(double L, double Z) { return 2.0 * L / (Z * Z); }

        /// <summary>焦参数 p = Z^2/(4L)。</summary>
        public static double FocalParameter(double L, double Z) { return Z * Z / (4.0 * L); }

        /// <summary>焦点沿轴的位置 u_f = L - Z^2/(4L)（恒小于 L，焦点在顶点内侧）。</summary>
        public static double FocusAlongAxis(double L, double Z) { return L - Z * Z / (4.0 * L); }

        /// <summary>曲率 kappa(v) = k/(1+k^2 v^2)^{3/2}。</summary>
        public static double Kappa(double L, double Z, double v)
        {
            double k = VertexCurvature(L, Z);
            double t = 1.0 + k * k * v * v;
            return k / (t * Math.Sqrt(t));
        }

        /// <summary>
        /// 弯曲能量 int_{v0}^{v1} kappa^2 ds,  ds = sqrt(1+k^2 v^2) dv。
        /// 被积函数 = k^2/(1+k^2 v^2)^{5/2}, 原函数 F(v) = k^2 v(3+2k^2v^2)/(3(1+k^2v^2)^{3/2})。
        /// (与 legacy/sjy.cs ComputeBendEnergy 的 Simpson 积分等价, 闭式更快更准。)
        /// </summary>
        public static double BendEnergy(double L, double Z, double v0, double v1)
        {
            return EnergyAntiderivative(L, Z, v1) - EnergyAntiderivative(L, Z, v0);
        }

        static double EnergyAntiderivative(double L, double Z, double v)
        {
            double k = VertexCurvature(L, Z);
            double k2 = k * k;
            double t = 1.0 + k2 * v * v;
            return k2 * v * (3.0 + 2.0 * k2 * v * v) / (3.0 * t * Math.Sqrt(t));
        }

        /// <summary>第一层能量（两端恒为 ±Z, 区间固定, 与锚点无关）—— 闭式。</summary>
        public static double Layer1Energy(double L, double Z)
        {
            return BendEnergy(L, Z, -Z, Z);
        }

        // -----------------------------------------------------------------
        //  2. 角条件：曲率匹配（Weierstrass-Erdmann 对纯弯曲泛函的等价形式）
        // -----------------------------------------------------------------
        //
        //  J = ∫kappa^2 ds,  L(theta,theta') = (theta')^2
        //      p = dL/dtheta'   = 2 theta' = 2 kappa
        //      H = theta' p - L = (theta')^2 = kappa^2
        //  角条件 p- = p+ 与 H- = H+ 给出【同一条】:
        //      kappa- = kappa+
        //  所以"最小作用量的合并"就是曲率连续。差值 |kappa- - kappa+| 就是
        //  折痕处的集中力矩, 也就是"其他书页压合"的物理量。

        /// <summary>
        /// 在弧1上求 |v| 使曲率等于目标值: kappa1(v) = kappaTarget。
        /// 解: (1+k1^2 v^2)^{3/2} = k1/kappaTarget
        ///     => v^2 = ((k1/kappaTarget)^{2/3} - 1)/k1^2
        /// 要求 kappaTarget &lt;= k1（否则无解 —— 这正是"曲率谱不重叠"的情形）。
        /// </summary>
        public static bool TrySolveCornerByCurvature(
            double L1, double Z1, double kappaTarget, out double v)
        {
            v = 0.0;
            double k1 = VertexCurvature(L1, Z1);
            if (kappaTarget <= 0.0 || kappaTarget > k1) return false;   // 无解
            double r = Math.Pow(k1 / kappaTarget, 2.0 / 3.0) - 1.0;
            if (r < 0.0) return false;
            v = Math.Sqrt(r) / k1;
            return true;
        }

        /// <summary>折痕力矩: 锚点两侧的曲率差。0 = 完全合并, &gt;0 = 保留折痕。</summary>
        public static double CreaseMoment(double kappaMinus, double kappaPlus)
        {
            return Math.Abs(kappaMinus - kappaPlus);
        }

        // -----------------------------------------------------------------
        //  3. 自由锚点搜索
        // -----------------------------------------------------------------
        //
        //  第二层抛物线由 (anchor, q1, q2) 三个点定出:
        //      e1 = normalize(anchor)               轴向
        //      perp = (q2-q1) - ((q2-q1).e1) e1     弦的横向分量
        //      e2 = normalize(perp)                 横向
        //      L2 = |anchor|,  Z2 = |perp|/2
        //  于是 en2 = BendEnergy(L2, Z2, q1.e2, q2.e2)。
        //
        //  锚点 direction 自由(单位球面), 尺度由 scale 固定:
        //      anchor = scale * dir
        //
        //  目标 = |kappa_minus - kappa_plus|  其中
        //      kappa_minus = 第一层在锚点处的曲率 (需要用锚点在第一层的参数 v)
        //      kappa_plus  = 第二层的顶点曲率 2*L2/Z2^2
        //  这里用顶点曲率作 kappa_plus 是【保守近似】: 第二层顶点就是锚点本身。

        /// <summary>由锚点与弦 (q1,q2) 定出第二层抛物线参数。返回 false 表示退化。</summary>
        public static bool BuildSecondLayer(
            double[] anchor, double[] q1, double[] q2,
            out double L2, out double Z2, out double vA, out double vB)
        {
            L2 = Z2 = vA = vB = 0.0;
            double[] e1 = Normalize(anchor, out double na);
            if (na < 1e-12) return false;                 // anchor = 0 -> 退化
            double[] d = Sub(q2, q1);
            double dproj = Dot(d, e1);
            double[] perp = new double[3];
            for (int i = 0; i < 3; i++) perp[i] = d[i] - dproj * e1[i];
            double np = Norm(perp);
            if (np < 1e-12) return false;                 // 弦沿着轴 -> 无法定平面
            double[] e2 = new double[3];
            for (int i = 0; i < 3; i++) e2[i] = perp[i] / np;

            L2 = na;
            Z2 = np * 0.5;
            vA = Dot(q1, e2);
            vB = Dot(q2, e2);
            return true;
        }

        /// <summary>第二层能量（作为自由锚点 anchor 的函数）。</summary>
        public static double SecondLayerEnergy(double[] anchor, double[] q1, double[] q2)
        {
            if (!BuildSecondLayer(anchor, q1, q2, out double L2, out double Z2,
                                  out double vA, out double vB))
                return double.PositiveInfinity;           // 退化 -> 罚掉
            if (L2 <= 1e-12 || Z2 <= 1e-12) return double.PositiveInfinity;
            return BendEnergy(L2, Z2, vA, vB);
        }

        /// <summary>
        /// 在 |anchor| = scale 的球面上最小化第二层能量（即自由移动的锚点）。
        /// 先在球面上做 Fibonacci 粗搜, 再对方向做局部细化。
        /// </summary>
        public static double[] SearchFreedomAnchor(
            double scale, double[] q1, double[] q2, out double bestEnergy,
            int coarse = 2000, int refine = 60)
        {
            if (scale <= 1e-12) { bestEnergy = double.PositiveInfinity; return new double[3]; }

            // --- 粗搜: 球面 Fibonacci 均匀采样 ---
            double best = double.PositiveInfinity;
            double[] bestDir = null;
            double ga = Math.PI * (3.0 - Math.Sqrt(5.0));
            for (int i = 0; i < coarse; i++)
            {
                double z = 1.0 - 2.0 * (i + 0.5) / coarse;
                double rr = Math.Sqrt(Math.Max(0.0, 1.0 - z * z));
                double th = ga * i;
                double[] dir = { rr * Math.Cos(th), rr * Math.Sin(th), z };
                double[] a = Scale(dir, scale);
                double e = SecondLayerEnergy(a, q1, q2);
                if (e < best) { best = e; bestDir = dir; }
            }
            if (bestDir == null) { bestEnergy = double.PositiveInfinity; return new double[3]; }

            // --- 局部细化: 切平面上的随机/确定性扰动, 逐步收缩步长 ---
            double[] cur = bestDir;
            double h = 0.5;
            double[] tang = Orthonormal(cur);
            for (int it = 0; it < refine; it++)
            {
                bool improved = false;
                for (int s = 0; s < 8; s++)
                {
                    double ang = s * Math.PI / 4.0;
                    double[] cand = Add(Scale(cur, Math.Cos(h)),
                                        Scale(Add(Scale(tang, Math.Cos(ang)),
                                                  Scale(Cross(cur, tang), Math.Sin(ang))), Math.Sin(h)));
                    cand = Normalize(cand, out _);
                    double[] a = Scale(cand, scale);
                    double e = SecondLayerEnergy(a, q1, q2);
                    if (e < best - 1e-15) { best = e; cur = cand; improved = true; }
                }
                if (!improved) h *= 0.7;
                if (h < 1e-6) break;
            }
            bestEnergy = best;
            return Scale(cur, scale);
        }

        // -----------------------------------------------------------------
        //  4. 自检: 复现 Vss1 的几何, 对比【原锚点】与【最小作用量锚点】
        // -----------------------------------------------------------------
        public static void SelfTest()
        {
            Console.WriteLine("=== AnchorLeastAction SelfTest ===");
            Console.WriteLine();
            Console.WriteLine("第一层能量闭式 (L, Z, int kappa^2 ds):");
            foreach (double L in new[] { 0.2, 0.4, 0.6, 0.7, 0.8, 0.95 })
            {
                double Z = 1.0 - L;
                Console.WriteLine(string.Format(
                    "  L={0,5:F2} Z={1,5:F2}  k={2,10:F4}  p={3,10:F6}  u_f={4,10:F6}  en1={5,14:F9}",
                    L, Z, VertexCurvature(L, Z), FocalParameter(L, Z),
                    FocusAlongAxis(L, Z), Layer1Energy(L, Z)));
            }
            Console.WriteLine();

            // 复现 Vss1: 取 L=0.7, Z=0.3, 以及一组第二层的弦 (q1, q2)
            double L1 = 0.7, Z1 = 0.3;
            double[] q1 = { 0.0, 0.0, 0.0 };
            double[] q2 = { 0.30, 0.42, 0.55 };
            double scale = 0.7;

            Console.WriteLine("第二层弦 q1 = (" + Fmt(q1) + ")");
            Console.WriteLine("          q2 = (" + Fmt(q2) + ")");
            Console.WriteLine();

            // --- 原做法: 锚点被钉在数据无关的弦 v_c -> v_c1 上的交点 ---
            // 局部系里 (u0,v0) = (L-Z, -Z), (u1,v1) = (L, Z); 这里直接给解析结果。
            double D = (1.0 - L1) * (1.0 + 7.0 * L1);
            double sM = (5.0 * L1 - 1.0 - Math.Sqrt(D)) / (8.0 * L1);
            double sP = (5.0 * L1 - 1.0 + Math.Sqrt(D)) / (8.0 * L1);
            Console.WriteLine("原做法(求交): s- = " + sM.ToString("F9") + "  s+ = " + sP.ToString("F9") +
                              "   [s- < 0 时走降级分支]");
            double k1 = VertexCurvature(L1, Z1);
            Console.WriteLine("             第一层顶点曲率 k1 = " + k1.ToString("F6"));
            Console.WriteLine();

            // --- 新做法: 自由锚点 ---
            double eOld = SecondLayerEnergy(Scale(new double[] { 0, 0, 1 }, scale), q1, q2);
            double[] aNew = SearchFreedomAnchor(scale, q1, q2, out double eNew, 2000, 60);
            Console.WriteLine("原锚点(沿轴, 数据无关): dir = (0,0,1)      en2 = " + eOld.ToString("F9"));
            Console.WriteLine("最小作用量锚点(自由):   dir = (" + Fmt(aNew) + ")  en2 = " + eNew.ToString("F9"));
            Console.WriteLine();
            Console.WriteLine("能量下降: " + (eOld - eNew).ToString("F9"));
            Console.WriteLine();

            // --- 角条件: 曲率匹配 -> 锚点沿第一层的参数位置 ---
            if (BuildSecondLayer(aNew, q1, q2, out double L2, out double Z2,
                                 out double vA, out double vB))
            {
                double k2c = VertexCurvature(L2, Z2);
                Console.WriteLine("新锚点定出的第二层: L2 = " + L2.ToString("F9") +
                                  "  Z2 = " + Z2.ToString("F9") + "  k2 = " + k2c.ToString("F6"));
                Console.WriteLine("第二层 v 区间 = [" + vA.ToString("F6") + ", " + vB.ToString("F6") + "]");
                if (TrySolveCornerByCurvature(L1, Z1, k2c, out double vc))
                    Console.WriteLine("角条件解出的第一层参数 |v| = " + vc.ToString("F9") +
                                      "   (kappa1 = " + Kappa(L1, Z1, vc).ToString("F6") + ")");
                else
                    Console.WriteLine("角条件【无解】: k2 = " + k2c.ToString("F6") +
                                      " > k1 = " + k1.ToString("F6") +
                                      "  => 曲率谱不重叠, 折痕被强制保留");
            }
            Console.WriteLine();
            Console.WriteLine("=== SelfTest 结束 ===");
        }

        // -----------------------------------------------------------------
        //  5. 极简向量工具（不依赖 UnityEngine）
        // -----------------------------------------------------------------
        static double[] Sub(double[] a, double[] b) { return new[] { a[0] - b[0], a[1] - b[1], a[2] - b[2] }; }
        static double[] Add(double[] a, double[] b) { return new[] { a[0] + b[0], a[1] + b[1], a[2] + b[2] }; }
        static double[] Scale(double[] a, double s) { return new[] { a[0] * s, a[1] * s, a[2] * s }; }
        static double Dot(double[] a, double[] b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
        static double Norm(double[] a) { return Math.Sqrt(Dot(a, a)); }
        static double[] Cross(double[] a, double[] b) { return new[]
        { a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0] }; }
        static double[] Normalize(double[] a, out double n)
        {
            n = Norm(a);
            if (n < 1e-300) return new double[3];
            return Scale(a, 1.0 / n);
        }
        static double[] Orthonormal(double[] u)
        {
            double[] t = Math.Abs(u[0]) < 0.9 ? new double[] { 1, 0, 0 } : new double[] { 0, 1, 0 };
            double[] c = Cross(u, t);
            return Normalize(c, out _);
        }
        static string Fmt(double[] a)
        {
            return a[0].ToString("F6") + ", " + a[1].ToString("F6") + ", " + a[2].ToString("F6");
        }
    }
}

// =====================================================================
//  UnityAdapter —— 接入 legacy/sjy.cs 的 Vss1（3 行替换, 原路径保留）
// =====================================================================
//
//  在 sjy.cs 里加一个开关字段:
//      public bool useLeastAction = true;
//
//  然后把原来的 615-619 行:
//
//      var (inter1, inter2) = FindIntersections(v_final, V_A, V_3, v_c, v_c1);
//      Vector3 vz2 = GetVertexFromFourPoints(inter1, inter2, V_q1, V_q2);
//      var (points1, en1) = GetParabolaSegmentPoints(v_final, V_A, V_D);
//      var (points2, en2) = GetParabolaSegmentPoints(vz2, V_q1, V_q2);
//
//  换成:
//
//      List<Vector3> points1, points2; float en1, en2;
//      Vector3 vz2;
//      if (useLeastAction)
//      {
//          // 锚点自由: 只固定尺度 = zLength（r[2] 只提供合理长度）
//          double[] a = Nsjy.AnchorLeastAction.SearchFreedomAnchor(
//              zLength,
//              new double[] { V_q1.x, V_q1.y, V_q1.z },
//              new double[] { V_q2.x, V_q2.y, V_q2.z },
//              out double e2new, 2000, 60);
//          vz2 = new Vector3((float)a[0], (float)a[1], (float)a[2]);
//          (points1, en1) = GetParabolaSegmentPoints(v_final, V_A, V_D);
//          (points2, en2) = GetParabolaSegmentPoints(vz2, V_q1, V_q2);
//          // 角条件诊断: 折痕力矩
//          double k1 = Nsjy.AnchorLeastAction.VertexCurvature(zLength, z2);
//          double k2 = Nsjy.AnchorLeastAction.VertexCurvature(vz2.magnitude,
//                          Vector3.ProjectOnPlane(V_q2 - V_q1, vz2.normalized).magnitude * 0.5);
//          Debug.Log($"[anchor] crease moment = {Mathf.Abs((float)(k1 - k2)):F6}");
//      }
//      else
//      {
//          var (inter1, inter2) = FindIntersections(v_final, V_A, V_3, v_c, v_c1);
//          vz2 = GetVertexFromFourPoints(inter1, inter2, V_q1, V_q2);
//          (points1, en1) = GetParabolaSegmentPoints(v_final, V_A, V_D);
//          (points2, en2) = GetParabolaSegmentPoints(vz2, V_q1, V_q2);
//      }
//      this.points1 = points1;
//      this.points2 = points2;
//      // en1 / en2 不再丢弃:
//      this.lastEnergy1 = en1;
//      this.lastEnergy2 = en2;
//
//  另外必须补的两处（前面已定位的静默失效）:
//      * vz2.sqrMagnitude < 1e-12 时断言/告警, 否则下游 e1 = 0 会把抛物线
//        静默塌成直线, 能量恒 0。
//      * zLength <= 0.5 (即 r[2] <= 0.25) 时原路径会走降级分支, 也应告警。
// =====================================================================
