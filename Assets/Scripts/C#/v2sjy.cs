using System;
using System.Collections.Generic;
using UnityEngine;
using Complex = System.Numerics.Complex;

/// <summary>
/// 透镜体积的简化计算 + 让 V2 以"刚度"的身份真正进入 ABCD。
///
/// 化简依据 (见 verify_v2_simplified.py / verify_lens_volume_identity.py):
///
///   1) 原 ComputeAllFeatures(sjy1.cs:386) 里 R = d/g,  x = 1 - d^2/(4R^2),
///      而 d^2/(4R^2) = g^2/4  ——  d 完全约掉。所以形状参数只由 g = ||dJ/ds|| 决定:
///
///          x = 1 - g^2/4          <-- 只依赖雅可比场的导数
///
///   2) I_x(5/2,1/2) 有初等闭式 (因为 (1/2)I_{sin^2 a}((n+1)/2,1/2) 是 n 维球冠体积分数,
///      (5/2,1/2) 唯一对应 n = 4):
///
///          Phi(g) = [12a - 8 sin2a + sin4a] / (6*pi),   a = arccos(g/2)
///
///      不需要 MathNet / 连分数 / Lanczos Gamma。a 很小时用级数避免相消。
///
///   3) 于是整条链只要两个标量:  d = ||J||,  g = ||dJ/ds||
///
///          I = d/g
///          V = (d/g)^4 * Phi(g)        (= (4/pi^2) * Vol_4(半径 I 的 4 维球冠))
///
///   4) g = d/(2R) 就是两个球的"重叠参数": 相交 <=> d <= 2R <=> g <= 2。
///      实现里 g >= 2 时 Phi 必须取 0 (原代码用 Mathf.Clamp01(x) 达到同样效果)。
/// </summary>
public static class LensVolumeSimple
{
    public const float GPolar = 2f;   // g 的上界: 透镜相切

    // ============================================================ Phi(g)

    /// <summary>Phi(g) = I_{1-g^2/4}(5/2, 1/2) = (4/pi^2)*Vol4(球冠)/(I^4)</summary>
    public static double Phi(double g)
    {
        if (g <= 0.0) return 1.0;
        if (g >= GPolar) return 0.0;

        double u = g * 0.5;                 // = cos(alpha)
        double alpha = Math.Acos(u);

        // a 很小时 12a 与 8sin2a 近相消。Taylor:
        //   12a - 8sin2a + sin4a = 6.4 a^5 - (15360/5040) a^7 + (258048/362880) a^9 - ...
        if (alpha < 0.05)
            return alpha * alpha * alpha * alpha * alpha
                   * (6.4 - (15360.0 / 5040.0) * alpha * alpha
                        + (258048.0 / 362880.0) * alpha * alpha * alpha * alpha)
                   / (6.0 * Math.PI);

        return (12.0 * alpha - 8.0 * Math.Sin(2.0 * alpha) + Math.Sin(4.0 * alpha))
               / (6.0 * Math.PI);
    }

    // ==================================================== 逐点 (d, g, I, V)

    public struct Lens
    {
        public float d;   // ||J||          两路径间距
        public float g;   // ||dJ/ds||      雅可比场的导数 (重叠参数)
        public float I;   // d/g            "刚度/曲率半径"
        public float V;   // (d/g)^4 * Phi  4 维球冠体积 (透镜体积)
        public float alpha; // arccos(g/2)  球冠半角(弧度)
    }

    /// <summary>逐点求 (d, g, I, V)。只要两条等参数化的路径。</summary>
    public static Lens[] Compute(List<Vector3> pointsA, List<Vector3> pointsB)
    {
        if (pointsA == null || pointsB == null) return new Lens[0];
        int N = Mathf.Min(pointsA.Count, pointsB.Count);
        var L = new Lens[N];
        if (N < 2) return L;

        // 1. 雅可比场 J 与间距 d
        var J = new Vector3[N];
        for (int i = 0; i < N; i++)
        {
            J[i] = pointsB[i] - pointsA[i];
            L[i].d = J[i].magnitude;
        }

        // 2. 局部弧长步长 (两条路径平均)
        var ds = new float[N];
        for (int i = 1; i < N; i++)
            ds[i] = 0.5f * (Vector3.Distance(pointsA[i], pointsA[i - 1])
                          + Vector3.Distance(pointsB[i], pointsB[i - 1]));
        ds[0] = ds[1];

        // 3. g, I, V   —— 这里就是全部, 没有 R / R^2 / x / Clamp01 / 不完全Beta
        const float eps = 1e-6f;
        for (int i = 0; i < N; i++)
        {
            Vector3 gJ;
            if (i == 0) gJ = (J[1] - J[0]) / Mathf.Max(ds[1], eps);
            else if (i == N - 1) gJ = (J[N - 1] - J[N - 2]) / Mathf.Max(ds[N - 1], eps);
            else
            {
                float step = Mathf.Max(0.5f * (ds[i] + ds[i + 1]), eps);
                gJ = (J[i + 1] - J[i - 1]) / (2f * step);
            }
            L[i].g = gJ.magnitude + eps;

            float I = L[i].d / L[i].g;
            L[i].I = I;
            L[i].V = I * I * I * I * (float)Phi(L[i].g);
            L[i].alpha = Mathf.Acos(Mathf.Clamp(L[i].g * 0.5f, -1f, 1f));
        }
        return L;
    }

    /// <summary>与原 LocalLensVolumeExtractor.ComputeAllFeatures 同签名, 可直接替换调用。</summary>
    public static void ComputeAllFeatures(List<Vector3> pointsA, List<Vector3> pointsB,
                                          out float[] I_arr, out float[] V_arr)
    {
        var L = Compute(pointsA, pointsB);
        int N = L.Length;
        I_arr = new float[N];
        V_arr = new float[N];
        for (int i = 0; i < N; i++) { I_arr[i] = L[i].I; V_arr[i] = L[i].V; }
    }

    // ==================================================== 归一化 (关键)

    /// <summary>
    /// V 跨 10 个数量级 (2e-11 ~ 0.76, 中位/max = 5.8e-5)。按 max 归一化会把 70% 的点
    /// 压到相位 &lt; 1 度。Vn 用 log 归一化铺到 [0,1], 相位 &lt;1 度的点降到 1%。
    /// </summary>
    public static float[] LogNormalize(float[] V)
    {
        int N = V.Length;
        var outp = new float[N];
        if (N == 0) return outp;
        double lo = double.MaxValue, hi = double.MinValue;
        for (int i = 0; i < N; i++)
        {
            double lv = Math.Log(Math.Max(V[i], 1e-30));
            if (lv < lo) lo = lv;
            if (lv > hi) hi = lv;
        }
        double span = Math.Max(hi - lo, 1e-9);
        for (int i = 0; i < N; i++)
            outp[i] = (float)((Math.Log(Math.Max(V[i], 1e-30)) - lo) / span);
        return outp;
    }

    /// <summary>
    /// 把 L 条 lane 展开到 nUv 个 uv 点。
    ///   Pair : uv 点 i 用 lane i/2       (双锥两叶配对 —— 与 th4 每 lane 出两条一致)
    ///   Mod  : uv 点 i 用 lane i%L
    /// 注意: V_arr 的长度是 points1.Count(=100), 而 uvs.Length 通常是 200,
    ///       直接写 V_arr[i] 会 IndexOutOfRange。
    /// </summary>
    public enum LaneMap { Pair = 0, Mod = 1 }

    public static float[] Expand(float[] VnLanes, int nUv, LaneMap map = LaneMap.Pair)
    {
        int L = VnLanes.Length;
        var o = new float[nUv];
        if (L == 0) { for (int i = 0; i < nUv; i++) o[i] = 1f; return o; }
        for (int i = 0; i < nUv; i++)
        {
            int k = (map == LaneMap.Pair) ? Mathf.Min(i / 2, L - 1) : (i % L);
            o[i] = VnLanes[k];
        }
        return o;
    }

    // ==================================================== 刚度项

    /// <summary>
    /// 刚度投影势。物理上"刚度 × 曲率": stress = stiffness * strain。
    /// 原 sjy.cs:1014 只有 0.5*H*z (纯曲率), 名字叫 stiffness 却没有 stiffness ——
    /// 乘上透镜体积后才是刚度为 V 的介质里的曲率项。
    /// </summary>
    /// <param name="grad_du">NumericalGradient(uvs, dw/du)  : H_uu + i H_uv</param>
    /// <param name="grad_dv">NumericalGradient(uvs, dw/dv)  : H_uv + i H_vv</param>
    /// <param name="Vn">每个 uv 点的刚度权重 (null 则退化为原版)</param>
    public static Complex[] StiffnessProjection(Vector2[] uvs, Complex[] grad_du,
                                               Complex[] grad_dv, float[] Vn = null)
    {
        int N = uvs.Length;
        var outp = new Complex[N];
        for (int i = 0; i < N; i++)
        {
            float u = uvs[i].x, v = uvs[i].y;
            double H_uu = grad_du[i].Real;
            double H_uv = grad_du[i].Imaginary;
            double H_vv = grad_dv[i].Imaginary;

            double re = 0.5 * (H_uu * u + H_uv * v);
            double im = 0.5 * (H_uv * u + H_vv * v);

            float w = (Vn == null) ? 1f : Vn[i];
            outp[i] = new Complex(w * re, w * im);
        }
        return outp;
    }

    /// <summary>合成 totalGrad:  6 * (stiff + AB + quat)</summary>
    public static Complex[] TotalGrad(Complex[] stiff, Complex[] AB, Complex[] quat,
                                      float k = 6f)
    {
        int N = stiff.Length;
        var o = new Complex[N];
        for (int i = 0; i < N; i++)
            o[i] = k * (stiff[i] + (AB == null ? Complex.Zero : AB[i])
                              + (quat == null ? Complex.Zero : quat[i]));
        return o;
    }

    // ==================================================================
    //  叶面透镜场 —— 取代 Vss1 + ComputeAllFeatures + RBF 三条链路
    //
    //  双锥两叶在叶点 (u,v) 的像就是天然的 pointsA / pointsB:
    //      A = qs  * vs3(u,v)      B = qs2 * vs3(u,v)
    //      J = B - A
    //      d = ||J||
    //      g = || [dJ/du, dJ/dv] ||_F        叶面雅可比的 Frobenius 范数
    //  然后 (d,g) -> (I,V) 走闭式。梯度/Hessian 都是精确的, 不需要插值器。
    // ==================================================================

    public struct LeafLens
    {
        public float[] d;   // ||J||
        public float[] g;   // ||[dJ/du, dJ/dv]||_F
        public float[] I;   // d/g
        public float[] V;   // (d/g)^4 * Phi(g)
        public int Count;
    }

    /// <summary>th4 的两个叶 (不归一化), 返回 A, B 与 J = B - A</summary>
    public static void NappePair(Vector2 uv, float rp, double d2Rad, Quaternion qs,
                                 Quaternion qs2, out Vector3 A, out Vector3 B)
    {
        float Rs = uv.magnitude;
        float thetas = Mathf.Atan2(uv.y, uv.x);
        if (thetas < 0f) thetas += 2f * Mathf.PI;
        double sd = Math.Sin(d2Rad), cd = Math.Cos(d2Rad);
        double r = rp * Math.Pow(Rs, sd);
        double phis = thetas * sd;
        var vs3 = new Vector3((float)(r * sd * Math.Cos(phis)),
                              (float)(r * sd * Math.Sin(phis)),
                              (float)(r * cd));
        A = qs * vs3;
        B = qs2 * vs3;
    }

    /// <summary>逐点求叶面透镜场: (u,v) -> d, g, I, V。不用 Vss1, 不用排序。</summary>
    public static LeafLens ComputeLeafLens(Vector2[] uvs, float rp, float d2deg,
                                           Vector3 v3, int K = 16)
    {
        int N = (uvs == null) ? 0 : uvs.Length;
        var L = new LeafLens { d = new float[N], g = new float[N],
                               I = new float[N], V = new float[N], Count = N };
        if (N == 0) return L;

        double d2Rad = d2deg * Math.PI / 180.0;
        var vn = v3.normalized;
        var qs = Quaternion.FromToRotation(new Vector3(0f, 0f, 1f), vn);
        var qs2 = Quaternion.FromToRotation(new Vector3(0f, 0f, 1f), -vn);

        var J = new Vector3[N];
        for (int i = 0; i < N; i++)
        {
            Vector3 A, B;
            NappePair(uvs[i], rp, d2Rad, qs, qs2, out A, out B);
            J[i] = B - A;
            L.d[i] = J[i].magnitude;
        }

        var M = new double[K, 3];
        var b = new double[K, 3];
        var sol = new double[3, 3];
        for (int i = 0; i < N; i++)
        {
            int m = 0;
            // 取 K 近邻 (不排序, 按 uv 距离)
            int[] idx = NearestK(uvs, i, K);
            m = idx.Length;
            if (m < 4) { L.g[i] = 1f; L.I[i] = L.d[i]; L.V[i] = 0f; continue; }
            for (int k = 0; k < m; k++)
            {
                int j = idx[k];
                M[k, 0] = uvs[j].x; M[k, 1] = uvs[j].y; M[k, 2] = 1.0;
                b[k, 0] = J[j].x; b[k, 1] = J[j].y; b[k, 2] = J[j].z;
            }
            for (int c = 0; c < 3; c++)
            {
                double[] col = new double[m];
                for (int k = 0; k < m; k++) col[k] = b[k, c];
                if (!LeastSquares3(M, col, m, out double[] s))
                { L.g[i] = 1f; L.I[i] = L.d[i]; L.V[i] = 0f; continue; }
                sol[0, c] = s[0]; sol[1, c] = s[1]; sol[2, c] = s[2];
            }
            // Ju = (sol[0,0], sol[0,1], sol[0,2]),  Jv = (sol[1,*])
            double ju2 = sol[0,0]*sol[0,0] + sol[0,1]*sol[0,1] + sol[0,2]*sol[0,2];
            double jv2 = sol[1,0]*sol[1,0] + sol[1,1]*sol[1,1] + sol[1,2]*sol[1,2];
            float gg = (float)Math.Sqrt(ju2 + jv2) + 1e-6f;
            L.g[i] = gg;
            float I = L.d[i] / gg;
            L.I[i] = I;
            L.V[i] = I * I * I * I * (float)Phi(gg);
        }
        return L;
    }

    static int[] NearestK(Vector2[] uvs, int i, int K)
    {
        int N = uvs.Length;
        var dist = new float[N];
        for (int j = 0; j < N; j++)
        {
            float dx = uvs[j].x - uvs[i].x, dy = uvs[j].y - uvs[i].y;
            dist[j] = dx * dx + dy * dy;
        }
        var ord = new int[N];
        for (int j = 0; j < N; j++) ord[j] = j;
        Array.Sort(ord, (a, c) => dist[a].CompareTo(dist[c]));
        int m = 0, cap = Mathf.Min(K, N);
        var o = new int[cap];
        for (int j = 0; j < N && m < cap; j++)
            if (dist[ord[j]] > 1e-24f) o[m++] = ord[j];
        if (m == cap) return o;
        var t = new int[m];
        Array.Copy(o, t, m);
        return t;
    }

    /// <summary>最小二乘 solve  M(m,3) * s(3) = col(m), 用 3x3 正规方程 + 全主元</summary>
    static bool LeastSquares3(double[,] M, double[] col, int m, out double[] s)
    {
        s = new double[3];
        var A = new double[3, 3];
        var rhs = new double[3];
        for (int a = 0; a < 3; a++)
        {
            for (int c = 0; c < 3; c++)
            {
                double t = 0;
                for (int k = 0; k < m; k++) t += M[k, a] * M[k, c];
                A[a, c] = t;
            }
            double u = 0;
            for (int k = 0; k < m; k++) u += M[k, a] * col[k];
            rhs[a] = u;
        }
        // Gauss with partial pivoting
        for (int p = 0; p < 3; p++)
        {
            int piv = p;
            for (int r = p + 1; r < 3; r++)
                if (Math.Abs(A[r, p]) > Math.Abs(A[piv, p])) piv = r;
            if (Math.Abs(A[piv, p]) < 1e-14) return false;
            if (piv != p)
            {
                for (int c = 0; c < 3; c++) { double t = A[p, c]; A[p, c] = A[piv, c]; A[piv, c] = t; }
                double tt = rhs[p]; rhs[p] = rhs[piv]; rhs[piv] = tt;
            }
            for (int r = p + 1; r < 3; r++)
            {
                double f = A[r, p] / A[p, p];
                for (int c = p; c < 3; c++) A[r, c] -= f * A[p, c];
                rhs[r] -= f * rhs[p];
            }
        }
        for (int r = 2; r >= 0; r--)
        {
            double t = rhs[r];
            for (int c = r + 1; c < 3; c++) t -= A[r, c] * s[c];
            s[r] = t / A[r, r];
        }
        return true;
    }

    // ==================================================================
    //  概率场的 Hessian —— 取代 SphericalRBF(X(V2)) 那条链
    //  概率本来就在 uv 面上采样, 直接叶面局部二次拟合:
    //      w ≈ a + b*du + c*dv + d*du^2 + e*du*dv + f*dv^2
    //      H = [[2d, e], [e, 2f]]
    // ==================================================================

    public struct Hess
    {
        public float[] H_uu, H_uv, H_vv;
    }

    public static Hess LocalQuadHessian(Vector2[] uvs, float[] w, int K = 12)
    {
        int N = uvs.Length;
        var H = new Hess { H_uu = new float[N], H_uv = new float[N], H_vv = new float[N] };
        var M = new double[K, 6];
        for (int i = 0; i < N; i++)
        {
            var idx = NearestK(uvs, i, K);
            int m = idx.Length;
            if (m < 6) continue;
            var col = new double[m];
            for (int k = 0; k < m; k++)
            {
                int j = idx[k];
                double du = uvs[j].x - uvs[i].x, dv = uvs[j].y - uvs[i].y;
                M[k, 0] = 1.0; M[k, 1] = du; M[k, 2] = dv;
                M[k, 3] = du * du; M[k, 4] = du * dv; M[k, 5] = dv * dv;
                col[k] = w[j];
            }
            if (!LeastSquares(M, col, m, 6, out double[] s)) continue;
            H.H_uu[i] = (float)(2.0 * s[3]);
            H.H_uv[i] = (float)s[4];
            H.H_vv[i] = (float)(2.0 * s[5]);
        }
        return H;
    }

    static bool LeastSquares(double[,] M, double[] col, int m, int n, out double[] s)
    {
        s = new double[n];
        var A = new double[n, n];
        var rhs = new double[n];
        for (int a = 0; a < n; a++)
        {
            for (int c = 0; c < n; c++)
            {
                double t = 0;
                for (int k = 0; k < m; k++) t += M[k, a] * M[k, c];
                A[a, c] = t;
            }
            double u = 0;
            for (int k = 0; k < m; k++) u += M[k, a] * col[k];
            rhs[a] = u;
        }
        for (int p = 0; p < n; p++)
        {
            int piv = p;
            for (int r = p + 1; r < n; r++)
                if (Math.Abs(A[r, p]) > Math.Abs(A[piv, p])) piv = r;
            if (Math.Abs(A[piv, p]) < 1e-16) return false;
            if (piv != p)
            {
                for (int c = 0; c < n; c++) { double t = A[p, c]; A[p, c] = A[piv, c]; A[piv, c] = t; }
                double tt = rhs[p]; rhs[p] = rhs[piv]; rhs[piv] = tt;
            }
            for (int r = p + 1; r < n; r++)
            {
                double f = A[r, p] / A[p, p];
                for (int c = p; c < n; c++) A[r, c] -= f * A[p, c];
                rhs[r] -= f * rhs[p];
            }
        }
        for (int r = n - 1; r >= 0; r--)
        {
            double t = rhs[r];
            for (int c = r + 1; c < n; c++) t -= A[r, c] * s[c];
            s[r] = t / A[r, r];
        }
        return true;
    }

    /// <summary>从 Hessian 直接构造刚度投影势 (不再需要 NumericalGradient 两层求导)</summary>
    public static Complex[] StiffnessProjectionFromH(Vector2[] uvs, Hess H, float[] Vn = null)
    {
        int N = uvs.Length;
        var o = new Complex[N];
        for (int i = 0; i < N; i++)
        {
            double u = uvs[i].x, v = uvs[i].y;
            double re = 0.5 * (H.H_uu[i] * u + H.H_uv[i] * v);
            double im = 0.5 * (H.H_uv[i] * u + H.H_vv[i] * v);
            float wgt = (Vn == null) ? 1f : Vn[i];
            o[i] = new Complex(wgt * re, wgt * im);
        }
        return o;
    }

    // ==================================================================
    //  ABCD 四通道
    //
    //      dU/dz = A z + B zbar + C z^2 + D zbar^2 + 高阶项
    //
    //  A,B,C,D 不是四个独立的物理实体, 也不是可单独调节的参数, 而是"外部势梯度"
    //  在复平面上低阶展开后的四个复系数通道, 由刚度 / AB 色散 / 四元数旋转三类
    //  贡献在【全体点集上】整体最小二乘拟合得到。改变任意一点, 四者会一起变,
    //  所以它们对应的是整体性破缺的四个通道。
    //
    //    A  线性全纯     刚度各向同性 (c11+c22)/2; AB 的 alpha*z; 四元数 i*dphi/dz*z
    //                    线性全纯项, 保持旋转对称 / 整体相位
    //                    -> 纯 A 时解为圆; 决定整体尺度, 径向增长, 整体旋转相位
    //    B  线性非全纯   刚度各向异性 (c11-c22)/2 + i*c12; AB 偶极 beta*zbar
    //                    最低阶旋转对称破缺
    //                    -> B != 0 时圆变椭圆; 是一阶破缺的直接标志
    //    C  二次全纯     AB 高阶色散 gamma*z^2; 四元数强梯度的全纯部分
    //                    高阶形变
    //                    -> 把椭圆推向非对称卵形线 / 对数螺旋带
    //    D  二次非全纯   AB 高阶色散 delta*zbar^2; 四元数强梯度的非全纯 / 非对易部分
    //                    高阶破缺与手性扭曲
    //                    -> 非紧致开口, 无限叶, 无理斜率螺旋
    //
    //  判别式  Delta = (B^2 - 4AC) / (D^2 - 4BC)
    //          Delta ∈ Q  -> 支撑集紧致 (圆 / 椭圆 / 卵形线)
    //          Delta ∉ Q  -> 支撑集非紧致, 开口螺旋 ("无限叶")
    //
    //  全纯项只含 z (A z, C z^2); 非全纯项含 zbar (B zbar, D zbar^2)。
    //  非全纯项的出现, 就是旋转对称破缺的数学源头。
    //
    //  对照: sjysjy.cs 的 Discriminant(A,B,C,D) 正是这个 Delta;
    //        legacy\sjy.cs 的 BuildABCD 只用了分子 B^2-4AC, 丢掉了 D, 是不完整的。
    //
    //  可辨识性: 若 uv 点云落在同一条圆上 (|z| 恒定), 则 zbar = R^2/z 且
    //    zbar^2 = R^4/z^2, 四个基列只剩 2 维张成, 四通道不可辨识。
    //    要求 uv 的半径有变化 (叶片点云通常满足)。
    // ==================================================================

    public struct Channels
    {
        public Complex A, B, C, D;
        public int N;                 // 参与拟合的点数
        public double ResidualRms;    // 拟合残差 RMS

        public double Norm { get { return Math.Sqrt(Sq(A) + Sq(B) + Sq(C) + Sq(D)); } }
        public double HoloNorm { get { return Math.Sqrt(Sq(A) + Sq(C)); } }   // 全纯 (只含 z)
        public double AntiNorm { get { return Math.Sqrt(Sq(B) + Sq(D)); } }   // 非全纯 (含 zbar)
        public double HoloFraction { get { double n = Norm; return n > 0.0 ? HoloNorm / n : 0.0; } }
        /// <summary>一阶旋转破缺强度 (|B| 在总模长中的占比)</summary>
        public double RotationBreaking { get { double n = Norm; return n > 0.0 ? B.Magnitude / n : 0.0; } }
        static double Sq(Complex z) { return z.Real * z.Real + z.Imaginary * z.Imaginary; }
    }

    public static string Fmt(Complex z)
    {
        return string.Format("{0:F5}{1}{2:F5}i", z.Real, z.Imaginary < 0 ? "-" : "+",
                             Math.Abs(z.Imaginary));
    }

    /// <summary>
    /// 四通道整体最小二乘:  W_i = A z_i + B zbar_i + C z_i^2 + D zbar_i^2
    /// 基 [z, zbar, z^2, zbar^2], 与 sjy.cs 的 FitPolynomial 完全一致。
    /// 两次 MGS-QR 求解, 免许可 (不依赖 Numerics.NET 的 Matrix / SVD)。
    /// </summary>
    public static Channels FitChannels(Vector2[] uvs, Complex[] W)
    {
        return FitChannels(uvs, W, null);
    }

    /// <summary>keep[i] = false 表示该点不参与拟合 (留一法用)</summary>
    public static Channels FitChannels(Vector2[] uvs, Complex[] W, bool[] keep)
    {
        int N = uvs.Length;
        var cols = new Complex[4][];
        for (int c = 0; c < 4; c++) cols[c] = new Complex[N];
        var y = new Complex[N];
        int m = 0;
        for (int i = 0; i < N; i++)
        {
            if (keep != null && !keep[i]) continue;
            if (W == null) continue;
            Complex z = new Complex(uvs[i].x, uvs[i].y);
            Complex zb = Complex.Conjugate(z);
            cols[0][m] = z;
            cols[1][m] = zb;
            cols[2][m] = z * z;
            cols[3][m] = zb * zb;
            y[m] = W[i];
            m++;
        }

        var res = new Channels();
        res.N = m;
        if (m < 4) return res;

        for (int c = 0; c < 4; c++)
        {
            var t = new Complex[m];
            Array.Copy(cols[c], t, m);
            cols[c] = t;
        }
        var yy = new Complex[m];
        Array.Copy(y, yy, m);

        var sol = Lstsq4(cols, yy);
        res.A = sol[0]; res.B = sol[1]; res.C = sol[2]; res.D = sol[3];

        double s = 0;
        for (int i = 0; i < m; i++)
        {
            Complex z = cols[0][i], zb = cols[1][i];
            Complex fit = res.A * z + res.B * zb + res.C * (z * z) + res.D * (zb * zb);
            Complex d = yy[i] - fit;
            s += d.Real * d.Real + d.Imaginary * d.Imaginary;
        }
        res.ResidualRms = Math.Sqrt(s / m);
        return res;
    }

    /// <summary>4 列复最小二乘 (two-pass MGS-QR, 比正规方程抗病态)</summary>
    public static Complex[] Lstsq4(Complex[][] cols, Complex[] y)
    {
        int n = 4, N = y.Length;
        var Q = new Complex[n][];
        var R = new Complex[n, n];
        for (int j = 0; j < n; j++)
        {
            var v = new Complex[N];
            for (int i = 0; i < N; i++) v[i] = cols[j][i];
            for (int pass = 0; pass < 2; pass++)
                for (int k = 0; k < j; k++)
                {
                    Complex r = Complex.Zero;
                    for (int i = 0; i < N; i++) r += Complex.Conjugate(Q[k][i]) * v[i];
                    for (int i = 0; i < N; i++) v[i] -= r * Q[k][i];
                }
            double nrm = 0;
            for (int i = 0; i < N; i++) nrm += v[i].Magnitude * v[i].Magnitude;
            nrm = Math.Sqrt(nrm);
            if (nrm < 1e-300) nrm = 1e-300;
            var q = new Complex[N];
            for (int i = 0; i < N; i++) q[i] = v[i] / nrm;
            Q[j] = q;
            for (int k = 0; k < j; k++)
            {
                Complex r = Complex.Zero;
                for (int i = 0; i < N; i++) r += Complex.Conjugate(Q[k][i]) * cols[j][i];
                R[k, j] = r;
            }
            R[j, j] = nrm;
        }
        var qhy = new Complex[n];
        for (int k = 0; k < n; k++)
        {
            Complex acc = Complex.Zero;
            for (int i = 0; i < N; i++) acc += Complex.Conjugate(Q[k][i]) * y[i];
            qhy[k] = acc;
        }
        var x = new Complex[n];
        for (int r = n - 1; r >= 0; r--)
        {
            Complex t = qhy[r];
            for (int c = r + 1; c < n; c++) t -= R[r, c] * x[c];
            x[r] = R[r, r].Magnitude < 1e-300 ? Complex.Zero : t / R[r, r];
        }
        return x;
    }

    // ---------------- 判别式 ----------------

    /// <summary>Delta = (B^2 - 4AC) / (D^2 - 4BC)</summary>
    public static Complex Delta(Channels ch)
    {
        return Delta(ch.A, ch.B, ch.C, ch.D);
    }

    public static Complex Delta(Complex A, Complex B, Complex C, Complex D)
    {
        Complex num = B * B - 4.0 * A * C;
        Complex den = D * D - 4.0 * B * C;
        if (den.Magnitude < 1e-300) return new Complex(double.NaN, double.NaN);
        return num / den;
    }

    /// <summary>
    /// Delta 是否(数值上)落在 Q 内 —— 即支撑集是否紧致。
    ///
    /// 判据是【连分数是否精确终止】, 而不是"分母界内是否存在很好的逼近":
    /// 后者对 sqrt(2) 之类会误判 (qMax = 1e5 时 114243/80782 的相对误差仅 2e-10)。
    /// 条件: 展开到某一项时小数部分 <= tol*(相对), 且分母未超过 qMax; 否则判为无理。
    /// 复数带虚部 => 必然非有理。
    /// </summary>
    public static bool DeltaIsRational(Complex delta, long qMax, double tol,
                                       out long pBest, out long qBest)
    {
        pBest = 0; qBest = 1;
        if (double.IsNaN(delta.Real) || double.IsInfinity(delta.Real)) return false;
        double scale = Math.Max(1.0, delta.Magnitude);
        if (Math.Abs(delta.Imaginary) > tol * scale) return false;   // 有虚部 => 无理

        double r = delta.Real;
        long p0 = 0, q0 = 1, p1 = 1, q1 = 0;    // p(-2)=0,q(-2)=1 ; p(-1)=1,q(-1)=0
        for (int it = 0; it < 64; it++)
        {
            double a = Math.Floor(r);
            if (Math.Abs(a) > 1e15) return false;
            long ai = (long)a;
            long p2, q2;
            try { checked { p2 = ai * p1 + p0; q2 = ai * q1 + q0; } }
            catch (OverflowException) { return false; }
            if (q2 == 0) return false;

            double frac = r - a;
            if (Math.Abs(frac) <= tol * Math.Max(1.0, Math.Abs(r)))   // 连分数到此精确终止
            {
                if (Math.Abs(q2) > qMax) return false;
                pBest = p2; qBest = q2;
                return true;
            }
            if (Math.Abs(q2) > qMax) return false;                    // 分母超界 => 无理

            p0 = p1; q0 = q1; p1 = p2; q1 = q2;
            r = 1.0 / frac;
        }
        return false;                                                 // 64 项未终止 => 无理
    }

    /// <summary>乘子比  ratio = (1 + sqrt(Delta)) / (1 - sqrt(Delta)), 归一到 |ratio| <= 1</summary>
    public static Complex RatioFromDelta(Complex delta)
    {
        Complex s = Complex.Sqrt(delta);
        Complex den = 1.0 - s;
        if (den.Magnitude < 1e-300) return new Complex(double.NaN, double.NaN);
        Complex r = (1.0 + s) / den;
        if (r.Magnitude > 1.0) r = 1.0 / r;
        return r;
    }

    /// <summary>tau = log(ratio) / (2*pi*i), 与 sjy.cs 的约定一致 (Im tau >= 0 自动成立)</summary>
    public static Complex TauFromDelta(Complex delta)
    {
        Complex r = RatioFromDelta(delta);
        if (double.IsNaN(r.Real)) return r;
        return Complex.Log(r) / new Complex(0.0, 2.0 * Math.PI);
    }

    /// <summary>按通道占比与 Delta 的有理性分类支撑集形状</summary>
    public static string Classify(Channels ch, double tol, long qMax)
    {
        double n = ch.Norm;
        if (n <= 0.0) return "全零场";
        bool B0 = ch.B.Magnitude <= tol * n;
        bool C0 = ch.C.Magnitude <= tol * n;
        bool D0 = ch.D.Magnitude <= tol * n;
        if (B0 && C0 && D0) return "圆 (纯 A, 线性全纯)";
        if (!B0 && C0 && D0) return "椭圆 (仅 B 的一阶非全纯破缺)";
        Complex d = Delta(ch);
        long p, q;
        if (DeltaIsRational(d, qMax, 1e-10, out p, out q))
            return string.Format("卵形线 / 对数螺旋带 (Delta = {0}/{1} 有理, 支撑集紧致)", p, q);
        return "开口螺旋 / 无限叶 (Delta 非有理, 支撑集非紧致)";
    }

    // ---------------- 分源分解 ----------------

    /// <summary>
    /// 三类贡献各自拟合一次, 回答"每个通道的主要来源是谁"。
    /// 返回 [0]=刚度 [1]=AB 色散 [2]=四元数旋转 [3]=合计。
    /// </summary>
    public static Channels[] FitChannelsBySource(Vector2[] uvs, Complex[] stiff,
                                                 Complex[] AB, Complex[] quat)
    {
        var res = new Channels[4];
        res[0] = FitChannels(uvs, stiff);
        res[1] = FitChannels(uvs, AB);
        res[2] = FitChannels(uvs, quat);
        int N = uvs.Length;
        var tot = new Complex[N];
        for (int i = 0; i < N; i++)
            tot[i] = (stiff == null ? Complex.Zero : stiff[i])
                   + (AB == null ? Complex.Zero : AB[i])
                   + (quat == null ? Complex.Zero : quat[i]);
        res[3] = FitChannels(uvs, tot);
        return res;
    }

    static string Dominant(Channels[] bySource, int chan)
    {
        string[] nm = { "刚度", "AB色散", "四元数" };
        var v = new double[3];
        for (int k = 0; k < 3; k++)
        {
            Complex c = (chan == 0) ? bySource[k].A
                      : (chan == 1) ? bySource[k].B
                      : (chan == 2) ? bySource[k].C : bySource[k].D;
            v[k] = c.Magnitude;
        }
        int b = 0;
        for (int k = 1; k < 3; k++) if (v[k] > v[b]) b = k;
        double sw = v[0] + v[1] + v[2];
        return string.Format("{0} {1:P0}", nm[b], sw > 0.0 ? v[b] / sw : 0.0);
    }

    // ---------------- 整体性 ----------------

    /// <summary>
    /// 逐点去掉后重拟合, 看四通道各自漂移多少。
    /// 这就是"A,B,C,D 不是四个旋钮, 而是整体拟合出来的通道"的定量说法。
    /// 漂移用同一标度 (full.Norm) 归一, 否则通道本身接近 0 时相对量会爆掉。
    /// </summary>
    public static string LeaveOneOutSpread(Vector2[] uvs, Complex[] W)
    {
        int N = uvs.Length;
        if (N < 6) return "点数不足";
        var full = FitChannels(uvs, W);
        var f = new Complex[] { full.A, full.B, full.C, full.D };
        double commonScale = Math.Max(full.Norm, 1e-300);
        var mx = new double[4];
        var rms = new double[4];
        var keep = new bool[N];
        for (int i = 0; i < N; i++)
        {
            for (int k = 0; k < N; k++) keep[k] = (k != i);
            var c = FitChannels(uvs, W, keep);
            var arr = new Complex[] { c.A, c.B, c.C, c.D };
            for (int ch = 0; ch < 4; ch++)
            {
                double rel = (arr[ch] - f[ch]).Magnitude / commonScale;
                if (rel > mx[ch]) mx[ch] = rel;
                rms[ch] += rel * rel;
            }
        }
        var sb = new System.Text.StringBuilder();
        string[] nm = { "A", "B", "C", "D" };
        sb.Append(string.Format("去掉任一点后四通道漂移 / 全场模长 (scale = {0:E3}), max / rms:  ",
                                commonScale));
        for (int ch = 0; ch < 4; ch++)
            sb.Append(string.Format("{0}: {1:E2} / {2:E2}   ", nm[ch], mx[ch],
                                    Math.Sqrt(rms[ch] / N)));
        return sb.ToString();
    }

    /// <summary>一次性打印四通道的全部结论</summary>
    public static string Describe(Channels ch, Channels[] bySource,
                                  double tol = 1e-2, long qMax = 100000)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("ABCD 四通道   dU/dz = A z + B zbar + C z^2 + D zbar^2 + 高阶项");
        sb.AppendLine("  A = " + Fmt(ch.A) + "   线性全纯    各向同性背景 / 整体尺度与径向增长");
        sb.AppendLine("  B = " + Fmt(ch.B) + "   线性非全纯  各向异性偶极 / 一阶旋转破缺");
        sb.AppendLine("  C = " + Fmt(ch.C) + "   二次全纯    高阶色散形变");
        sb.AppendLine("  D = " + Fmt(ch.D) + "   二次非全纯  手性扭曲 / 非对易");
        sb.AppendLine(string.Format("  |全纯| = {0:E4}   |非全纯| = {1:E4}   非全纯占比 = {2:P2}",
                                    ch.HoloNorm, ch.AntiNorm, 1.0 - ch.HoloFraction));
        sb.AppendLine(string.Format("  Delta = (B^2-4AC)/(D^2-4BC) = {0}", Fmt(Delta(ch))));
        sb.AppendLine("  形状: " + Classify(ch, tol, qMax));
        if (bySource != null && bySource.Length >= 3)
        {
            sb.AppendLine("  主源:  A <- " + Dominant(bySource, 0)
                        + "    B <- " + Dominant(bySource, 1)
                        + "    C <- " + Dominant(bySource, 2)
                        + "    D <- " + Dominant(bySource, 3));
        }
        sb.AppendLine(string.Format("  拟合点数 {0}   残差 RMS = {1:E3}", ch.N, ch.ResidualRms));
        return sb.ToString();
    }

    // ==================================================== 自检

    /// <summary>
    /// 通道自检: 用解析可预期的场验证拟合器 + "常数部分"定理。
    ///   1) W = alpha z + beta zbar (常数 Hessian 的刚度投影) -> C = D = 0
    ///   2) W = alpha z                                        -> B = C = D = 0
    ///   3) Delta 的有理性判据 (有理 / 无理 / 带虚部)
    ///   4) 留一法漂移
    /// </summary>
    public static string SelfTestChannels(int N = 64)
    {
        var uvs = new Vector2[N];
        for (int i = 0; i < N; i++)
        {
            double t = 2.0 * Math.PI * i / N;
            double rad = 0.2 + 1.3 * ((i * 37 % N) / (double)N);   // 半径必须变化, 否则不可辨识
            uvs[i] = new Vector2((float)(rad * Math.Cos(t)), (float)(rad * Math.Sin(t)));
        }
        var sb = new System.Text.StringBuilder();
        Complex alpha = new Complex(0.7, -0.3), beta = new Complex(0.4, 0.25);

        var w1 = new Complex[N];
        for (int i = 0; i < N; i++)
        {
            Complex z = new Complex(uvs[i].x, uvs[i].y);
            w1[i] = alpha * z + beta * Complex.Conjugate(z);
        }
        var c1 = FitChannels(uvs, w1);
        sb.AppendLine(string.Format(
            "常数部分定理  W = alpha z + beta zbar -> |A-alpha| = {0:E2}  |B-beta| = {1:E2}  |C| = {2:E2}  |D| = {3:E2}",
            (c1.A - alpha).Magnitude, (c1.B - beta).Magnitude, c1.C.Magnitude, c1.D.Magnitude));

        var w2 = new Complex[N];
        for (int i = 0; i < N; i++)
        {
            Complex z = new Complex(uvs[i].x, uvs[i].y);
            w2[i] = alpha * z;
        }
        var c2 = FitChannels(uvs, w2);
        sb.AppendLine(string.Format("纯 A 场  |B| = {0:E2}  |C| = {1:E2}  |D| = {2:E2}  ->  {3}",
            c2.B.Magnitude, c2.C.Magnitude, c2.D.Magnitude, Classify(c2, 1e-2, 100000)));

        long p, q;
        sb.AppendLine(string.Format("Delta 有理性  2/3       -> {0}  ({1}/{2})",
            DeltaIsRational(new Complex(2.0 / 3.0, 0.0), 100000, 1e-10, out p, out q), p, q));
        sb.AppendLine(string.Format("Delta 有理性  sqrt(2)   -> {0}  (应为 False)",
            DeltaIsRational(new Complex(Math.Sqrt(2.0), 0.0), 100000, 1e-10, out p, out q)));
        sb.AppendLine(string.Format("Delta 有理性  0.5+0.25i -> {0}  (带虚部必然无理)",
            DeltaIsRational(new Complex(0.5, 0.25), 100000, 1e-10, out p, out q)));
        sb.AppendLine(LeaveOneOutSpread(uvs, w1));
        return sb.ToString();
    }

    /// <summary>把简化版和原版逐点比一遍, 返回最大绝对差。</summary>
    public static string SelfTest(List<Vector3> pointsA, List<Vector3> pointsB)
    {
        float[] Iold, Vold;
        LocalLensVolumeExtractor.ComputeAllFeatures(pointsA, pointsB, out Iold, out Vold);
        float[] Inew, Vnew;
        ComputeAllFeatures(pointsA, pointsB, out Inew, out Vnew);

        double dI = 0, dV = 0;
        for (int i = 0; i < Mathf.Min(Iold.Length, Inew.Length); i++)
        {
            dI = Math.Max(dI, Math.Abs((double)Iold[i] - Inew[i]));
            dV = Math.Max(dV, Math.Abs((double)Vold[i] - Vnew[i]));
        }
        var L = Compute(pointsA, pointsB);
        double gmin = double.MaxValue, gmax = double.MinValue;
        foreach (var x in L) { gmin = Math.Min(gmin, x.g); gmax = Math.Max(gmax, x.g); }
        return string.Format(
            "LensVolumeSimple.SelfTest  N={0}  |dI|max={1:E3}  |dV|max={2:E3}  " +
            "g in [{3:F4}, {4:F4}]  Phi(g) in [{5:E3}, {6:E3}]",
            L.Length, dI, dV, gmin, gmax, Phi(gmin), Phi(gmax));
    }
}
class ABCD
{
    public static double Phi(double g)
    {
        if (g <= 0.0) return 1.0;
        if (g >= 2.0) return 0.0;

        double u = g * 0.5;
        double alpha = Math.Acos(u);

        // alpha 很小时泰勒展开，避免 12a - 8sin2a 相消
        if (alpha < 0.05)
            return alpha * alpha * alpha * alpha * alpha
                   * (6.4 - (15360.0 / 5040.0) * alpha * alpha
                        + (258048.0 / 362880.0) * alpha * alpha * alpha * alpha)
                   / (6.0 * Math.PI);

        return (12.0 * alpha - 8.0 * Math.Sin(2.0 * alpha) + Math.Sin(4.0 * alpha))
               / (6.0 * Math.PI);
    }
    public static double ShapeX(double g)
    {
        double x = 1.0 - 0.25 * g * g;
        return x < 0.0 ? 0.0 : x;   // 等价于 Clamp01
    }


}
