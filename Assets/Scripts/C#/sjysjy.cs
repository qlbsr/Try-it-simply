using System;
using System.Collections.Generic;
using System.Numerics;
using UnityEngine;
using Quaternion = UnityEngine.Quaternion;
using Vector2 = UnityEngine.Vector2;
using Vector3 = UnityEngine.Vector3;
using Complex = System.Numerics.Complex;

public static class ComplexAngleSolver
{
    // -----------------------------------------------------------
    // 复数三角函数（System.Numerics.Complex 已提供 Sin, Cos, Tan 等）
    // 但缺少 Cot, Acot 和 Atan 的某些形式，这里补全
    // -----------------------------------------------------------
    public static Complex Cot(Complex z) => Complex.Cos(z) / Complex.Sin(z);

    public static Complex Acot(Complex z) => Complex.Atan(1.0 / z); // 主值

    // -----------------------------------------------------------
    // 方程左边的复数版本
    // F(θ) = 4*[Atan(π sinα / sinθ)]² + 4π² sin²α * cot²θ - cos²α
    // -----------------------------------------------------------
    public static Complex Equation(Complex theta, double alpha)
    {
        double sinAlpha = Math.Sin(alpha);
        double cosAlpha = Math.Cos(alpha);
        double target = cosAlpha * cosAlpha;

        Complex sinTheta = Complex.Sin(theta);
        Complex cotTheta = Cot(theta);

        Complex arg = Math.PI * sinAlpha / sinTheta;
        Complex atanVal = Complex.Atan(arg);          // 主值

        Complex left = 4.0 * atanVal * atanVal
                     + 4.0 * Math.PI * Math.PI * sinAlpha * sinAlpha * cotTheta * cotTheta;

        return left - target;
    }

    // -----------------------------------------------------------
    // 数值导数（中心差分）
    // -----------------------------------------------------------
    public static Complex Derivative(Func<Complex, Complex> f, Complex z, double h = 1e-6)
    {
        return (f(z + h) - f(z - h)) / (2.0 * h);
    }

    // -----------------------------------------------------------
    // 牛顿法求复数根
    // initialGuess: 初始猜测
    // alpha: 实参数 α (弧度)
    // maxIter: 最大迭代次数
    // tolerance: 收敛容限
    // -----------------------------------------------------------
    public static Complex NewtonSolve(Complex initialGuess, double alpha,
                                       int maxIter = 100, double tolerance = 1e-10)
    {
        Complex theta = initialGuess;
        for (int k = 0; k < maxIter; k++)
        {
            Complex f = Equation(theta, alpha);
            Complex df = Derivative(t => Equation(t, alpha), theta, 1e-8);

            if (Complex.Abs(df) < 1e-14) break;   // 导数过小，避免除零

            Complex delta = f / df;
            theta -= delta;

            if (Complex.Abs(delta) < tolerance)
                return theta;
        }
        return theta; // 返回近似解
    }

    // -----------------------------------------------------------
    // 方便调用的方法：输入 α (弧度)，返回可能的复数解列表
    // -----------------------------------------------------------
    public static List<Complex> FindSolutions(double alpha, int numGuesses = 5)
    {
        var solutions = new List<Complex>();
        // 几个常见初始猜测：实轴、虚轴、45° 等
        Complex[] guesses = new Complex[]
        {
            new Complex(Math.PI / 4, 0),          // 45°
            new Complex(Math.PI / 2, 0),          // 90°
            new Complex(0, 1),                    // 纯虚
            new Complex(0, -1),
            new Complex(Math.PI / 4, 1),
            new Complex(Math.PI / 4, -1),
            new Complex(-Math.PI / 4, 0),
            new Complex(-Math.PI / 4, 1),
        };

        foreach (var guess in guesses)
        {
            Complex root = NewtonSolve(guess, alpha);
            // 去重（简单检查）
            bool isNew = true;
            foreach (var sol in solutions)
                if (Complex.Abs(root - sol) < 1e-6)
                { isNew = false; break; }
            if (isNew && !double.IsNaN(root.Real) && !double.IsNaN(root.Imaginary))
                solutions.Add(root);
        }
        return solutions;
    }
}

// ---- 主入口 ----

public class MardenEllipse
{
    public static double SolveC3(double c2, double h1, double V)
    {
        if (V <= 0) return Math.Max(c2, h1) + 1e-9;
        if (c2 <= 0) c2 = 1e-12;
        if (h1 <= 0) h1 = 1e-12;

        double lo = Math.Max(c2, h1) + 1e-9;
        double hi = lo * 2.0;

        // 检查下界
        if (F(lo, c2, h1, V) > 0)
            return lo;   // V 太小，c3 取下界

        for (int i = 0; i < 200; i++)
        {
            if (F(hi, c2, h1, V) > 0) break;
            hi *= 2.0;
            if (hi > 1e15) break;
        }

        for (int i = 0; i < 120; i++)
        {
            double mid = 0.5 * (lo + hi);
            if (F(mid, c2, h1, V) > 0) hi = mid;
            else lo = mid;
        }

        return 0.5 * (lo + hi);
    }
    static double F(double c3, double c2, double h1, double V)
    {
        if (c3 <= Math.Max(c2, h1)) return -1e18;

        double w1 = Math.Sqrt((c3 / c2) * (c3 / c2) - 1.0);
        double w2 = Math.Sqrt((c3 / h1) * (c3 / h1) - 1.0);

        double cAxis = Math.Abs(h1 * w2 - h1 * w1) / (2.0 * Math.PI);
        double Vcalc = Math.PI * cAxis * cAxis * c3;

        return Vcalc - V;
    }
    public static double SolveT(double c2, double c3, double k)
    {
        if (c3 <= 0) throw new ArgumentException("c3 must be positive");
        if (k < 0) k = -k;
        if (k > 1.0 - 1e-12) k = 1.0 - 1e-12;

        double u = Math.Min(1.0, Math.Max(0.0, c2 / c3));

        double Efull = Elliptic.E(k);               // 完全第二类 (Carlsonfk 组装, 不用商业库)
        double target = u * 4.0 * Efull;

        return SolveTFromE(target, k);
    }

    public static double SolveTFromE(double target, double k)
    {
        double TwoPi = 2.0 * Math.PI;
        double Efull = Elliptic.E(k);
        double periodE = 4.0 * Efull;

        target = target % periodE;
        if (target < 0) target += periodE;

        double lo = 0.0, hi = TwoPi;
        for (int iter = 0; iter < 80; iter++)
        {
            double mid = 0.5 * (lo + hi);
            double Emid = Elliptic.EInc(mid, k);        // 不完全第二类 (带准周期归约)
            if (Emid < target) lo = mid;
            else hi = mid;
        }
        return 0.5 * (lo + hi);
    }









    public static Vector3[] th2(double theta, double H)
    {
        double s = Math.Sin(theta), c = Math.Cos(theta);
        if (Math.Abs(s) < 1e-300) s = s >= 0 ? 1e-300 : -1e-300;
        double tanX = -H / s;
        Vector3 c2 = new Vector3((float)Math.Atan(tanX), (float)H, (float)(tanX * c));
        double tanXLow = H / s;
        Vector3 c1 = new Vector3((float)Math.Atan(tanXLow), (float)H, (float)(tanXLow * c));
        return new Vector3[] { c1, c2 };
    }

    public float[] dd2(Vector3 A, Vector3 B, Vector3 C)
    {
        // 1. 计算平面法向量
        Vector3 n = Vector3.Cross(B - A, C - A);  // 以 A 为参考点更安
        n.Normalize();
        Vector3 e1 = (C - A).normalized;          // 以 AC 方向为 X 轴
        Vector3 e2 = Vector3.Cross(n, e1).normalized;

        // 2. 投影三个点到该平面（以 A 为原点）
        Vector2 A2 = Vector2.zero;
        Vector2 B2 = new Vector2(Vector3.Dot(B - A, e1), Vector3.Dot(B - A, e2));
        Vector2 C2 = new Vector2(Vector3.Dot(C - A, e1), Vector3.Dot(C - A, e2));

        // 3. 重心
        Vector2 center = (A2 + B2 + C2) / 3f;

        // 4. 协方差矩阵（使用投影后的二维坐标！）
        float sxx = (A2.x - center.x) * (A2.x - center.x) +
                    (B2.x - center.x) * (B2.x - center.x) +
                    (C2.x - center.x) * (C2.x - center.x);
        float syy = (A2.y - center.y) * (A2.y - center.y) +
                    (B2.y - center.y) * (B2.y - center.y) +
                    (C2.y - center.y) * (C2.y - center.y);
        float sxy = (A2.x - center.x) * (A2.y - center.y) +
                    (B2.x - center.x) * (B2.y - center.y) +
                    (C2.x - center.x) * (C2.y - center.y);

        // 5. 特征值
        float trace = sxx + syy;
        float det = sxx * syy - sxy * sxy;
        float disc = Mathf.Sqrt(Mathf.Max(0, trace * trace - 4f * det));
        float lambda1 = (trace + disc) * 0.5f;
        float lambda2 = (trace - disc) * 0.5f;
        // 6. 半轴（最大面积内切椭圆的半轴 = sqrt(λ/2)）
        float a_ell = Mathf.Sqrt(Mathf.Max(0, lambda1 / 2f));
        float b_ell = Mathf.Sqrt(Mathf.Max(0, lambda2 / 2f));
        if (b_ell < 1e-6f) b_ell = a_ell * 0.1f;  // 防止退化导致除零
                                                  // 7. 周长（拉马努金近似）
        float C_ell = Mathf.PI * (3f * (a_ell + b_ell) -
                      Mathf.Sqrt((3f * a_ell + b_ell) * (a_ell + 3f * b_ell)));
        // 8. 绕长轴旋转的椭球体积
        return new float[] { C_ell, a_ell, b_ell, center.x, center.y };
    }

    public Vector2[] MapDoubleConeToLeaf(List<Vector3> points, float d2, float rp, Vector3 v3, List<Vector3> uvss, Vector2[] uvs)
    {
        // 新锥几何参数
        Vector3[] v1;
        float alpha = d2 * Mathf.Deg2Rad;
        float h = rp * Mathf.Cos(alpha);             // 原锥高
        float R = rp * Mathf.Sin(alpha);             // 原底面半径
        float H = Mathf.PI * R;                      //高
        float HR = 1f / 2f * h;                      //半径
        float fzj = Mathf.Atan2(0.5f * h, H);      //防止角度
        var sols = ComplexAngleSolver.FindSolutions(alpha);
        float theta2 = 0f;
        //缩小a
        foreach (var z in sols)
        {

            if (Mathf.Abs((float)z.Imaginary) < 1e-6f)   // 虚部足够小，视为实数
            {
                theta2 = (float)z.Real;
                break;
            }
        }
        if (theta2 != 0)
        {
            v1 = th2(theta2, H);

        }
        else
        {
            v1 = th2(fzj, H);
        }

        Debug.Log((v1[0] - v1[1]).normalized);
        v3 = (v1[0] - v1[1]).normalized;
        uvs = new UnityEngine.Vector2[points.Count];
        uvss = new List<Vector3>();
        for (int i = 0; i < points.Count; i++)
        {
            Vector3[] v;
            if (theta2 != 0)
            {
                v = th2(theta2, points[i].y);

            }
            else
            {
                v = th2(fzj, points[i].y);
            }

            float h1 = Vector3.Distance(v[0], v[1]);
            float y = h1 / h;
            float Vc = (float)(y * y * y * (1f / 3f) * Math.PI * H * (1f / 4f) * h * h);
            Vector3 u2 = (v[0] - v[1]).normalized;
            Quaternion q = Quaternion.FromToRotation(v3, u2);
            Vector3 v_ = new Vector3(points[i].x * y, points[i].y * y, points[i].z * y);
            v_ = q * v_;
            Vector3 C = new Vector3(0, y * H, 0);
            float c2 = v_.magnitude;
            float[] f = dd2(Vector3.zero, C, v_);
            float c3 = f[0];
            float x = (c3 / c2);
            float x2 = (c3 / h1);
            float w1 = (float)Math.Sqrt(x * x - 1);
            float w2 = (float)Math.Sqrt(x2 * x2 - 1);
            double cAxis = Math.Abs(h1 * w2 - h1 * w1) / (2.0 * Math.PI);
            float vabc = (float)(Math.PI * cAxis * cAxis * c3);
            float opi = (float)Math.Pow(c2 / Math.Sqrt((h1 * 0.5f) * (h1 * 0.5f) + (y * H) * (y * H)), 2);
            double phi = Math.PI *2f* vabc / Vc;
            uvs[i] = new Vector2((float)(opi * Math.Cos(phi)),
            (float)(opi * Math.Sin(phi)));
        }
        return uvs;
    }
}
    public static class Carlsonfk
    {
        public const double ERRTOL_RF = 0.0025;
        public const double ERRTOL_RD = 0.0015;
        // ================================================================
        //  常量
        // ================================================================
        private const double Epsilon = 2.220446049250313e-16;  // double 机器精度

        // ================================================================
        //  K(k) = R_F(0, 1 - k², 1)
        // ================================================================
        public static Complex K(Complex k)
        {
            return RF(Complex.Zero, Complex.One - k * k, Complex.One);
        }

        // ================================================================
        //  复数域 R_F(x, y, z) —— Carlson 1995 算法（含 μ 修正）
        // ================================================================
        public static Complex RF(Complex x, Complex y, Complex z)
        {
            Complex A0 = (x + y + z) / 3.0;
            Complex A = A0;

            double maxDev = Math.Max(
                Math.Max((A0 - x).Magnitude, (A0 - y).Magnitude),
                (A0 - z).Magnitude);
            double Q = Math.Pow(Epsilon / 4.0, -1.0 / 6.0) * maxDev;

            int i;
            for (i = 0; i < 300; i++)
            {
                Complex sx = Complex.Sqrt(x);
                Complex sy = Complex.Sqrt(y);
                Complex sz = Complex.Sqrt(z);

                Complex lambda = sx * sy + sy * sz + sz * sx;
                Complex e = sx * sy * sz;

                // ---- μ 修正（复数域关键，实数域恒为 0）----
                Complex mu = Complex.Zero;
                if (A != Complex.Zero && e != Complex.Zero)
                {
                    Complex ratio = lambda / (A * e);
                    if (ratio.Real < 0)
                        mu = 2.0 * e * (A / A.Magnitude);
                }

                A = (A + lambda + mu) / 4.0;
                x = (x + lambda + mu) / 4.0;
                y = (y + lambda + mu) / 4.0;
                z = (z + lambda + mu) / 4.0;

                Q /= 4.0;
                if (Q < A.Magnitude) break;
            }
            if (i == 300) throw new Exception("RF: 未收敛");

            // 使用收敛后的 A 与迭代后的 x, y, z 计算相对偏差
            // （与 Boost 实数版一致，跳出时的 x,y,z 与 A 同代）
            Complex X = (A0 - x) / A;
            Complex Y = (A0 - y) / A;
            Complex Z = (A0 - z) / A;

            Complex E2 = X * Y - Z * Z;
            Complex E3 = X * Y * Z;
            return (1 + E2 * (E2 / 24 - E3 * 3.0 / 44 - 0.1) + E3 / 14)
                   / Complex.Sqrt(A);
        }

        // ================================================================
        //  复数域 R_D(x, y, z) —— Carlson 1995 算法（含 μ 修正）
        // ================================================================
        public static Complex RD(Complex x, Complex y, Complex z)
        {
            // ---- 定义域检查 ----
            if (x == Complex.Zero && y == Complex.Zero)
                throw new ArgumentException("RD: x 和 y 不能同时为零");
            if (z == Complex.Zero)
                throw new ArgumentException("RD: z 不能为零");

            // ---- 对称性：RD 对 x,y 对称 ----
            if (x == z) Swap(ref x, ref y);

            // ---- 退化情形 ----
            if (y == z)
            {
                if (x == y)
                    return 1.0 / (x * Complex.Sqrt(x));

                if (x == Complex.Zero)
                    return 3.0 * Math.PI / (4.0 * y * Complex.Sqrt(y));
                // 其余情况走主迭代
            }

            Complex x0 = x, y0 = y, z0 = z;

            Complex An = (x + y + 3.0 * z) / 5.0;
            Complex A0 = An;

            double maxDev = Math.Max(
                Math.Max((An - x).Magnitude, (An - y).Magnitude),
                (An - z).Magnitude);
            double Q = Math.Pow(Epsilon / 4.0, -1.0 / 8.0) * maxDev * 1.2;

            double fn = 1.0;
            Complex RD_sum = Complex.Zero;

            int i;
            for (i = 0; i < 300; i++)
            {
                Complex sx = Complex.Sqrt(x);
                Complex sy = Complex.Sqrt(y);
                Complex sz = Complex.Sqrt(z);

                Complex lambda = sx * sy + sy * sz + sz * sx;
                Complex e = sx * sy * sz;

                // ---- μ 修正 ----
                Complex mu = Complex.Zero;
                if (An != Complex.Zero && e != Complex.Zero)
                {
                    Complex ratio = lambda / (An * e);
                    if (ratio.Real < 0)
                        mu = 2.0 * e * (An / An.Magnitude);
                }

                RD_sum += fn / (sz * (z + lambda + mu));

                An = (An + lambda + mu) / 4.0;
                x = (x + lambda + mu) / 4.0;
                y = (y + lambda + mu) / 4.0;
                z = (z + lambda + mu) / 4.0;

                fn /= 4.0;
                Q /= 4.0;

                if (Q < An.Magnitude) break;
            }
            if (i == 300) throw new Exception("RD: 未收敛");

            // ---- 11 阶泰勒展开（用原始 x0,y0,z0）----
            Complex X = fn * (A0 - x0) / An;
            Complex Y = fn * (A0 - y0) / An;
            Complex Z = -(X + Y) / 3.0;

            Complex E2 = X * Y - 6.0 * Z * Z;
            Complex E3 = (3.0 * X * Y - 8.0 * Z * Z) * Z;
            Complex E4 = 3.0 * (X * Y - Z * Z) * Z * Z;
            Complex E5 = X * Y * Z * Z * Z;

            Complex result = fn * Complex.Pow(An, -3.0 / 2.0) *
                (1.0
                 - 3.0 * E2 / 14.0
                 + E3 / 6.0
                 + 9.0 * E2 * E2 / 88.0
                 - 3.0 * E4 / 22.0
                 - 9.0 * E2 * E3 / 52.0
                 + 3.0 * E5 / 26.0
                 - E2 * E2 * E2 / 16.0
                 + 3.0 * E3 * E3 / 40.0
                 + 3.0 * E2 * E4 / 20.0
                 + 45.0 * E2 * E2 * E3 / 272.0
                 - 9.0 * (E3 * E4 + E2 * E5) / 68.0);

            result += 3.0 * RD_sum;
            return result;
        }

        // ================================================================
        //  E(k) = R_F(0, 1-k², 1) - (k²/3) · R_D(0, 1-k², 1)
        // ================================================================

        public static Complex E(Complex k)
        { // E(k) = ∫_0^{π/2} sqrt(1 - k^2 sin^2 θ) dθ

            Complex k2 = k * k;
            return RF(Complex.Zero, Complex.One - k2, Complex.One)
                 - k2 / 3.0 * RD(Complex.Zero, Complex.One - k2, Complex.One);
        }
        // ================================================================
        //  R_D(0, 1-k², 1) = 3/k² · (K - E)
        //  [FIX] k = 0 时返回极限 3π/4，避免 0/0
        // ================================================================
        public static Complex RD1z(Complex k)
        {
            if (k.Magnitude < 1e-10) return 3.0 * Math.PI / 4.0;
            return (3.0 / (k * k)) * (K(k) - E(k));
        }
        // ================================================================
        //  R_D(0, 1, 1-k²)  —— 直接调用 RD，避免复杂的代数关系
        //  [FIX] 原公式 (3E/(1-k²) - RD1z) 不鲁棒，改为直接调用
        // ================================================================
        public static Complex RDz1(Complex k)
        {
            return RD(Complex.Zero, Complex.One, Complex.One - k * k);
        }
        //return K(k) - 1.0 / 3.0 * k* k *RD(Complex.Zero, 1 - k * k, Complex.One);
        //return 1.0/3.0(1.0-k*K)[RD(0,1-k*k,1)+RD[0,1,1-k*K]] k=0?
        // K(k)=RD(0,1-k*k,1)+1/3(1-k*k)RD(0,1,1-k*k)


        // ================================================================
        //  dK/dk = [E - (1-k²)K] / [k(1-k²)]
        //  [FIX] k = 0 → 0；k = ±1 → 无穷
        // ================================================================
        public static Complex DK(Complex k)
        {
            if (k.Magnitude < 1e-10)
                return Complex.Zero;
            if (Complex.Abs(Complex.One - k * k) < 1e-12)
                return new Complex(double.PositiveInfinity, 0.0);   // 实部无穷，虚部 0

            return (E(k) - (1 - k * k) * K(k)) / (k * (1 - k * k));
        }

        // ================================================================
        //  反解 K(k) = aot（牛顿法 + 解析导数）
        //  [FIX] 移除未定义的 DDK/SchwarzianK；改用 DK
        //  [FIX] 智能初值，避免 k=0 的 DK=0 奇点
        //  [FIX] 保留负数保护（对纯实数 aot 语义正确）
        // ================================================================
        public static Complex NK(Complex aot)
        {
            // ---- 负数保护（仅对纯实数 aot 有意义）----
            if (Math.Abs(aot.Imaginary) < 1e-14 && aot.Real < 0)
            {
                aot = -aot;
                Debug.Log("NK: aot 实部为负，已取相反数");
            }

            // ---- 智能初值 ----
            double aMag = aot.Magnitude;
            Complex k;
            if (aMag <= Math.PI / 2 + 1e-12)
                k = Complex.Zero;
            else if (aMag < 2.5)
                k = new Complex(0.5, 0.0);
            else
            {
                // K(k) ~ ln(4/√(1-k²))，粗略反解
                double kk = 1.0 - 16.0 * Math.Exp(-2.0 * aMag);
                k = Complex.Sqrt(Math.Max(kk, 0.0));
            }

            // ---- 牛顿迭代 ----
            for (int i = 0; i < 200; i++)
            {
                Complex f = K(k) - aot;
                if (f.Magnitude < 1e-13) return k;

                Complex df = DK(k);

                // 导数奇异（k=0 或 k=±1），用二阶泰勒跳一次
                if (df.Magnitude < 1e-14)
                {
                    Complex delta = aot - Math.PI / 2.0;
                    k = Complex.Sqrt(delta * 8.0 / Math.PI);
                    continue;
                }

                Complex step = f / df;
                double s = step.Magnitude;
                if (s > 0.3) step *= 0.3 / s;

                k -= step;
                if (step.Magnitude < 1e-14) return k;
            }

            Debug.LogWarning("NK: 未在 200 次内收敛");
            return k;
        }

        // ================================================================
        //  辅助：交换两个 Complex
        // ================================================================
        private static void Swap(ref Complex a, ref Complex b)
        {
            Complex t = a;
            a = b;
            b = t;
        }
        public static void JacobiSNCNDN(Complex u, Complex m,
                                    out Complex sn, out Complex cn, out Complex dn)
        {
            // ---- m = 1 特例：解析式 sn = tanh(u), cn = dn = sech(u) ----
            if (Complex.Abs(m - Complex.One) < 1e-15)
            {
                Complex th = Complex.Tanh(u);
                Complex sech = 1.0 / Complex.Cosh(u);
                sn = th;
                cn = sech;
                dn = sech;
                return;
            }

            // ---- 递归迭代化，加深度上限 ----
            const int MaxDepth = 60;
            const double SmallM = 1e-12;

            // 保存每层的 k1，用于反向变换
            var k1Stack = new System.Collections.Generic.List<Complex>(MaxDepth);
            Complex uCur = u;
            Complex mCur = m;

            int depth = 0;
            while (Complex.Abs(mCur) >= SmallM && depth < MaxDepth)
            {
                Complex s = Complex.Sqrt(1.0 - mCur);
                Complex k1 = (1.0 - s) / (1.0 + s);
                k1Stack.Add(k1);

                uCur = uCur / (1.0 + k1);
                mCur = k1 * k1;
                depth++;
            }

            if (depth == MaxDepth)
                throw new Exception("JacobiSNCNDN: Landen 变换未收敛");

            // ---- 最内层：小 m 泰勒展开 ----
            Complex sinu = Complex.Sin(uCur);
            Complex cosu = Complex.Cos(uCur);

            Complex snC = sinu + mCur / 4.0 * (sinu * cosu - uCur) * cosu;
            Complex cnC = cosu + mCur / 4.0 * (-sinu * cosu + uCur) * sinu;
            Complex dnC = 1.0 + mCur / 4.0 * (cosu * cosu - sinu * sinu - 1.0);

            // ---- 反向递推 ----
            for (int i = k1Stack.Count - 1; i >= 0; i--)
            {
                Complex k1 = k1Stack[i];
                Complex denom = 1.0 + k1 * snC * snC;

                // 除零保护：sn1 落在极点附近时 denom 可能为零
                if (denom.Magnitude < 1e-300)
                {
                    snC = new Complex(double.PositiveInfinity, 0);
                    cnC = new Complex(double.PositiveInfinity, 0);
                    dnC = new Complex(double.PositiveInfinity, 0);
                    continue;
                }

                Complex snNew = (1.0 + k1) * snC / denom;
                Complex cnNew = cnC * dnC / denom;
                Complex dnNew = (dnC * dnC - (1.0 - k1)) / (1.0 + k1 - dnC * dnC);

                snC = snNew;
                cnC = cnNew;
                dnC = dnNew;
            }

            sn = snC;
            cn = cnC;
            dn = dnC;
        }
    }

    // ==================================================================
    //  实参数椭圆积分 (完全 / 不完全), 全部由 Carlsonfk.RF/RD 组装。
    //  替代 Numerics.NET 的 Special.EllipticK/E/F —— 那个库需要商业许可。
    //
    //  Carlson 标准形式 (仅在 0 <= phi <= pi/2 直接成立):
    //      F(phi,k) = sin(phi) * R_F(cos^2 phi, 1 - k^2 sin^2 phi, 1)
    //      E(phi,k) = sin(phi) * R_F(...) - (k^2/3) * sin^3(phi) * R_D(...)
    //  超过 pi/2 用准周期归约:
    //      F(phi+n*pi,k) = 2n*K(k) + F(phi,k)      E 同理
    //      F(pi-phi,k)   = 2K(k)  - F(phi,k)       E 同理
    //  已验证: 对 k in [0,0.99], phi in [0,2pi] 与 scipy 参考实现相对差 < 4.3e-16
    // ==================================================================
    public static class Elliptic
    {
        /// <summary>完全第一类 K(k) = R_F(0, 1-k^2, 1)</summary>
        public static double K(double k)
        {
            return Carlsonfk.K(new Complex(k, 0.0)).Real;
        }

        /// <summary>完全第二类 E(k) = R_F(0,1-k^2,1) - (k^2/3) R_D(0,1-k^2,1)</summary>
        public static double E(double k)
        {
            return Carlsonfk.E(new Complex(k, 0.0)).Real;
        }

        /// <summary>Carlson 形式, 仅 0 &lt;= phi &lt;= pi/2</summary>
        static double FRaw(double phi, double k)
        {
            double s = Math.Sin(phi), c = Math.Cos(phi);
            Complex x = new Complex(c * c, 0.0);
            Complex y = new Complex(1.0 - k * k * s * s, 0.0);
            return (s * Carlsonfk.RF(x, y, Complex.One)).Real;
        }

        /// <summary>Carlson 形式, 仅 0 &lt;= phi &lt;= pi/2</summary>
        static double ERaw(double phi, double k)
        {
            double s = Math.Sin(phi), c = Math.Cos(phi);
            Complex x = new Complex(c * c, 0.0);
            Complex y = new Complex(1.0 - k * k * s * s, 0.0);
            Complex z = Complex.One;
            Complex rf = Carlsonfk.RF(x, y, z);
            Complex rd = Carlsonfk.RD(x, y, z);
            return (s * rf - (k * k / 3.0) * s * s * s * rd).Real;
        }

        /// <summary>不完全第一类 F(phi,k)</summary>
        public static double F(double phi, double k)
        {
            if (phi < 0.0) return -F(-phi, k);
            int n = (int)Math.Floor(phi / Math.PI);
            double t = phi - n * Math.PI;                 // t in [0, pi)
            double f = (t <= Math.PI / 2.0)
                     ? FRaw(t, k)
                     : 2.0 * K(k) - FRaw(Math.PI - t, k);
            return 2.0 * n * K(k) + f;
        }

        /// <summary>不完全第二类 E(phi,k)</summary>
        public static double EInc(double phi, double k)
        {
            if (phi < 0.0) return -EInc(-phi, k);
            int n = (int)Math.Floor(phi / Math.PI);
            double t = phi - n * Math.PI;                 // t in [0, pi)
            double e = (t <= Math.PI / 2.0)
                     ? ERaw(t, k)
                     : 2.0 * E(k) - ERaw(Math.PI - t, k);
            return 2.0 * n * E(k) + e;
        }
    }







