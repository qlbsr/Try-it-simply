using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using UnityEngine;
using Complex = System.Numerics.Complex;
using Vector3 = UnityEngine.Vector3;
public class ThreeAnglesAnalyzer : MonoBehaviour
{
    void Start()
    {
        // ============ 1. 加载原始点，算 (r30, r45) ============
        var relist = JsonVectorParser.jsonpy("points2");
        List<Vector3> points = relist.Vector3List;
        float rp = nsjy.ComputeRp(points);
        var yzqx1 = nsjy.yzqx(points, out var r45, out var r30,
                               out var r74, rp);

        var (t30, t45) = nsjy.ComputeTaus();
        Debug.Log($"t30 = {t30}, t45 = {t45}");
        Debug.Log($"点数 = {r30.Length}");

        // ============ 2. 提取三个角 ============
        int N = r30.Length;
        double[] thPar = new double[N];
        double[] th30 = new double[N];
        double[] th45 = new double[N];

        for (int i = 0; i < N; i++)
        {
            var (p, a, b) = ThreeAngles.Extract(r30[i], r45[i], t30, t45);
            thPar[i] = p;
            th30[i] = a;
            th45[i] = b;
        }
        for (int i = 0; i < Math.Min(10, N); i++)
            Debug.Log($"P{i}: θ∥={thPar[i]:F4}, θ30={th30[i]:F4}, θ45={th45[i]:F4}");



    }

    // ====================================================
    // 圆形统计
    // ====================================================
    static void PrintCircularStats(double[] angles, string name)
    {
        // 圆形均值
        double cosSum = angles.Average(a => Math.Cos(a));
        double sinSum = angles.Average(a => Math.Sin(a));
        double meanAngle = Math.Atan2(sinSum, cosSum);
        double R = Math.Sqrt(cosSum * cosSum + sinSum * sinSum);

        // 圆形标准差（Mardia 公式）
        double std = Math.Sqrt(-2 * Math.Log(Math.Max(R, 1e-12)));

        Debug.Log($"[{name}] 圆形均值 = {meanAngle:F4}, " +
                  $"集中度 R = {R:F4}, 圆形标准差 = {std:F4}");
        // R ≈ 0.07（1/√N）→ 均匀分布
        // R > 0.3 → 集中
    }

    static double CircularCorrelation(double[] a, double[] b)
    {
        // 圆形-圆形相关系数（Jammalamadaka）
        int n = a.Length;
        double aBar = Math.Atan2(
            a.Average(x => Math.Sin(x)),
            a.Average(x => Math.Cos(x)));
        double bBar = Math.Atan2(
            b.Average(x => Math.Sin(x)),
            b.Average(x => Math.Cos(x)));

        double num = 0, denA = 0, denB = 0;
        for (int i = 0; i < n; i++)
        {
            double da = Math.Sin(a[i] - aBar);
            double db = Math.Sin(b[i] - bBar);
            num += da * db;
            denA += da * da;
            denB += db * db;
        }
        return num / Math.Sqrt(Math.Max(denA * denB, 1e-30));
    }

    // ====================================================
    // T³ 联合结构
    // ====================================================
    static double[] JointConcentration(double[] a, double[] b, double[] c)
    {
        int n = a.Length;
        // 三个角的联合圆形集中度
        Complex sum1 = Complex.Zero, sum2 = Complex.Zero, sum3 = Complex.Zero;
        for (int i = 0; i < n; i++)
        {
            sum1 += Complex.FromPolarCoordinates(1, a[i]);
            sum2 += Complex.FromPolarCoordinates(1, b[i]);
            sum3 += Complex.FromPolarCoordinates(1, c[i]);
        }
        double R = (sum1.Magnitude + sum2.Magnitude + sum3.Magnitude) / (3.0 * n);

        // 平均最近邻距离（T³ 上的环形距离）
        double sumNN = 0;
        int count = 0;
        for (int i = 0; i < Math.Min(n, 100); i++)
        {
            double minD = double.MaxValue;
            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;
                double d = TorusDistSq(
                    a[i], b[i], c[i],
                    a[j], b[j], c[j]);
                if (d < minD) minD = d;
            }
            sumNN += Math.Sqrt(minD);
            count++;
        }

        return new double[] { R, sumNN / count };
    }

    static double TorusDistSq(double a1, double b1, double c1,
                               double a2, double b2, double c2)
    {
        double da = WrapAngle(a1 - a2);
        double db = WrapAngle(b1 - b2);
        double dc = WrapAngle(c1 - c2);
        return da * da + db * db + dc * dc;
    }

    static double WrapAngle(double d)
    {
        while (d > Math.PI) d -= 2 * Math.PI;
        while (d < -Math.PI) d += 2 * Math.PI;
        return d;
    }

    // ====================================================
    // Fourier 分析
    // ====================================================
    static void PrintFourierModes(double[] a, double[] b, double[] c, int mode)
    {
        int n = a.Length;
        Complex A = Complex.Zero, B = Complex.Zero, C = Complex.Zero;
        for (int i = 0; i < n; i++)
        {
            A += Complex.FromPolarCoordinates(1, mode * a[i]);
            B += Complex.FromPolarCoordinates(1, mode * b[i]);
            C += Complex.FromPolarCoordinates(1, mode * c[i]);
        }
        A /= n; B /= n; C /= n;
        Debug.Log($"k={mode}: |θ∥|={A.Magnitude:F4}, " +
                  $"|θ30|={B.Magnitude:F4}, |θ45|={C.Magnitude:F4}");
        // 噪声级 = 1/√N ≈ 0.07
        // 显著 > 0.2 → 有 k 阶对称
    }

    // ====================================================
    // 最近邻距离
    // ====================================================
    static double[] NearestNeighborDistances(double[] a, double[] b, double[] c, int sampleCount)
    {
        int n = a.Length;
        int sample = Math.Min(sampleCount, n);
        double[] result = new double[sample];

        var rng = new System.Random(42);
        var indices = Enumerable.Range(0, n).OrderBy(_ => rng.Next()).Take(sample).ToArray();

        for (int k = 0; k < sample; k++)
        {
            int i = indices[k];
            double minD = double.MaxValue;
            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;
                double d = TorusDistSq(a[i], b[i], c[i], a[j], b[j], c[j]);
                if (d < minD) minD = d;
            }
            result[k] = Math.Sqrt(minD);
        }
        return result;
    }

    static double StdDev(double[] x)
    {
        double m = x.Average();
        return Math.Sqrt(x.Average(v => (v - m) * (v - m)));
    }

    // ====================================================
    // 关键：检测特殊相位关系
    // ====================================================
    static void CheckPhaseRelations(double[] a, double[] b, double[] c)
    {
        int n = a.Length;

        // 检验：θ30 - θ45 是否锁定？
        double[] diff = new double[n];
        for (int i = 0; i < n; i++)
            diff[i] = WrapAngle(b[i] - c[i]);
        double Rdiff = Math.Sqrt(
            Math.Pow(diff.Average(x => Math.Cos(x)), 2) +
            Math.Pow(diff.Average(x => Math.Sin(x)), 2));
        Debug.Log($"R(θ30 - θ45) = {Rdiff:F4}  (> 0.3 → 锁定)");

        // 检验：θ∥ + θ30 + θ45 是否锁在某值
        double[] sum3 = new double[n];
        for (int i = 0; i < n; i++)
            sum3[i] = WrapAngle(a[i] + b[i] + c[i]);
        double Rsum = Math.Sqrt(
            Math.Pow(sum3.Average(x => Math.Cos(x)), 2) +
            Math.Pow(sum3.Average(x => Math.Sin(x)), 2));
        Debug.Log($"R(θ∥+θ30+θ45) = {Rsum:F4}");

        // 检验：θ∥ = θ30 = θ45？(对角线)
        double diagR = 0;
        for (int i = 0; i < n; i++)
        {
            double d1 = WrapAngle(a[i] - b[i]);
            double d2 = WrapAngle(b[i] - c[i]);
            if (Math.Abs(d1) < 0.3 && Math.Abs(d2) < 0.3) diagR++;
        }
        Debug.Log($"接近对角线的比例 = {diagR / n:F4}");







    }









}
public static class ThreeAngles
{
    // z = a + b*tau, a,b ∈ [0,1)
    // 返回 (a, b)
    public static double DistFromAngles(
       double a, double b, Complex tau)
    {
        // z = a + b*tau
        double zr = a + b * tau.Real;
        double zi = b * tau.Imaginary;
        return Math.Sqrt(zr * zr + zi * zi);
    }

    // 乘积版本：三个角 + t30, t45 → 距离
    public static double DistFromThreeAngles(
        double thetaPar,
        double theta30,
        double theta45,
        Complex t30, Complex t45)
    {
        // 归一化到 [-0.5, 0.5)
        thetaPar = thetaPar - Math.Round(thetaPar);
        theta30 = theta30 - Math.Round(theta30);
        theta45 = theta45 - Math.Round(theta45);

        double d1 = DistFromAngles(thetaPar, theta30, t30);
        double d2 = DistFromAngles(thetaPar, theta45, t45);

        return Math.Sqrt(d1 * d1 + d2 * d2);
    }
    public static (double a, double b) GridCoords(
        Complex z, Complex tau,
        int M = 30, int N = 30)
    {
        Complex bestLambda = Complex.Zero;
        double bestDist = double.MaxValue;

        for (int m = -M; m <= M; m++)
            for (int n = -N; n <= N; n++)
            {
                Complex lambda = new Complex(m, 0)
                               + new Complex(n, 0) * tau;
                double d = (z - lambda).Magnitude;
                if (d < bestDist)
                {
                    bestDist = d;
                    bestLambda = lambda;
                }
            }

        Complex w = z - bestLambda;
        double b = w.Imaginary / tau.Imaginary;
        double a = w.Real - b * tau.Real;

        a = a - Math.Floor(a);
        b = b - Math.Floor(b);
        return (a, b);
    }

    // 三个角
    public static (double thPar, double th30, double th45) Extract(
        Complex z30, Complex z45,
        Complex t30, Complex t45,
        int M = 30, int N = 30)
    {
        var (a30, b30) = GridCoords(z30, t30, M, N);
        var (a45, b45) = GridCoords(z45, t45, M, N);

        // 共享实轴方向的相位：取 a30 和 a45 的平均
        double thPar = 2 * Math.PI * 0.5 * (a30 + a45);

        // 两个独立方向的相位
        double th30 = 2 * Math.PI * b30;
        double th45 = 2 * Math.PI * b45;

        return (thPar, th30, th45);
    }
}



public static class LieGroup
{
    // ============================================================
    // 1. 辛形式矩阵 J（2N×2N）
    // ============================================================
    public static Matrix<double> BuildJ(int N)
    {
        int n = 2 * N;
        var J = Matrix<double>.Build.Dense(n, n);
        for (int i = 0; i < N; i++)
        {
            J[i, N + i] = 1;
            J[N + i, i] = -1;
        }
        return J;
    }

    // ============================================================
    // 2. 李代数 → 李群：M = exp(X)
    //    X ∈ sp(2N, R)，M ∈ Sp(2N, R)
    // ============================================================
    public static Matrix<double> ExpToGroup(Matrix<double> X)
    {
        // MathNet 自带矩阵指数
        return X.Exponential();
    }

    // ============================================================
    // 3. 李群 → 李代数：X = log(M)
    // ============================================================
    public static Matrix<double> LogFromGroup(Matrix<double> M)
    {
        return M.Logarithm();
    }

    // ============================================================
    // 4. 投影到辛李代数
    //    X ∈ sp(2N) ⟺ X^T J + J X = 0
    //    投影公式：X_proj = 0.5 * (X - J^{-1} X^T J)
    // ============================================================
    public static Matrix<double> ProjectToSpLie(Matrix<double> X, Matrix<double> J)
    {
        var Jinv = J.Inverse();
        var Xt = X.Transpose();
        var term2 = Jinv.Multiply(Xt).Multiply(J);
        return 0.5 * (X - term2);
    }

    // ============================================================
    // 5. 把任意 2N×2N 矩阵参数化为辛李代数
    //    参数维度：N(2N+1)
    // ============================================================
    public static Matrix<double> FlatToSpLie(double[] flat, int N)
    {
        int n = 2 * N;
        var X = Matrix<double>.Build.Dense(n, n);

        // 分块：X = [[A, B], [C, -A^T]]
        // A: N×N 任意，B: N×N 对称，C: N×N 对称
        int idx = 0;

        // A 块（N×N）
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                X[i, j] = flat[idx++];

        // B 块（N×N 对称）
        for (int i = 0; i < N; i++)
            for (int j = i; j < N; j++)
            {
                double val = flat[idx++];
                X[i, N + j] = val;
                X[j, N + i] = val;
            }

        // C 块（N×N 对称）
        for (int i = 0; i < N; i++)
            for (int j = i; j < N; j++)
            {
                double val = flat[idx++];
                X[N + i, j] = val;
                X[N + j, i] = val;
            }

        // -A^T 块
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                X[N + i, N + j] = -X[j, i];

        return X;
    }

    public static int SpLieDim(int N) { return N * (2 * N + 1); }

    // ============================================================
    // 6. 等差递推（李代数坐标下）
    //    X_n = X_0 + n·d
    // ============================================================
    public static Matrix<double> LinearInterpInLie(
        Matrix<double> X0, Matrix<double> XN, int n, int N_total)
    {
        double t = (double)n / N_total;
        return (1 - t) * X0 + t * XN;
    }

    // ============================================================
    // 7. 前后推中间（算术平均）
    // ============================================================
    public static Matrix<double> InferMiddleInLie(
        Matrix<double> X_prev, Matrix<double> X_next)
    {
        return 0.5 * (X_prev + X_next);
    }

    // ============================================================
    // 8. Sp(2N, R) 作用在 Siegel 矩阵 Ω 上
    //    Ω → (A Ω + B)(C Ω + D)^{-1}
    // ============================================================
    public static Matrix<Complex> ApplyToSiegel(
        Matrix<double> M, Matrix<Complex> Omega, int N)
    {
        // 分块 M = [[A, B], [C, D]]
        var A = M.SubMatrix(0, N, 0, N).Map(x => new Complex(x, 0));
        var B = M.SubMatrix(0, N, N, N).Map(x => new Complex(x, 0));
        var C = M.SubMatrix(N, N, 0, N).Map(x => new Complex(x, 0));
        var D = M.SubMatrix(N, N, N, N).Map(x => new Complex(x, 0));

        var num = A * Omega + B;
        var den = C * Omega + D;
        return num * den.Inverse();
    }
}

public class MemorySystem
{
    // ============================================================
    // 状态
    // ============================================================
    public int N;                                // 椭圆曲线因子数
    public double[] X_total;                     // 李代数记忆（10 维 for N=2）
    public double thPar, th30, th45;             // 当前三个角
    public List<Complex> exploredAngles;         // 探索出的复数角

    // ============================================================
    // 构造
    // ============================================================
    public MemorySystem(int N)
    {
        this.N = N;
        X_total = new double[LieGroup.SpLieDim(N)];
        exploredAngles = new List<Complex>();
    }

    // ============================================================
    // 单步更新：从数据学习 + 累积
    // ============================================================
    public void Step(double[] X_delta)
    {
        // 1. 累积李代数
        for (int i = 0; i < X_total.Length; i++)
            X_total[i] += X_delta[i];

        // 2. 投影到合法李代数（可选，防止数值漂移）
        var J = LieGroup.BuildJ(N);
        var X = LieGroup.FlatToSpLie(X_total, N);
        X = LieGroup.ProjectToSpLie(X, J);

        // 3. 反展平
        X_total = SpLieToFlat(X, N);
    }

    private double[] SpLieToFlat(Matrix<double> X, int N)
    {
        var flat = new List<double>();
        // A 块
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                flat.Add(X[i, j]);
        // B 块（对称）
        for (int i = 0; i < N; i++)
            for (int j = i; j < N; j++)
                flat.Add(X[i, N + j]);
        // C 块（对称）
        for (int i = 0; i < N; i++)
            for (int j = i; j < N; j++)
                flat.Add(X[N + i, j]);
        return flat.ToArray();
    }

    // ============================================================
    // 累积路径长度（用于检测系统行为）
    // ============================================================
    public double PathNorm()
    {
        double s = 0;
        foreach (var x in X_total) s += x * x;
        return Math.Sqrt(s);
    }

    // ============================================================
    // 恢复 M_total
    // ============================================================
    public Matrix<double> GetMTotal()
    {
        var X = LieGroup.FlatToSpLie(X_total, N);
        return LieGroup.ExpToGroup(X);
    }

    // ============================================================
    // 从任意两点恢复整条链（等差假设）
    // ============================================================
    public double[][] DecodeChain(int totalSteps)
    {
        double[][] chain = new double[totalSteps + 1][];
        for (int n = 0; n <= totalSteps; n++)
        {
            chain[n] = new double[X_total.Length];
            double t = (double)n / totalSteps;
            for (int k = 0; k < X_total.Length; k++)
                chain[n][k] = t * X_total[k];  // X_0 = 0
        }
        return chain;
    }

    // ============================================================
    // 前后推中间
    // ============================================================
    public double[] InferMiddle(double[] X_prev, double[] X_next)
    {
        double[] mid = new double[X_prev.Length];
        for (int i = 0; i < mid.Length; i++)
            mid[i] = 0.5 * (X_prev[i] + X_next[i]);
        return mid;
    }
}
public static class ComplexToAngle
{
    // ============================================================
    // 1. 模参数 τ → 角度（via λ(τ) 的相位）
    // ============================================================
    public static double TauToAngle(Complex tau)
    {
        Complex q = Complex.Exp(new Complex(0, Math.PI) * tau);
        Complex t2 = Theta2(q);
        Complex t3 = Theta3(q);
        Complex lambda = Complex.Pow(t2 / t3, 4);
        return lambda.Phase;   // ∈ (-π, π]
    }

    // ============================================================
    // 2. 角度 → 模参数 τ（粗反解）
    // ============================================================
    public static Complex AngleToTau(double angle)
    {
        // 从 λ 反推 τ 需要数值求解
        // 简化版本：默认映射到 i 附近的模参数
        Complex lambda_target = Complex.FromPolarCoordinates(1, angle);

        // 用牛顿法解 λ(τ) = lambda_target
        Complex tau = new Complex(0, 1);   // 初值
        for (int iter = 0; iter < 30; iter++)
        {
            Complex q = Complex.Exp(new Complex(0, Math.PI) * tau);
            Complex t2 = Theta2(q);
            Complex t3 = Theta3(q);
            Complex lambda = Complex.Pow(t2 / t3, 4);

            Complex diff = lambda - lambda_target;
            if (diff.Magnitude < 1e-8) break;

            // 数值导数
            Complex h = new Complex(1e-6, 0);
            Complex qh = Complex.Exp(new Complex(0, Math.PI) * (tau + h));
            Complex lambda_h = Complex.Pow(Theta2(qh) / Theta3(qh), 4);
            Complex dlambda = (lambda_h - lambda) / h;

            if (dlambda.Magnitude < 1e-15) break;
            tau -= diff / dlambda;
        }
        return tau;
    }

    // ============================================================
    // 3. 复数角 jd2 → (角度, 半径)
    // ============================================================
    public static (double angle, double radius) Jd2ToAngleRadius(Complex jd2)
    {
        return (jd2.Phase, jd2.Magnitude);
    }

    // ============================================================
    // 4. (角度, 半径) → 复数角 jd2
    // ============================================================
    public static Complex AngleRadiusToJd2(double angle, double radius)
    {
        return Complex.FromPolarCoordinates(radius, angle);
    }

    // ============================================================
    // 5. 计算复数角 jd2（从 d1, d2）
    // ============================================================
    public static Complex ComputeComplexJd2(double d1, double d2)
    {
        double k = Math.Sin(d2 * Math.PI / 180);
        if (Math.Abs(k) < 1e-10) return Complex.Zero;

        double dk = 1 - 4 * k * k;
        Complex jk = Complex.Sqrt(new Complex(dk, 0));

        Complex e1 = (1 - jk) / (2 * k);
        Complex e2 = (1 + jk) / (2 * k);

        Complex sj = (e2 * e2 - 1) / (e2 * e2 + 1);

        return Complex.Asin(sj) * (180 / Math.PI);
    }

    // ============================================================
    // 6. 从 jd2 反推 (d1, d2)
    // ============================================================
    public static (double d1, double d2) Jd2ToAngles(Complex jd2)
    {
        // jd2 = asin(sj)·180/π → sj = sin(jd2·π/180)
        Complex sj = Complex.Sin(jd2 * Math.PI / 180);

        // sj = (e2²-1)/(e2²+1) → e2² = (1+sj)/(1-sj)
        Complex e2_sq = (1 + sj) / (1 - sj);
        Complex e2 = Complex.Sqrt(e2_sq);

        // 从 e2 反推 k：2k·e2 = 1 + jk，且 jk² = 1 - 4k²
        // 简化：假设 k 接近 0.5
        double k = 1.0 / (e2.Magnitude + 1);
        k = Math.Max(0.001, Math.Min(0.999, k));

        double d2 = Math.Asin(k) * 180 / Math.PI;
        double d1 = 45 - d2;
        return (d1, d2);
    }

    // ============================================================
    // 7. 环面距离
    // ============================================================
    public static double TorusDistance(double a, double b)
    {
        double d = Math.Abs(a - b);
        return Math.Min(d, 2 * Math.PI - d);
    }

    public static double TorusDistanceN(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = TorusDistance(a[i], b[i]);
            sum += d * d;
        }
        return Math.Sqrt(sum);
    }

    // ============================================================
    // 8. 约束检查（4D/5D/6D）
    // ============================================================
    public static bool PassAllConstraints(Complex jd2)
    {
        // 4D：合法性
        if (double.IsNaN(jd2.Real) || double.IsNaN(jd2.Imaginary))
            return false;

        var (d1, d2) = Jd2ToAngles(jd2);
        if (d1 < 0 || d1 > 45) return false;
        if (d2 < 0 || d2 > 45) return false;

        // 5D：η 合法性
        double eta = jd2.Imaginary;
        if (Math.Abs(eta) > 10.0) return false;
        if (Math.Abs(eta) < 1e-6) return false;

        // 6D：Hecke 闭合
        if (!IsHeckeClosed(eta, 1.0)) return false;

        return true;
    }

    public static bool IsHeckeClosed(double eta, double latticePeriod,
                                     double tol = 1e-3, int maxT = 100)
    {
        if (Math.Abs(eta) < 1e-6) return true;
        for (int T = 1; T <= maxT; T++)
        {
            double product = eta * T / latticePeriod;
            if (Math.Abs(product - Math.Round(product)) < tol)
                return true;
        }
        return false;
    }

    // ============================================================
    // 9. Theta 函数
    // ============================================================
    public static Complex Theta2(Complex q)
    {
        Complex sum = Complex.Zero;
        for (int n = 0; n < 60; n++)
            sum += Complex.Pow(q, n * (n + 1));
        return 2 * Complex.Pow(q, 0.25) * sum;
    }

    public static Complex Theta3(Complex q)
    {
        Complex sum = Complex.One;
        for (int n = 1; n < 60; n++)
            sum += 2 * Complex.Pow(q, n * n);
        return sum;
    }
}