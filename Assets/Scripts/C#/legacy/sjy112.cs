using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using System.Numerics;
using Vector3 = UnityEngine.Vector3;
using Complex = System.Numerics.Complex;
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

