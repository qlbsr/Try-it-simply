using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using UnityEngine;
using static alglib;
using Quaternion = UnityEngine.Quaternion;
using Vector3 = UnityEngine.Vector3;

public class n2sjy2 : MonoBehaviour
{
    public Complex[] r45;
    public Complex[] r30;
    public Complex[] r16;
    public Complex[] r74;
    public float rp;
    public float a;
    public Vector3 f1z;
    public Vector3 f2z;
    public List<Vector3> points;//角平方线就是两个向量相加
   
    private void Start()
    {
        var (t1, t2) = ComputeTaus();
        VectorList vectorList = JsonVectorParser.jsonpy("pyjson");
        points = vectorList.Vector3List;
        rp = ComputeRp(points);
        float d2 = Mathf.Asin(Mathf.Cos(30 * Mathf.Deg2Rad) / Mathf.PI);
        a = rp * (1 + Mathf.Sin(d2));
        float e3 = (float)(Math.Cos(d2) * 2 / (1 + Math.Sin(d2)));
        float h = rp * Mathf.Cos(d2);
        float c = h * e3;
        List<(Complex x, Complex y)> yzqx1 = yzqx(points, out r45, out r30, out r74, rp);
        Complex t_ = new Complex(0, 1);
        Vector3 v = pca(points);
        float[][] probs = DeviationCalculator.ComputeProbabilitiesFromTaus(r30, r45, t1, t2, a);
        var (F1, F2) = ExtractFoci(points, probs[1], probs[3][1], probs[2], probs[3][2]);
        Vector3 v1 = (F1 - F2).normalized;
        var (f1, f2) = FitFociByProbability(points, probs[0], F1, F2, a);
        Vector3 v2 = (f1 - f2).normalized;
      float[][] probs01 = DeviationCalculator.ComputeProbabilitiesFromTaus(r30, r45, t_, t_, a);
        var (F01,F02) = ExtractFoci(points, probs01[1], probs01[3][1], probs01[2], probs01[3][2]);
        Vector3 v3 = (F01 - F02).normalized;
        var (f01, f02) = FitFociByProbability(points, probs01[0], F01, F02, a);
        Vector3 v4 = (f01 - f02).normalized;
        Vector3 vj = v.normalized + (F1 - F2).normalized;
        var (t1_01, t2_01, F1_n, F2_n) = RefineModuliByAxis(points, t_, t_, vj, 50);
        float angleDegv = Vector3.Angle(vj, v);
        for (int i =0;i < 50; i++)
        {
            float[][] probsnew = DeviationCalculator.ComputeProbabilitiesFromTaus(r30, r45, t1, t2, a);
            var (t1_new, t2_new, F1_new, F2_new) = RefineModuliByAxis(points, t_, t_, (f1 - f2).normalized, 50);
            // var (f11, f22) = FitFociByProbability(points, probsnew[0], F1_new , F2_new, a);
            float angleDeg = Vector3.Angle((f1 - f2), v);
            float angleDeg1 = Vector3.Angle((f1 - f2), (F1 - F2));
            Debug.Log((angleDeg, angleDeg1, angleDegv));
            if ((angleDeg + angleDeg1) * 0.5 - Math.Min(angleDeg, angleDeg1) <= 8) { break; }
            Vector3 axis = Vector3.Cross((F1_new - F2_new).normalized, (F1 - F2).normalized);
            Vector3 axis1 = Vector3.Cross((F1_new - F2_new).normalized, (F01 - F02).normalized);
            Quaternion rotation = Quaternion.AngleAxis(angleDeg * -1, axis);
            Quaternion rotation1 = Quaternion.AngleAxis(angleDeg * -1, axis1);
            Vector3 newv = rotation * (F01 -F02).normalized ;
            Vector3 newF = rotation1 * (F1 - F2).normalized;
            var (t1_, t2_ ,deltaDeg,angleDeg_,angleDeg1_,evals,F1z,F2z) = RefineTausWithNM(t1_new, t2_new, points, r30, r45, a, newv, newF);
            t1 = t1_;
            t2 = t2_;
            f1 = F1z;
            f2 = F2z;
            this.f1z = f1;
            this.f2z = f2;
            Debug.Log((angleDeg_, angleDeg1_));
            Debug.Log((t1, t2));
        }
        float[] s0 = BatchProbability(v*c, v*-c, points, a);
        float[] s1 = BatchProbability(v1 * c, v1 * -c, points, a);
        float[] s2 = BatchProbability(v2 * c, v2 * -c, points, a);
        float[] s3 = BatchProbability(v3 * c, v3 * -c, points, a);
        float[] s4 = BatchProbability(v4 * c, v4 * -c, points, a);
        Vector3 v5 = (f1z - f2z).normalized;
        float[] s5 = BatchProbability(v5 * c, v5 * -c, points, a);
        Debug.Log((s0[0], s1[0], s2[0], s3[0], s4[0], s5[0]));

























    }
    /// <summary>
    /// 从当前 t1,t2 出发，使用 Nelder-Mead 直接最小化 |Δ|。
    /// 输入已预处理的数据，避免重复计算 rp, cone, a, r30, r45, v, d0。
    /// </summary>
    public static (Complex t1, Complex t2, float deltaDeg, float angleDeg, float angleDeg1, int evals,Vector3 F1z,Vector3 F2z)
        RefineTausWithNM(
            Complex t1Init,
            Complex t2Init,
            List<Vector3> points,
            Complex[] r30,
            Complex[] r45,
            float a,
            Vector3 pcav,
            Vector3 F1F2v)
    {
       
        double[] lb = { -0.5, 0.5, -0.5, 0.5 };
        double[] ub = { 0.5, 1.5, 0.5, 1.5 };
        Vector3 F1z;
        Vector3 F2z;

        // 目标函数：|Δ|
        float Objective(double[] x, out float angleDeg, out float angleDeg1)
        {
            var t1 = new Complex(x[0], x[1]);
            var t2 = new Complex(x[2], x[3]);
            float[][] probs = DeviationCalculator.ComputeProbabilitiesFromTaus(r30, r45, t1, t2, a);
            var (F1, F2) = nsjy.ExtractFoci(points, probs[1], probs[3][1], probs[2], probs[3][2]);
            var (f1, f2) = FitFociByProbability(points, probs[0], F1, F2, a);
            Vector3 d = f1 - f2;
            F1z = f1;
            F2z = f2;
            if (d.sqrMagnitude < 1e-12f)
            {
                angleDeg = 180f;
                angleDeg1 = 180f;
                return 0f;
            }
            d.Normalize();
            angleDeg = Vector3.Angle(d,pcav);
            angleDeg1 = Vector3.Angle(d, F1F2v);
            return Mathf.Abs(angleDeg - angleDeg1);
        }

        // 初始点
        double[] x0 = { t1Init.Real, t1Init.Imaginary, t2Init.Real, t2Init.Imaginary };

        // 调用内嵌 Nelder-Mead（或复用 nsjy4 中的 NelderMead.Minimize）
        var (bestX, bestF, evals) = NelderMead.Minimize(
            x => Objective(x, out _, out _), x0, lb, ub,
            maxEvals: 600, stopF: 10f);   // 目标 |Δ| ≤ 10°

        Complex t1Opt = new Complex(bestX[0], bestX[1]);
        Complex t2Opt = new Complex(bestX[2], bestX[3]);
        Objective(bestX, out float ad, out float ad1);
      
        return (t1Opt,t2Opt,(float)bestF, ad, ad1, evals,F1z,F2z);
    }
    public static Vector3 pca(List<Vector3> points)
    {
        double[] r = new double[3];
        int dim = 3;
        var matrix = DenseMatrix.Create(points.Count, dim, (i, j) =>
        {
            Vector3 p = points[i];
            if (j == 0) return p.x;
            if (j == 1) return p.y;
            return p.z;
        });
        var pca = new PCAScikitLearn();
        pca.Fit(matrix, nComponents: 3);
        r[0] = pca.ExplainedVarianceRatio[0];
        r[1] = pca.ExplainedVarianceRatio[1];
        r[2] = pca.ExplainedVarianceRatio[2];
        Vector3 pc1 = new Vector3(
    (float)pca.Components[0, 0],
    (float)pca.Components[0, 1],
    (float)pca.Components[0, 2]);
        float absX = Mathf.Abs(pc1.x);
        float absY = Mathf.Abs(pc1.y);
        float absZ = Mathf.Abs(pc1.z);
        float max = Mathf.Max(absX, absY, absZ);
        float fj = Mathf.Atan2(pc1.z, pc1.x) * Mathf.Rad2Deg;
        float d2 = Mathf.Acos(pc1.y) * Mathf.Rad2Deg;
        Vector3 v3 = new Vector3(
              Mathf.Sin(d2 * Mathf.Deg2Rad) * Mathf.Cos(fj * Mathf.Deg2Rad),
              Mathf.Cos(d2 * Mathf.Deg2Rad),
              Mathf.Sin(d2 * Mathf.Deg2Rad) * Mathf.Sin(fj * Mathf.Deg2Rad)
          );
        return v3;

    }
    public static float ComputeRp(List<Vector3> points)
    {
        if (points.Count == 0) return 0f;

        float sum = 0f;
        foreach (var p in points)
        {
            sum += p.magnitude; // sqrt(x² + y² + z²)
        }
        float rp = sum / points.Count;


        return rp;
    }
    public static (Complex tau1, Complex tau2) ComputeTaus()
    {
        Complex K2 = Carlsonfk.K(2.0); // K(2)

        Complex Kprime2 = Carlsonfk.K(Complex.Sqrt(1 - 2.0 * 2.0)); // K(i√3)

        Complex tau1 = Complex.ImaginaryOne * Kprime2 / K2;


        Complex Ksqrt2 = Carlsonfk.K(Math.Sqrt(2)); // K(√2)

        Complex KprimeSqrt2 = Carlsonfk.K(Complex.Sqrt(1 - 2.0)); // K(i)

        Complex tau2 = Complex.ImaginaryOne * KprimeSqrt2 / Ksqrt2;

        if (tau1.Imaginary < 0) tau1 = -tau1;
        if (tau2.Imaginary < 0) tau2 = -tau2;

        return (tau1, tau2);
    }

    public static Complex InverseTh4(Vector3 vs3, float d2, float rp)
    {
        d2 = d2 * Mathf.Deg2Rad;
        // 1. 计算 rs, zs, phis
        float rs = Mathf.Sqrt(vs3.x * vs3.x + vs3.y * vs3.y);
        float zs = vs3.z;
        float phis = Mathf.Atan2(vs3.y, vs3.x);
        if (phis < 0) phis += 2 * Mathf.PI;

        // 2. 计算 r 和 thetas
        float r = Mathf.Sqrt(rs * rs + zs * zs);
        float sin_d2 = Mathf.Sin(d2);
        float cos_d2 = Mathf.Cos(d2);
        // 验证：rs = r * sin_d2, zs = r * cos_d2，若不一致则可能有误差
        // 但假设数据一致，直接由 phis 求 thetas
        float thetas = phis / sin_d2;
        // 将 thetas 归一化到 [0, 2π)
        while (thetas < 0) thetas += 2 * Mathf.PI;
        while (thetas >= 2 * Mathf.PI) thetas -= 2 * Mathf.PI;

        // 3. 计算 Rs
        float Rs = Mathf.Pow(r / rp, 1.0f / sin_d2);

        // 4. 构造 uvs
        float u = Rs * Mathf.Cos(thetas);
        float v = Rs * Mathf.Sin(thetas);

        return new Complex(u, v);
    }
    public static List<(Complex x, Complex y)> yzqx(List<Vector3> points, out Complex[] r45, out Complex[] r30, out Complex[] r74, float rp)
    {
        if (points == null || points.Count == 0)
        {
            r45 = null;
            r30 = null;
            r74 = null;
            return new List<(Complex x, Complex y)>();
        }

        int count = points.Count;
        r45 = new Complex[count];
        r30 = new Complex[count];
        r74 = new Complex[count];
        // 第一步：计算原始值
        for (int i = 0; i < count; i++)
        {
            r45[i] = InverseTh4(points[i], 45, rp);
            r30[i] = InverseTh4(points[i], 30, rp);
            r74[i] = InverseTh4(points[i], 74, rp);
        }

        //Debug.Log($"[{string.Join(", ", r45.Select(v => v.ToString("F6")))}]");
        //Debug.Log($"[{string.Join(", ", r30.Select(v => v.ToString("F6")))}]");
        // 第二步：构造原始列表（包含所有组合）
        var rawList = new List<(Complex x, Complex y)>();

        for (int i = 0; i < count; i++)
        {
            rawList.Add((r30[i], r45[i]));
        }
        // 第三步：过滤无效值（NaN 或 Infinity）
        var validList = rawList
            .Where(t => !(double.IsNaN(t.x.Real) || double.IsInfinity(t.x.Real) ||
                          double.IsNaN(t.x.Imaginary) || double.IsInfinity(t.x.Imaginary) ||
                          double.IsNaN(t.y.Real) || double.IsInfinity(t.y.Real) ||
                          double.IsNaN(t.y.Imaginary) || double.IsInfinity(t.y.Imaginary)))
            .ToList();

        if (validList.Count == 0)
            return validList; // 全部无效，返回空列表

        // 第四步：过滤“极大值”（这里使用模长阈值）
        // 计算所有模长的绝对值，取 99.9% 分位数作为阈值，或直接定义固定阈值（例如 1e10）
        double threshold = 1e10; // 可根据数据调整
        var filteredList = validList
            .Where(t => t.x.Magnitude < threshold && t.y.Magnitude < threshold)
            .ToList();

        if (filteredList.Count == 0)
            return filteredList;

        double maxMagnitude = filteredList
       .SelectMany(t => new[] { t.x.Magnitude, t.y.Magnitude })
       .Max();

        if (maxMagnitude > 0)
        {
            var normalizedList = filteredList
                .Select(t => (x: t.x / maxMagnitude, y: t.y / maxMagnitude))
                .ToList();
            return normalizedList;
        }
        else
        {
            // 若最大模长为 0（全零），则无需归一化，直接返回
            return filteredList;
        }

    }
    public static (Vector3 F1, Vector3 F2) FitFociByProbability(
    List<Vector3> points, float[] trueProb,
    Vector3 initF1, Vector3 initF2, float a,
    int maxIter = 300, float learningRate = 0.0001f, float lambdaSep = 1.0f)
    {
        Vector3 F1 = initF1;
        Vector3 F2 = initF2;
        float eps = 1e-3f; // 更大的安全距离

        for (int iter = 0; iter < maxIter; iter++)
        {
            Vector3 gradF1 = Vector3.zero;
            Vector3 gradF2 = Vector3.zero;
            float loss = 0f;

            for (int i = 0; i < points.Count; i++)
            {
                Vector3 p = points[i];
                float d1 = Vector3.Distance(p, F1);
                float d2 = Vector3.Distance(p, F2);
                float safeD1 = Mathf.Max(d1, eps);
                float safeD2 = Mathf.Max(d2, eps);

                float delta = d1 + d2 - 2f * a;
                float fitProb = Mathf.Exp(-Mathf.Abs(delta) / (2f * a));
                float diff = fitProb - trueProb[i];
                loss += diff * diff;

                float sign = delta >= 0 ? 1f : -1f;
                float coef = diff * fitProb * sign / (2f * a);

                Vector3 grad1 = coef * (p - F1) / safeD1;
                Vector3 grad2 = coef * (p - F2) / safeD2;

                // 单点梯度裁剪
                gradF1 += Vector3.ClampMagnitude(grad1, 1.0f);
                gradF2 += Vector3.ClampMagnitude(grad2, 1.0f);
            }

            // 焦点分离正则
            Vector3 sep = F1 - F2;
            float sepLoss = lambdaSep * sep.sqrMagnitude;
            loss += sepLoss;
            gradF1 += 2f * lambdaSep * sep;
            gradF2 -= 2f * lambdaSep * sep;

            // 整体梯度裁剪
            gradF1 = Vector3.ClampMagnitude(gradF1, 5f);
            gradF2 = Vector3.ClampMagnitude(gradF2, 5f);

            F1 -= learningRate * gradF1;
            F2 -= learningRate * gradF2;

            // 检查数值
            if (!IsFinite(F1) || !IsFinite(F2))
            {
                Debug.LogWarning("数值发散，停止迭代");
                break;
            }

            if (loss < 1e-12f) break;
        }

        return (F1, F2);
    }

    static bool IsFinite(Vector3 v)
    {
        return !float.IsNaN(v.x) && !float.IsNaN(v.y) && !float.IsNaN(v.z) &&
               !float.IsInfinity(v.x) && !float.IsInfinity(v.y) && !float.IsInfinity(v.z);
    }
    public static float[] BatchProbability(Vector3 F1, Vector3 F2, List<Vector3> points, float a)
    {
        float[] fff = new float[points.Count];
        for (int i = 0; i < points.Count; i++)
        {
            Vector3 p = points[i];
            float d1 = Vector3.Distance(p, F1);
            float d2 = Vector3.Distance(p, F2);
            float delta = d1 + d2 - 2f * a;
            fff[i] = Mathf.Exp(-Mathf.Abs(delta) / (2 * a));
        }
        return fff;

    }
    public static (Vector3 F1, Vector3 F2) ExtractFoci(
    List<Vector3> points,
    float[] probFocus1, float sigma1,   // 统一 float
    float[] probFocus2, float sigma2)
    {
        int n = points.Count;
        if (n < 4 || probFocus1.Length != n || probFocus2.Length != n)
            throw new System.ArgumentException("点数和概率数组长度不匹配或点数不足4");

        List<float> dist1 = new List<float>(n);
        List<float> dist2 = new List<float>(n);

        for (int i = 0; i < n; i++)
        {
            float p1 = Mathf.Clamp(probFocus1[i], 1e-6f, 1f - 1e-6f);
            float p2 = Mathf.Clamp(probFocus2[i], 1e-6f, 1f - 1e-6f);
            dist1.Add(sigma1 * Mathf.Sqrt(-2f * Mathf.Log(p1)));
            dist2.Add(sigma2 * Mathf.Sqrt(-2f * Mathf.Log(p2)));
        }

        Vector3 F1 = FitFocus(points, dist1);
        Vector3 F2 = FitFocus(points, dist2);
        return (F1, F2);
    }
    public static Vector3 FitFocus(List<Vector3> points, List<float> distances)
    {
        int n = points.Count;
        Vector3 p0 = points[0];
        float d0 = distances[0];

        // 构建线性系统 A * F = b
        var A = Matrix<double>.Build.Dense(n - 1, 3);
        var b = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(n - 1);

        for (int i = 1; i < n; i++)
        {
            Vector3 diff = points[i] - p0;
            A[i - 1, 0] = 2.0 * diff.x;
            A[i - 1, 1] = 2.0 * diff.y;
            A[i - 1, 2] = 2.0 * diff.z;

            double right = points[i].sqrMagnitude - p0.sqrMagnitude
                           - (distances[i] * distances[i] - d0 * d0);
            b[i - 1] = right;
        }

        // 最小二乘解
        var At = A.Transpose();
        var AtA = At * A;
        var Atb = At * b;
        var solution = AtA.Solve(Atb);

        return new Vector3((float)solution[0], (float)solution[1], (float)solution[2]);
    }

    public (Complex t1, Complex t2, Vector3 F1, Vector3 F2) RefineModuliByAxis(
      List<Vector3> pts, Complex t10, Complex t20, UnityEngine.Vector3 pcaAxis,
      int maxIter = 50,
      float angleTolDeg = 0.5f,
      float fdH = 1e-3f,        // 数值雅可比步长（按 τ 尺度）
      float wDir = 1f,         // 方向闭合项权重（硬约束，优先保证闭合）
      float wSelf = 1f,         // 概率自洽项权重
      float wTheory = 1e-3f) // 理论正则权重（防止病态漂移）
    {
        int n = pts.Count;
        if (n < 4) throw new ArgumentException("至少 4 个点");

        // ---- 固定预处理（与 Start 一致）----
        rp = ComputeRp(pts);
        float cone = Mathf.Asin(Mathf.Cos(30f * Mathf.Deg2Rad) / Mathf.PI); // ≈16°
        a = rp * (1f + Mathf.Sin(cone));
        r30 = new Complex[n]; r45 = new Complex[n];
        for (int i = 0; i < n; i++) { r30[i] = InverseTh4(pts[i], 30, rp); r45[i] = nsjy.InverseTh4(pts[i], 45, rp); }
        ;
        Vector3 axis = pcaAxis.normalized;
        double[] x = { t10.Real, t10.Imaginary, t20.Real, t20.Imaginary };
        // ---- 优化变量 x = [Re t1, Im t1, Re t2, Im t2]，每步约束到基本域 ----
        double[] r = null;
        Vector3 F1 = Vector3.zero, F2 = Vector3.zero;
        double cost = double.MaxValue, lambda = 1e-3;

        for (int iter = 0; iter < maxIter; iter++)
        {
            (r, F1, F2) = EvaluateResiduals(x, pts, axis, wDir, wSelf, wTheory, t10, t20);
            cost = 0; foreach (double v in r) cost += v * v;

            Vector3 dir = F1 - F2;
            float angleDeg = 180f;
            if (dir.sqrMagnitude > 1e-12f)
            {
                dir.Normalize();
                if (Vector3.Dot(dir, axis) < 0) dir = -dir;   // 只约束“直线方向”，忽略 ±
                angleDeg = Vector3.Angle(dir, axis);
            }
          
            Debug.Log($"[iter {iter}] cost={cost:E3} angle={angleDeg:F3}°  t1=({x[0]:F4},{x[1]:F4}) t2=({x[2]:F4},{x[3]:F4})");
            if (angleDeg < angleTolDeg) { Debug.Log("闭环收敛"); break; }

            // ---- 数值雅可比 J(m×4) ----
            double[,] J = new double[r.Length, 4];
            for (int k = 0; k < 4; k++)
            {
                double[] xp = (double[])x.Clone(); xp[k] += fdH;
                NormalizeTau(xp);
                var (rp2, _, _) = EvaluateResiduals(xp, pts, axis, wDir, wSelf, wTheory, t10, t20);
                for (int i = 0; i < r.Length; i++) J[i, k] = (rp2[i] - r[i]) / fdH;
            }

            // ---- LM 法方程 (JᵀJ + λ·diag)Δ = −Jᵀr ----
            double[,] A = new double[4, 4]; double[] g = new double[4];
            for (int i = 0; i < r.Length; i++)
                for (int c = 0; c < 4; c++)
                {
                    g[c] += J[i, c] * r[i];
                    for (int d = 0; d < 4; d++) A[c, d] += J[i, c] * J[i, d];
                }
            double[,] Aaug = (double[,])A.Clone();
            for (int c = 0; c < 4; c++) Aaug[c, c] += lambda * (A[c, c] + 1e-12);
            double[] delta = Solve4(Aaug, g);

            // ---- 阻尼线搜索 ----
            double[] xtry = (double[])x.Clone();
            for (int c = 0; c < 4; c++) xtry[c] -= delta[c];
            NormalizeTau(xtry);
            var (r2, _, _) = EvaluateResiduals(xtry, pts, axis, wDir, wSelf, wTheory, t10, t20);
            double cost2 = 0; foreach (double v in r2) cost2 += v * v;
            if (cost2 < cost) { x = xtry; lambda = Math.Max(lambda / 3.0, 1e-8); }
            else { lambda = Math.Min(lambda * 3.0, 1e4); }
        }

        (r, F1, F2) = EvaluateResiduals(x, pts, axis, wDir, wSelf, wTheory, t10, t20);
        return (new Complex(x[0], x[1]), new Complex(x[2], x[3]), F1, F2);
    }

    /// <summary>残差：前 n 项=概率自洽(除以√n)，随后 3 项=方向闭合，最后 4 项=理论正则</summary>
    private (double[] r, UnityEngine.Vector3 F1, Vector3 F2) EvaluateResiduals(
        double[] x, List<Vector3> pts, Vector3 axis,
        float wDir, float wSelf, float wTheory, Complex t10, Complex t20)
    {
        int n = pts.Count;
        Complex t1 = new Complex(x[0], x[1]);
        Complex t2 = new Complex(x[2], x[3]);

        // 主链路：t ─► 格概率 ─► 焦点
        float[][] probs = DeviationCalculator.ComputeProbabilitiesFromTaus(r30, r45, t1, t2, a);
        var (F1, F2) = ExtractFoci(pts, probs[1], probs[3][1], probs[2], probs[3][2]);

        // 修正视角：从焦点反推概率（注意括号！）


        
        double[] r = new double[n + 9];
        double invSqrtN = 1.0 / Math.Sqrt(n);
        for (int i = 0; i < n; i++)
        {
            Vector3 p = pts[i];
            float d1 = Vector3.Distance(p, F1);
            float d2 = Vector3.Distance(p, F2);
            float delta = d1 + d2 - 2f * a;
            float probFoci = Mathf.Exp(-Mathf.Abs(delta) / (2f * a));  // 修正后的 BatchProbability
            r[i] = wSelf * (probs[0][i] - probFoci) * invSqrtN;
        }

        double im1 = x[1];
        double dev1 = 0;
        for(int i = 0; i < 100; i++)
        {
            if(im1 - Math.Pow(2,dev1) < 0)
            {
                break;
            }
            dev1 += 1;
        }
        double dy = 1;
        for (int i =0; i< dev1; i++)
        {
            dy += 1 / Math.Pow(2, i);
        }
        // 闭合条件：(F1−F2) 平行于 PCA 主轴
        Vector3 dir = F1 - F2;
        if (dir.sqrMagnitude < 1e-12f) dir = axis;   // 焦点退化保护
        dir.Normalize();
        if (Vector3.Dot(dir, axis) < 0) dir = -dir;
        Vector3 e = dir - axis;
        r[n] = wDir * e.x;
        r[n + 1] = wDir * e.y;
        r[n + 2] = wDir * e.z;
        // 在 r 数组长度改为 n + 11
       

        // ... 原有概率自洽和方向闭合不变 ...

        // 新增：将 t1 拉向 (0,1)
      
        // 理论正则（弱）
        r[n + 3] = wTheory * (x[0] - t10.Real);
        r[n + 4] = wTheory * (x[1] - t10.Imaginary);
        r[n + 5] = wTheory * (x[2] - t20.Real);
        r[n + 6] = wTheory * (x[3] - t20.Imaginary);
       
        r[n + 7] = dy * (x[1] - 1);   // Im t1 接近 1
      
        r[n + 8] = dy* (x[3] - 1);   // Im t2 接近 1
        return (r, F1, F2);
    }

    /// <summary>τ 归一化到基本域：Im>0、|Re|≤0.5、|τ|≥1（消除模群冗余）</summary>
    private void NormalizeTau(double[] x)
    {
        for (int k = 0; k < 4; k += 2)
        {
            double re = x[k];
            double im = x[k + 1];

            // 1. 实部 ≤ 0

            for (int iter = 0; iter < 100; iter++)
            {
                // 实部平移到 [-0.5, 0.5]
                re -= Math.Round(re);

                double mod2 = re * re + im * im;
                if (mod2 >= 1.0 - 1e-14)
                    break;   // 已满足 |τ| ≥ 1

                // 模长 < 1，应用 τ ← −1/τ
                double den = mod2;
                double newRe = -re / den;
                double newIm = im / den;
                re = newRe;
                im = newIm;
            }
            x[k] = re;
            x[k + 1] = im;
        }
    }

    private static double[] Solve4(double[,] A, double[] b)
    {
        int n = 4;
        double[,] aug = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) aug[i, j] = A[i, j];
            aug[i, n] = b[i];
        }
        for (int i = 0; i < n; i++)
        {
            int piv = i;
            for (int k = i + 1; k < n; k++)
                if (Math.Abs(aug[k, i]) > Math.Abs(aug[piv, i])) piv = k;
            for (int j = i; j <= n; j++) { double t = aug[i, j]; aug[i, j] = aug[piv, j]; aug[piv, j] = t; }
            if (Math.Abs(aug[i, i]) < 1e-15) continue;
            for (int k = i + 1; k < n; k++)
            {
                double f = aug[k, i] / aug[i, i];
                for (int j = i; j <= n; j++) aug[k, j] -= f * aug[i, j];
            }
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = aug[i, n];
            for (int j = i + 1; j < n; j++) x[i] -= aug[i, j] * x[j];
            if (Math.Abs(aug[i, i]) > 1e-15) x[i] /= aug[i, i];
        }
        return x;
    }
}
public static class DeviationCalculator
{




    /// <summary>
    /// 软最小距离: softmin(z) = -γ·ln(Σ_λ exp(-|z-λ|/γ))，数值稳定形式
    ///   = minDist - γ·ln(Σ exp(-(dist-minDist)/γ))。
    /// γ 自适应: γ = gammaFraction × scale，
    ///   scale 默认 = 最近两格点距离间隙 dists[1]-dists[0]（≈格点间距，
    ///   即"点云到格点距离"的自然尺度；minDist≈0 时退化为格点间距，仍有效）；
    ///   也可外部传入(如点云级 σ)覆盖。
    ///   夹取到 [γFloor·scale, scale]：γ 过大 → softmin 退化为均值(梯度消失)；
    ///   γ 过小 → exp 下溢/退化为硬 min(数值不稳定)。
    /// </summary>
    private static double SoftMinDistance(Complex z, List<Complex> lattice,
                                          double gammaFraction = 0.1,
                                          double scale = -1.0)
    {
        int n = lattice.Count;
        double[] dists = new double[n];
        double minDist = double.MaxValue;
        for (int i = 0; i < n; i++)
        {
            double dist = (z - lattice[i]).Magnitude;
            dists[i] = dist;
            if (dist < minDist) minDist = dist;
        }

        // 尺度: 外部提供(点云级 σ) 或内部自适应(最近格点间隙 ≈ 格点间距)
        if (scale <= 0)
        {
            Array.Sort(dists);
            scale = (n > 1) ? Math.Max(dists[1] - dists[0], 1e-12) : 1e-12;
        }

        double gamma = Math.Max(1e-6 * scale,
                                Math.Min(gammaFraction * scale, scale));
        double sum = 0.0;
        for (int i = 0; i < n; i++)
            sum += Math.Exp(-(dists[i] - minDist) / gamma);
        return minDist - gamma * Math.Log(sum);
    }
    public static float[][] ComputeProbabilitiesFromTaus(Complex[] r30, Complex[] r45, Complex t1, Complex t2, float a, bool useSoftMin = false)
    {
        // 生成两个一维椭圆曲线格
        var lattice1 = GenerateLattice1D(t1, 20);
        var lattice2 = GenerateLattice1D(t2, 20);

        int n = r30.Length;
        float[] d1Arr = new float[n];
        float[] d2Arr = new float[n];

        // 1. 计算每个点到各自格的最近距离
        //    useSoftMin=true 时用自适应 γ 软最小距离 (连续可微, 缓解硬 min 的平台/梯度消失)
        for (int i = 0; i < n; i++)
        {
            double d1 = useSoftMin ? SoftMinDistance(r30[i], lattice1) : NearestDistance(r30[i], lattice1);
            double d2 = useSoftMin ? SoftMinDistance(r45[i], lattice2) : NearestDistance(r45[i], lattice2);
            d1Arr[i] = (float)d1;
            d2Arr[i] = (float)d2;
        }
        float[] sigm = new float[3];
        // 2. 分别估计 sigma1 和 sigma2（使用 RMS）
        float sigma1 = ComputeRMS(d1Arr);
        float sigma2 = ComputeRMS(d2Arr);
        sigm[1] = sigma1;
        sigm[2] = sigma2;
        // 3. 计算两个独立概率场
        float[] prob1 = new float[n]; // 第一个焦点对应的概率
        float[] prob2 = new float[n]; // 第二个焦点对应的概率

        for (int i = 0; i < n; i++)
        {
            prob1[i] = Mathf.Exp(-(d1Arr[i] * d1Arr[i]) / (2f * sigma1 * sigma1));
            prob2[i] = Mathf.Exp(-(d2Arr[i] * d2Arr[i]) / (2f * sigma2 * sigma2));
        }

        // 可选：整体概率（使用组合距离的 sigma），但通常分开使用即可
        sigm[0] = 2 * a;
        float[] distTotal = new float[n];
        for (int i = 0; i < n; i++)
            distTotal[i] = (d1Arr[i] + d2Arr[i] - sigm[0]);
        float[] probTotal = new float[n];
        for (int i = 0; i < n; i++)
            probTotal[i] = Mathf.Exp(-Math.Abs(distTotal[i]) / sigm[0]);

        // 返回三个概率数组
        return new float[][] { probTotal, prob1, prob2, sigm };
    }

    private static float ComputeRMS(float[] values)
    {
        float sumSq = 0f;
        foreach (float v in values) sumSq += v * v;
        return Mathf.Sqrt(sumSq / values.Length);
    }

    private static List<Complex> GenerateLattice1D(Complex tau, int range)
    {
        var lattice = new List<Complex>();
        for (int m = -range; m <= range; m++)
            for (int n = -range; n <= range; n++)
                lattice.Add(m + n * tau);
        return lattice;
    }

    private static double NearestDistance(Complex z, List<Complex> lattice)
    {
        double minSq = double.MaxValue;
        foreach (var lambda in lattice)
        {
            var diff = z - lambda;
            double sq = diff.Real * diff.Real + diff.Imaginary * diff.Imaginary;
            if (sq < minSq) minSq = sq;
        }
        return Math.Sqrt(minSq);
    }


}
public static class NelderMead
{
    public delegate double Objective(double[] x);

    /// <summary>最小化 f(x), 返回 (最优x, 最优值, 求值次数)。stopF: 达到该目标值即提前停止。</summary>
    public static (double[] bestX, double bestF, int evals) Minimize(
        Objective f, double[] x0, double[] lb, double[] ub,
        int maxEvals = 2000, double stopF = double.NegativeInfinity, double tol = 1e-6)
    {
        int n = x0.Length;
        var X = new double[n + 1][];
        var fx = new double[n + 1];

        // 初始单纯形
        for (int i = 0; i <= n; i++)
        {
            var xi = (double[])x0.Clone();
            if (i > 0)
            {
                double step = (xi[i - 1] == 0) ? 0.05 : 0.05 * Math.Abs(xi[i - 1]);
                xi[i - 1] += step;
            }
            for (int k = 0; k < n; k++) xi[k] = Clamp(xi[k], lb[k], ub[k]);
            X[i] = xi;
        }
        int evals = 0;
        for (int i = 0; i <= n; i++) { fx[i] = f(X[i]); evals++; }
        double[] bestX = (double[])X[0].Clone();
        double bestF = fx[0];

        while (evals < maxEvals)
        {
            // 按 fx 升序排序 (X 同步)
            for (int i = 0; i <= n; i++)
                for (int j = i + 1; j <= n; j++)
                    if (fx[j] < fx[i])
                    {
                        var tf = fx[i]; fx[i] = fx[j]; fx[j] = tf;
                        var tx = X[i]; X[i] = X[j]; X[j] = tx;
                    }

            if (fx[0] < bestF) { bestF = fx[0]; bestX = (double[])X[0].Clone(); }
            if (bestF <= stopF) break;

            // 收敛: 单纯形收缩
            double spread = 0;
            for (int k = 0; k < n; k++)
            {
                double m = 0;
                for (int i = 0; i < n; i++) m += X[i][k];
                m /= n;
                spread += (X[n][k] - m) * (X[n][k] - m);
            }
            if (spread < tol) break;

            // 质心 (去掉最差点 X[n])
            var centroid = new double[n];
            for (int k = 0; k < n; k++)
            {
                double m = 0;
                for (int i = 0; i < n; i++) m += X[i][k];
                centroid[k] = m / n;
            }

            // 反射
            var xr = new double[n];
            for (int k = 0; k < n; k++)
                xr[k] = Clamp(centroid[k] + 1.0 * (centroid[k] - X[n][k]), lb[k], ub[k]);
            double fr = f(xr);
            evals++;

            if (fr < fx[0])
            {
                // 扩张
                var xe = new double[n];
                for (int k = 0; k < n; k++)
                    xe[k] = Clamp(centroid[k] + 2.0 * (xr[k] - centroid[k]), lb[k], ub[k]);
                double fe = f(xe);
                evals++;
                if (fe < fr) { X[n] = xe; fx[n] = fe; }
                else { X[n] = xr; fx[n] = fr; }
            }
            else if (fr < fx[n - 1])
            {
                X[n] = xr;
                fx[n] = fr;
            }
            else
            {
                // 收缩
                var xc = new double[n];
                for (int k = 0; k < n; k++)
                    xc[k] = Clamp(centroid[k] + 0.5 * (X[n][k] - centroid[k]), lb[k], ub[k]);
                double fc = f(xc);
                evals++;
                if (fc < fx[n]) { X[n] = xc; fx[n] = fc; }
                else
                {
                    // 整体收缩到最优点
                    for (int i = 1; i <= n; i++)
                        for (int k = 0; k < n; k++)
                            X[i][k] = Clamp(X[0][k] + 0.5 * (X[i][k] - X[0][k]), lb[k], ub[k]);
                    for (int i = 1; i <= n; i++) { fx[i] = f(X[i]); evals++; }
                }
            }
        }
        return (bestX, bestF, evals);
    }

    private static double Clamp(double v, double lo, double hi)
    {
        return v < lo ? lo : (v > hi ? hi : v);
    }
}
