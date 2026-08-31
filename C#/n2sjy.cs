using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using UnityEngine;
using static alglib;
using Vector3 = UnityEngine.Vector3;

public class n2sjy : MonoBehaviour
{
    public Complex[] r45;
    public Complex[] r30;
    public Complex[] r16;
    public Complex[] r74;
    public float rp;
    public float a;
    public List<Vector3> points;
    void Start()
    {
        var random = new System.Random();
        List<float> fjValues = new List<float>();

        for (int trial = 0; trial < 20; trial++)
        {
            // 1. 生成随机点集（单位球体内）

            for (int i = 0; i < 200; i++)
            {
                Vector3 p = UnityEngine.Random.insideUnitSphere;
                points.Add(p);
            }

        }
        var (t10, t20) = nsjy.ComputeTaus();       // 理论初值（主参数）
        Vector3 axis = nsjy.pca(points).normalized;
        var (t1f, t2f, F1f, F2f) = RefineModuliByAxis(points, t10, t20, axis);
        Debug.Log($"{axis},{(F1f - F2f).normalized}");
        Debug.Log((t1f, t2f));
        Debug.Log($"[{string.Join(", ", points.Select(v => v.ToString("F6")))}]");
    }
    // ================= 全流程闭环：t1,t2 主参数 → 概率 → 焦点 → PCA 方向 =================
    public (Complex t1, Complex t2, Vector3 F1, Vector3 F2) RefineModuliByAxis(
        List<Vector3> pts, Complex t10, Complex t20, UnityEngine.Vector3 pcaAxis,
        int maxIter = 50,
        float angleTolDeg = 0.5f,
        float fdH = 1e-3f,        // 数值雅可比步长（按 τ 尺度）
        float wDir = 20f,         // 方向闭合项权重（硬约束，优先保证闭合）
        float wSelf = 1f,         // 概率自洽项权重
        float wTheory = 1e-3f)    // 理论正则权重（防止病态漂移）
    {
        int n = pts.Count;
        if (n < 4) throw new ArgumentException("至少 4 个点");

        // ---- 固定预处理（与 Start 一致）----
        rp = nsjy.ComputeRp(pts);
        float cone = Mathf.Asin(Mathf.Cos(30f * Mathf.Deg2Rad) / Mathf.PI); // ≈16°
        a = rp * (1f + Mathf.Sin(cone));
        r30 = new Complex[n]; r45 = new Complex[n];
        for (int i = 0; i < n; i++) { r30[i] = nsjy.InverseTh4(pts[i], 30,rp); r45[i] = nsjy.InverseTh4(pts[i], 45,rp); };
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
                angleDeg = Mathf.Acos(Mathf.Clamp(Vector3.Dot(dir, axis), -1f, 1f)) * Mathf.Rad2Deg;
            }
            if( iter == 3&&angleDeg> 30&& x[0] == t10.Real&& x[1] == t10.Imaginary)
            {
                Complex t3 = new Complex(0, 1);
                Complex t4 = new Complex(0,1);
                 var (r1, F11, F22) = EvaluateResiduals(x, pts, axis, wDir, wSelf, wTheory, t3, t4);
                Vector3 dir0 = F11 - F22;
                float angleDeg2 = Mathf.Acos(Mathf.Clamp(Vector3.Dot(dir0, axis), -1f, 1f)) * Mathf.Rad2Deg;
                if( angleDeg2 < angleDeg)
                {
                    x[0] = 0;
                    x[1] = 1;
                    x[2] = 0;
                    x[3] = 1;
                }
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
    private (double[] r, Vector3 F1, Vector3 F2) EvaluateResiduals(
        double[] x, List<Vector3> pts, Vector3 axis,
        float wDir, float wSelf, float wTheory, Complex t10, Complex t20)
    {
        int n = pts.Count;
        Complex t1 = new Complex(x[0], x[1]);
        Complex t2 = new Complex(x[2], x[3]);

        // 主链路：t ─► 格概率 ─► 焦点
        float[][] probs = DeviationCalculator.ComputeProbabilitiesFromTaus(r30, r45, t1, t2, a);
        var (F1, F2) =nsjy.ExtractFoci(pts, probs[1], probs[3][1], probs[2], probs[3][2]);

        // 修正视角：从焦点反推概率（注意括号！）


        double[] r = new double[n + 7];
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

        // 闭合条件：(F1−F2) 平行于 PCA 主轴
        Vector3 dir = F1 - F2;
        if (dir.sqrMagnitude < 1e-12f) dir = axis;   // 焦点退化保护
        dir.Normalize();
        if (Vector3.Dot(dir, axis) < 0) dir = -dir;
        Vector3 e = dir - axis;
        r[n] = wDir * e.x;
        r[n + 1] = wDir * e.y;
        r[n + 2] = wDir * e.z;

        // 理论正则（弱）
        r[n + 3] = wTheory * (x[0] - t10.Real);
        r[n + 4] = wTheory * (x[1] - t10.Imaginary);
        r[n + 5] = wTheory * (x[2] - t20.Real);
        r[n + 6] = wTheory * (x[3] - t20.Imaginary);

        return (r, F1, F2);
    }

    /// <summary>τ 归一化到基本域：Im>0、|Re|≤0.5、|τ|≥1（消除模群冗余）</summary>
    private static void NormalizeTau(double[] x)
    {
        for (int k = 0; k < 4; k += 2)
        {
            double re = x[k];
            double im = x[k + 1];

            // 确保虚部为正且不过小
            if (im <= 1e-4) im = 1e-4;

            // 迭代应用模群变换，直到进入基本域
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
