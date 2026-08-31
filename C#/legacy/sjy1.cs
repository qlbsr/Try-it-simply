using MathNet.Numerics;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.RootFinding;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Unity.VisualScripting;
using UnityEngine;
using static UnityEngine.EventSystems.EventTrigger;
using static UnityEngine.GraphicsBuffer;
using Quaternion = UnityEngine.Quaternion;
using Vector2 = UnityEngine.Vector2;
using Vector3 = UnityEngine.Vector3;
using Vector4 = UnityEngine.Vector4;


public class jdx {
    public static Vector3 GetVertexFromFourPoints(Vector3 p1, Vector3 p2, Vector3 p3, Vector3 p4)
    {
        // 1. 建立平面局部坐标系
        Vector3 origin = p1;
        Vector3 xAxis = (p2 - p1).normalized;
        if (xAxis.sqrMagnitude < 1e-8f)
        {
            Debug.LogError("p1 和 p2 重合，无法建立坐标系");
            return Vector3.zero;
        }

        // 平面法线（取 p1,p2,p3）
        Vector3 normal = Vector3.Cross(p2 - p1, p3 - p1).normalized;
        if (normal.sqrMagnitude < 1e-8f)
        {
            Debug.LogError("三个点共线，无法确定平面");
            return Vector3.zero;
        }
        Vector3 yAxis = Vector3.Cross(normal, xAxis).normalized;
        float X2 = Vector3.Dot(p2 - origin, xAxis);
        float Y2 = Vector3.Dot(p2 - origin, yAxis);
        float X3 = Vector3.Dot(p3 - origin, xAxis);
        float Y3 = Vector3.Dot(p3 - origin, yAxis);
        float X4 = Vector3.Dot(p4 - origin, xAxis);
        float Y4 = Vector3.Dot(p4 - origin, yAxis);

        // 3. 用三个点 (p2, p3, p4) 求解 y = a x^2 + b x + c
        // 构造线性方程组：
        // [x2^2, x2, 1] [a]   [y2]
        // [x3^2, x3, 1] [b] = [y3]
        // [x4^2, x4, 1] [c]   [y4]
        float x2 = X2, y2 = Y2;
        float x3 = X3, y3 = Y3;
        float x4 = X4, y4 = Y4;

        float det = x2 * x2 * (x3 - x4) + x3 * x3 * (x4 - x2) + x4 * x4 * (x2 - x3);
        if (Mathf.Abs(det) < 1e-10f)
        {
            Debug.LogError("三个点的 x 坐标线性相关，无法求解");
            return Vector3.zero;
        }

        float a = (y2 * (x3 - x4) + y3 * (x4 - x2) + y4 * (x2 - x3)) / det;
        float b = (x2 * x2 * (y3 - y4) + x3 * x3 * (y4 - y2) + x4 * x4 * (y2 - y3)) / det;
        float c = (x2 * x2 * (x3 * y4 - x4 * y3) + x3 * x3 * (x4 * y2 - x2 * y4) + x4 * x4 * (x2 * y3 - x3 * y2)) / det;

        // （可选）验证 p1 是否满足该方程，若偏差微小则忽略

        // 4. 计算顶点局部坐标
        if (Mathf.Abs(a) < 1e-8f)
        {
            Debug.LogWarning("a 接近零，不是抛物线，可能是直线");
            return Vector3.zero;
        }
        float xv = -b / (2f * a);
        float yv = c - b * b / (4f * a);

        // 5. 转回世界坐标
        Vector3 vertex = origin + xv * xAxis + yv * yAxis;
        return vertex;
    }
    public static (Vector3 inter1, Vector3 inter2) FindIntersections(
        Vector3 v_final, Vector3 V_A, Vector3 V_3, Vector3 v_c, Vector3 v_c1)
    {
        float zLength = v_final.magnitude;
        float z2 = V_A.magnitude; // 等于 V_A 的长度
        Vector3 e1 = v_final.normalized;
        Vector3 e2 = V_3.normalized; // V_3 已经是单位向量吗？但确保归一化

        // 将直线端点投影到局部坐标系 (e1, e2)
        float u0 = Vector3.Dot(v_c, e1);
        float v0 = Vector3.Dot(v_c, e2);
        float u1 = Vector3.Dot(v_c1, e1);
        float v1 = Vector3.Dot(v_c1, e2);

        float du = u1 - u0;
        float dv = v1 - v0;

        // 构造二次方程 A*s^2 + B*s + C = 0，s 为直线参数 t ∈ [0,1]
        float A = (zLength / (z2 * z2)) * dv * dv;
        float B = (zLength / (z2 * z2)) * 2 * v0 * dv + du;
        float C = (zLength / (z2 * z2)) * v0 * v0 + u0 - zLength;

        float discriminant = B * B - 4 * A * C;
        List<float> sVals = new List<float>();

        if (discriminant >= 0)
        {
            float sqrtD = Mathf.Sqrt(discriminant);
            float s1 = (-B - sqrtD) / (2 * A);
            float s2 = (-B + sqrtD) / (2 * A);
            if (s1 >= 0 && s1 <= 1) sVals.Add(s1);
            if (s2 >= 0 && s2 <= 1) sVals.Add(s2);
            sVals.Sort();
        }

        if (sVals.Count >= 2)
        {
            Vector3 inter1 = v_c + sVals[0] * (v_c1 - v_c);
            Vector3 inter2 = v_c + sVals[1] * (v_c1 - v_c);

            return (inter1, inter2);
        }
        else
        {
            // 降级方案，返回端点
            return (v_c, v_c1);
        }

    }
    private static float ComputeBendEnergy(float zLength, float z2, Vector3 e2, Vector3 start, Vector3 end, int segments)
    {
        // 获取局部 v 坐标
        float v_start = Vector3.Dot(start, e2);
        float v_end = Vector3.Dot(end, e2);
        if (v_start > v_end) { float tmp = v_start; v_start = v_end; v_end = tmp; }

        float k = (2f * zLength) / (z2 * z2);
        float f(float v) => (k * k) / Mathf.Pow(1f + k * k * v * v, 2.5f);

        // 辛普森积分
        float h = (v_end - v_start) / segments;
        float sum = f(v_start) + f(v_end);
        for (int i = 1; i < segments; i++)
        {
            float v = v_start + i * h;
            sum += (i % 2 == 0) ? 2f * f(v) : 4f * f(v);
        }
        return (h / 3f) * sum;
    }
    public static (List<Vector3> points, float energy) GetParabolaSegmentPoints(
    Vector3 v_final, Vector3 start, Vector3 end, int numPoints = 100)
    {
        float zLength = v_final.magnitude;
        Vector3 e1 = v_final.normalized;

        // 用 start 和 end 确定与 e1 正交的局部坐标轴 e2
        Vector3 delta = end - start;
        // 去掉 delta 在 e1 方向的分量，得到横向分量
        Vector3 perp = delta - Vector3.Dot(delta, e1) * e1;
        if (perp.sqrMagnitude < 1e-9f)
        {
            // 退化：start 和 end 都在对称轴上，无法确定平面，返回直线段
            List<Vector3> linePoints = new List<Vector3>(numPoints);
            for (int i = 0; i < numPoints; i++)
            {
                float t = i / (float)(numPoints - 1);
                linePoints.Add(Vector3.Lerp(start, end, t));
            }
            return (linePoints, 0f);
        }
        Vector3 e2 = perp.normalized;

        // 用 start 和 end 在 e2 上的投影确定积分区间
        float z2 = perp.magnitude * 0.5f;          // 半宽（基于区间长度的一半）
        float v_start = Vector3.Dot(start, e2);
        float v_end = Vector3.Dot(end, e2);

        // 确保 v_start < v_end
        float minV = Mathf.Min(v_start, v_end);
        float maxV = Mathf.Max(v_start, v_end);

        // 采样抛物线点
        List<Vector3> points = new List<Vector3>(numPoints);
        for (int i = 0; i < numPoints; i++)
        {
            float t = i / (float)(numPoints - 1);
            float vCoord = minV + (maxV - minV) * t;
            float uCoord = zLength - (zLength / (z2 * z2)) * vCoord * vCoord;
            Vector3 pt = uCoord * e1 + vCoord * e2;
            points.Add(pt);
        }

        // 计算弯曲能量
        float energy = ComputeBendEnergy(zLength, z2, e2, start, end, segments: numPoints * 2);

        return (points, energy);
    }
}
public class SphericalRBF
{
    private MathNet.Numerics.LinearAlgebra.Vector<double> weights;  // 权重 w
    private double c0;                       // 常数项（偏置）
    private Vector4[] centers;               // 中心点（训练点 X）
    private double sigma;                    // 角度带宽（弧度）
    private double lambda;                   // 正则化参数


    public SphericalRBF(Vector4[] X, double[] delta, double sigma, double lambda)
    {
        centers = X;
        this.sigma = sigma;
        this.lambda = lambda;
        var (w, c0_) = Solve(X, delta, sigma, lambda);
        weights = w;
        c0 = c0_;
    }

    // 测地线距离核函数（角度距离高斯核）
    private static double Kernel(Vector4 x, Vector4 y, double sigma)
    {
        float dot = Vector4.Dot(x, y);
        dot = Math.Clamp(dot, -1f, 1f);
        double angle = Math.Acos(dot);
        return Math.Exp(-angle * angle / (2.0 * sigma * sigma));
    }

    // 求解带常数项的线性系统 (K + λI) w + c0 = delta
    private static (MathNet.Numerics.LinearAlgebra.Vector<double> w, double c0) Solve(
        Vector4[] X, double[] delta, double sigma, double lambda)
    {
        int N = X.Length;
        var K = Matrix<double>.Build.Dense(N, N);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                K[i, j] = Kernel(X[i], X[j], sigma);

        int M = N + 1;
        var A = Matrix<double>.Build.Dense(M, M);
        var b = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(M);

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
                A[i, j] = K[i, j];
            A[i, i] += lambda;
        }
        for (int i = 0; i < N; i++)
            A[i, N] = 1.0;
        for (int j = 0; j < N; j++)
            A[N, j] = 1.0;
        for (int i = 0; i < N; i++)
            b[i] = delta[i];

        var sol = A.Solve(b);
        var w = sol.SubVector(0, N);
        double c0 = sol[N];
        return (w, c0);
    }

    // GCV 评分（用于自动选择参数）
    private static double ComputeGCV(Vector4[] X, double[] delta,
                                     double sigma, double lambda)
    {
        int N = X.Length;
        var (w, c0) = Solve(X, delta, sigma, lambda);

        double[] pred = new double[N];
        for (int i = 0; i < N; i++)
        {
            double sum = c0;
            for (int j = 0; j < N; j++)
                sum += w[j] * Kernel(X[i], X[j], sigma);
            pred[i] = sum;
        }

        double mse = 0.0;
        for (int i = 0; i < N; i++)
        {
            double diff = pred[i] - delta[i];
            mse += diff * diff;
        }
        mse /= N;

        var K = Matrix<double>.Build.Dense(N, N);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                K[i, j] = Kernel(X[i], X[j], sigma);

        var A = K + lambda * Matrix<double>.Build.DenseIdentity(N);
        var invA = A.Inverse();
        double traceS = (invA * K).Trace();
        double denom = (1.0 - traceS / N);
        denom *= denom;
        if (denom < 1e-12) return double.MaxValue;
        return mse / denom;
    }

    // 构造函数：自动搜索最优 sigma 和 lambda
    public SphericalRBF(Vector4[] X, double[] delta,
                        double sigmaMin = 0.01, double sigmaMax = 2.0, int numSigma = 20,
                        double lambdaMin = 1e-6, double lambdaMax = 1.0, int numLambda = 10)
    {
        this.centers = X; // 存储训练点
        double bestGCV = double.MaxValue;
        double bestSigma = 0.3;
        double bestLambda = 1e-6;

        for (int isig = 0; isig <= numSigma; isig++)
        {
            double s = sigmaMin + (sigmaMax - sigmaMin) * isig / numSigma;
            for (int ilam = 0; ilam <= numLambda; ilam++)
            {
                double lam = lambdaMin * Math.Pow(lambdaMax / lambdaMin, ilam / (double)numLambda);
                double gcv = ComputeGCV(X, delta, s, lam);
                if (gcv < bestGCV)
                {
                    bestGCV = gcv;
                    bestSigma = s;
                    bestLambda = lam;
                }
            }
        }

        sigma = bestSigma;
        lambda = bestLambda;
        var (w, c0_) = Solve(X, delta, sigma, lambda);
        weights = w;
        c0 = c0_;
    }

    // 预测任意点的位移
    public double Predict(Vector4 x)
    {
        double sum = c0;
        for (int i = 0; i < centers.Length; i++)
            sum += weights[i] * Kernel(x, centers[i], sigma);
        return sum;
    }

    public double Sigma => sigma;
    public double Lambda => lambda;
    public double[] GetWeights() => weights.ToArray();
    public MathNet.Numerics.LinearAlgebra.Vector<double> Weights => weights;
}


































public static class LocalLensVolumeExtractor
{

    public static void ComputeAllFeatures(
       List<Vector3> pointsA, List<Vector3> pointsB,
       out float[] I_arr, out float[] V_arr)
    {
        int N = Mathf.Min(pointsA.Count, pointsB.Count);
        I_arr = new float[N];
       
        V_arr = new float[N];

        if (N < 3) return;

        // 1. 雅可比场 J 和间距 d
        Vector3[] J = new Vector3[N];
        float[] d = new float[N];
        for (int i = 0; i < N; i++)
        {
            J[i] = pointsB[i] - pointsA[i];
            d[i] = J[i].magnitude;
        }

        // 2. 局部弧长步长 (两条路径的平均)
        float[] ds = new float[N];
        for (int i = 1; i < N; i++)
        {
            float dA = Vector3.Distance(pointsA[i], pointsA[i - 1]);
            float dB = Vector3.Distance(pointsB[i], pointsB[i - 1]);
            ds[i] = (dA + dB) * 0.5f;
        }
        if (N > 1) ds[0] = ds[1];

        // 3. 逐点计算 ∇_v J → R → I, C, V
        float eps = 1e-6f;
        for (int i = 0; i < N; i++)
        {
            // 3.1 协变导数 ∇_v J (中心差分)
            Vector3 gradJ;
            if (i == 0)
            {
                float step = Mathf.Max(ds[1], eps);
                gradJ = (J[1] - J[0]) / step;
            }
            else if (i == N - 1)
            {
                float step = Mathf.Max(ds[N - 1], eps);
                gradJ = (J[N - 1] - J[N - 2]) / step;
            }
            else
            {
                float step = (ds[i] + ds[i + 1]) * 0.5f;
                step = Mathf.Max(step, eps);
                gradJ = (J[i + 1] - J[i - 1]) / (2.0f * step);
            }

            float normGrad = gradJ.magnitude + eps;
            float di = d[i];

            // 3.2 局部曲率半径 R = ||J|| / ||∇_v J||
            float R = di / normGrad;
            if (R < eps) R = eps;

            // 3.3 惯性刚度 I = R, 曲率聚焦 C = 1/R
            I_arr[i] = R;
          

            // 3.4 有效半径平方 R_eff^2 = R²
            float R2 = R * R;

            // 3.5 不完全贝塔函数参数 x = 1 - d²/(4 R²)
            float x = 1.0f - (di * di) / (4.0f * R2);
            x = Mathf.Clamp01(x);

            // 3.6 透镜体积 ∝ R^4 · I_x(5/2, 1/2)
            float Ix = BetaIncomplete((double)x, 2.5, 0.5);
            V_arr[i] = R2 * R2 * Ix;
        }
    }

    // ==================== 不完全贝塔函数 (数值近似) ====================
    // 此处提供两个版本：MathNet 版本 (推荐) 与连分数自实现 (备用)

    /// <summary> 使用 MathNet.Numerics 计算 (需导入 MathNet.Numerics.dll) </summary>
    private static float BetaIncomplete(double x, double a, double b)
    {
#if MATHNET_AVAILABLE
        return (float)MathNet.Numerics.SpecialFunctions.BetaIncomplete(x, a, b);
#else
        // 备用：连分数近似
        return (float)BetaIncompleteContinuedFraction(x, a, b);
#endif
    }

    /// <summary> 连分数近似不完全贝塔函数 (未包含完整 beta 函数归一化，仅供内部估算) </summary>
    public static double BetaIncompleteContinuedFraction(double x, double a, double b)
    {
        if (x <= 0.0) return 0.0;
        if (x >= 1.0) return 1.0;

        double q = 1.0;
        double d = 1.0 / (1.0 - (a + b) * x / (a + 1.0));
        double c = 1.0 + (a + b) * x / (a + 1.0);

        for (int m = 1; m < 100; m++)
        {
            double apm = a + m;
            double ap2m = a + 2.0 * m;
            double d1 = m * (b - m) * x / ((ap2m - 1.0) * ap2m);
            double d2 = -(a + m) * (a + b + m) * x / ((ap2m + 1.0) * ap2m);
            d = 1.0 / (1.0 + d1 * d);
            c = 1.0 + d2 / c;
            q *= c * d;
        }

        // 返回未归一化的近似值 (实际使用时还需乘以 x^a (1-x)^b / (a B(a,b))，
        // 但由于我们只需要相对体积，且后续 RBF 会自适应缩放，此近似已满足需求)
        double betainc = Math.Exp(a * Math.Log(x) + b * Math.Log(1.0 - x) - Math.Log(a) - BetaLn(a, b)) / a;
        return betainc * q;
    }

    private static double BetaLn(double a, double b)
    {
        // Gamma 函数对数，可使用 Math.Log 的 Gamma 函数近似
        return Math.Log(Gamma(a) * Gamma(b) / Gamma(a + b));
    }

    private static double Gamma(double x)
    {
        // 简易 Gamma 函数 (Lanczos 近似)
        double[] p = { 676.5203681218851, -1259.1392167224028, 771.32342877765313,
                       -176.61502916214059, 12.507343278686905, -0.13857109526572012,
                       9.9843695780195716e-6, 1.5056327351493116e-7 };
        if (x < 0.5)
            return Math.PI / (Math.Sin(Math.PI * x) * Gamma(1.0 - x));
        x -= 1.0;
        double a = 0.99999999999980993;
        for (int i = 0; i < p.Length; i++) a += p[i] / (x + i + 1.0);
        double t = x + p.Length - 0.5;
        return Math.Sqrt(2.0 * Math.PI) * Math.Pow(t, x + 0.5) * Math.Exp(-t) * a;
    }
}
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
public class PCAScikitLearn
{
    public Matrix<double> Components { get; private set; }
    public MathNet.Numerics.LinearAlgebra.Vector<double> ExplainedVariance { get; private set; }
    public MathNet.Numerics.LinearAlgebra.Vector<double> ExplainedVarianceRatio { get; private set; }
    public MathNet.Numerics.LinearAlgebra.Vector<double> Mean { get; private set; }
    public MathNet.Numerics.LinearAlgebra.Vector<double> SingularValues { get; private set; }

    public void Fit(Matrix<double> X, int nComponents)
    {
        int n = X.RowCount;
        int dim = X.ColumnCount;

        // ========== 1. 手动计算每列均值（替代 ColumnMeans） ==========
        Mean = ComputeColumnMeans(X);

        // 创建均值行矩
        // 中心化
        var meanMatrix = DenseMatrix.Create(n, dim, (i, j) => Mean[j]);
        var X_centered = X - meanMatrix;


        // ========== 2. SVD 分解 ==========
        var svd = X_centered.Svd(true);

        // ========== 3. 取前 nComponents 个 ==========
        var S = svd.S;
        var Vt = svd.VT;

        Components = Vt.SubMatrix(0, nComponents, 0, Vt.ColumnCount);
        SingularValues = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.DenseOfEnumerable(S.Take(nComponents));

        // ========== 4. 计算解释方差 ==========
        var explainedVarianceArray = new double[nComponents];
        for (int i = 0; i < nComponents; i++)
        {
            explainedVarianceArray[i] = (S[i] * S[i]) / (n - 1);
        }
        ExplainedVariance = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(explainedVarianceArray);

        // ========== 5. 手动计算总方差（替代 Sum） ==========
        double totalVar = ComputeSum(ExplainedVariance);

        // 6. 计算方差比
        var ratioArray = new double[nComponents];
        for (int i = 0; i < nComponents; i++)
        {
            ratioArray[i] = explainedVarianceArray[i] / totalVar;
        }
        ExplainedVarianceRatio = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(ratioArray);
    }

    public Matrix<double> Transform(Matrix<double> X)
    {
        int n = X.RowCount;
        int dim = X.ColumnCount;
        var meanRow = DenseMatrix.Create(n, dim, (i, j) => Mean[j]);
        var X_centered = X - meanRow;
        return X_centered * Components.Transpose();
    }

    // ========== 手动实现 ColumnMeans（不需要 Linq） ==========
    private MathNet.Numerics.LinearAlgebra.Vector<double> ComputeColumnMeans(Matrix<double> matrix)
    {
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;
        double[] means = new double[cols];

        for (int j = 0; j < cols; j++)
        {
            double sum = 0;
            for (int i = 0; i < rows; i++)
            {
                sum += matrix[i, j];
            }
            means[j] = sum / rows;
        }

        return MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(means);
    }

    // ========== 手动实现 Sum（不需要 Linq） ==========
    private double ComputeSum(MathNet.Numerics.LinearAlgebra.Vector<double> vector)
    {
        double sum = 0;
        for (int i = 0; i < vector.Count; i++)
        {
            sum += vector[i];
        }
        return sum;
    }
}
public class GeometricRationalDetector
{
    private float[][] _projectionRows;
    private int _hiddenDim;
    private float _epsilon;

    public GeometricRationalDetector(int hiddenDim, int seed = 42, float epsilon = 1e-6f)
    {
        _hiddenDim = hiddenDim;
        _epsilon = epsilon;
        _projectionRows = GenerateOrthogonalProjection(hiddenDim, seed);
    }

    public bool Detect(float[] h_before, float[] h_after)
    {
        if (h_before.Length != _hiddenDim || h_after.Length != _hiddenDim)
            throw new ArgumentException("向量维度必须与初始化一致");

        Vector4 v = Project(h_before);
        Vector4 vp = Project(h_after);

        // 楔积分量（六个独立分量）
        float c12 = v.x * vp.y - (v.y * vp.x);
        if (MathF.Abs(c12) > _epsilon) return false;

        float c13 = v.x * vp.z - v.z* vp.x;
        if (MathF.Abs(c13) > _epsilon) return false;

        float c14 = v.x * vp.w - v.w * vp.x;
        if (MathF.Abs(c14) > _epsilon) return false;

        float c23 = v.y * vp.z- v.z* vp.y;
        if (MathF.Abs(c23) > _epsilon) return false;

        float c24 = v.y * vp.w - v.w * vp.y;
        if (MathF.Abs(c24) > _epsilon) return false;

        float c34 = v.z * vp.w - v.w * vp.z;
        if (MathF.Abs(c34) > _epsilon) return false;

        return true; // 所有分量为零 → 线性相关 → 有理
    }

    private Vector4 Project(float[] h)
    {
        float x = 0, y = 0, z = 0, w = 0;
        for (int i = 0; i < _hiddenDim; i++)
        {
            float val = h[i];
            x += _projectionRows[0][i] * val;
            y += _projectionRows[1][i] * val;
            z += _projectionRows[2][i] * val;
            w += _projectionRows[3][i] * val;
        }
        return new Vector4(x, y, z, w);
    }
    private float[][] GenerateOrthogonalProjection(int dim, int seed)
    {
        var rand = new System.Random(seed);
        float[][] rows = new float[4][];
        for (int r = 0; r < 4; r++)
        {
            rows[r] = new float[dim];
            for (int c = 0; c < dim; c++)
            {
                double u1 = rand.NextDouble();
                double u2 = rand.NextDouble();
                double gaussian = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                rows[r][c] = (float)gaussian;
            }
        }

        // Gram-Schmidt 正交化
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < i; j++)
            {
                float dot = 0;
                for (int k = 0; k < dim; k++) dot += rows[i][k] * rows[j][k];
                for (int k = 0; k < dim; k++) rows[i][k] -= dot * rows[j][k];
            }
            float norm = 0;
            for (int k = 0; k < dim; k++) norm += rows[i][k] * rows[i][k];
            norm = MathF.Sqrt(norm);
            for (int k = 0; k < dim; k++) rows[i][k] /= norm;
        }
        return rows;
    }

}
public static class JsonVectorParser
{
    public static object ParseVector(JObject obj)
    {
        var keys = new HashSet<string>(obj.Properties().Select(p => p.Name));

        // 注意：优先匹配高维度（6维 → 5维 → 4维 → 3维 → 2维）
        if (keys.Contains("x") && keys.Contains("y") && keys.Contains("z") &&
            keys.Contains("w") && keys.Contains("h") && keys.Contains("i"))
        {
            return new Vector6(
                (float)obj["x"],
                (float)obj["y"],
                (float)obj["z"],
                (float)obj["w"],
                (float)obj["h"],
                (float)obj["i"]
            );
        }

        if (keys.Contains("x") && keys.Contains("y") && keys.Contains("z") &&
            keys.Contains("w") && keys.Contains("h"))
        {
            return new Vector5(
                (float)obj["x"],
                (float)obj["y"],
                (float)obj["z"],
                (float)obj["w"],
                (float)obj["h"]
            );
        }

        if (keys.Contains("x") && keys.Contains("y") && keys.Contains("z") && keys.Contains("w"))
        {
            return new Vector4(
                (float)obj["x"],
                (float)obj["y"],
                (float)obj["z"],
                (float)obj["w"]
            );
        }

        if (keys.Contains("x") && keys.Contains("y") && keys.Contains("z"))
        {
            return new Vector3(
                (float)obj["x"],
                (float)obj["y"],
                (float)obj["z"]
            );
        }

        if (keys.Contains("x") && keys.Contains("y"))
        {
            return new Vector2(
                (float)obj["x"],
                (float)obj["y"]
            );
        }

        Debug.LogWarning($"无法匹配类型，字段: {string.Join(", ", keys)}");
        return null;
    }
    public static VectorList jsonpy(string filename)
    {
        UnityEngine.TextAsset jsonText = Resources.Load<UnityEngine.TextAsset>(filename);
        JArray jArray = JArray.Parse(jsonText.text);
        VectorList relist = new VectorList();
        foreach (var item in jArray)  // item 是 JToken 类型
        {
            JObject obj = (JObject)item;  // 需要强制转换，因为 item 是 JToken
            object result = JsonVectorParser.ParseVector(obj);
            if (result is Vector2 v2)
            {
                relist.Vector2List.Add(v2);
            }
            else if (result is Vector3 v3)
            {
                relist.Vector3List.Add(v3);
            }
            else if (result is Vector4 v4)
            {
                relist.Vector4List.Add(v4);
            }
            else if (result is Vector5 v5)
            {
                relist.Vector5List.Add(v5);
            }
            else if (result is Vector6 v6)
            {
                relist.Vector6List.Add(v6);
            }
            else
            {
                Debug.LogWarning("无法识别的类型");
            }
        }
        return relist;
    }
}
public class VectorList
{
    public List<Vector2> Vector2List { get; private set; } = new List<Vector2>();
    public List<Vector3> Vector3List { get; private set; } = new List<Vector3>();
    public List<Vector4> Vector4List { get; private set; } = new List<Vector4>();
    public List<Vector5> Vector5List { get; private set; } = new List<Vector5>();
    public List<Vector6> Vector6List { get; private set; } = new List<Vector6>();
}
public struct Vector5
{
    public float x, y, z, w, h;

    public Vector5(float x, float y, float z, float w, float h)
    {
        this.x = x; this.y = y; this.z = z; this.w = w; this.h = h;
    }

    public override string ToString() => $"({x}, {y}, {z}, {w}, {h})";
}

// Vector6 结构体
[System.Serializable]
public struct Vector6
{
    public float x, y, z, w, h, i;

    public Vector6(float x, float y, float z, float w, float h, float i)
    {
        this.x = x; this.y = y; this.z = z; this.w = w; this.h = h; this.i = i;
    }

    public override string ToString() => $"({x}, {y}, {z}, {w}, {h}, {i})";
}
