using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Vector2 = UnityEngine.Vector2;
using Vector3 = UnityEngine.Vector3;



// ---------- test 类完整流程 ----------
class test : MonoBehaviour
{
    public sjy sjy11;

    [Range(0f, 1f)]
    public float perturbation = 1e-6f;
    void Start()
    {
        // 1. 读取点云
        VectorList vectorList = JsonVectorParser.jsonpy("points2");
        List<Vector3> pointCloud = vectorList.Vector3List;

        if (true)
        {
            var rng = new System.Random(42);
            for (int i = 0; i < pointCloud.Count; i++)
            {
                float dx = ((float)rng.NextDouble() - 0.5f) * 2 * perturbation;
                float dy = ((float)rng.NextDouble() - 0.5f) * 2 * perturbation;
                float dz = ((float)rng.NextDouble() - 0.5f) * 2 * perturbation;
                pointCloud[i] = new Vector3(
                    pointCloud[i].x + dx,
                    pointCloud[i].y + dy,
                    pointCloud[i].z + dz);
            }


        }




        if (pointCloud == null || pointCloud.Count == 0)
        {
            Debug.LogError("点云为空");
            return;
        }

        // 2. 初始化几何处理器
        sjy11 = new sjy();
        sjy11.vector3s = pointCloud;

        // 3. 计算全局半顶角 d2 

        sjy11.jh(1);
       
        sjy11.dn();

        sjy11.sxxx();
        // 此时 sjy.rp, sjy.v3 已被设置
        // 5. 取第一组点集（圆上交点）作为基准点
        List<Vector3> points = new List<Vector3>(sjy11.point2[0]);
        if (points == null || points.Count == 0)
        {
            Debug.LogError("基准点集为空");
            return;
        }

        // 6. 计算椭球吻合概率
        float[] probs = sjy11.ComputeProbabilities(points);
        if (probs == null || probs.Length == 0)
        {
            Debug.LogError("概率数组为空");
            return;
        }

        // 7. 生成叶平面映射，填充 uvss 列表
        Vector2[] uvs = sjy11.MapDoubleConeToLeaf(points);

     
     






    }

    










}
