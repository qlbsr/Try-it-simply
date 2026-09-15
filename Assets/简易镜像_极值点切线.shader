Shader "Custom/简易镜像_极值点切线"
{
    Properties
    {
        _Color ("Color", Color) = (1,1,1,1)
        _MainTex ("Albedo (RGB)", 2D) = "white" {}
        _Glossiness ("Smoothness", Range(0,1)) = 0.5
        _Metallic ("Metallic", Range(0,1)) = 0.0
        _LineWidth ("Line Width", Range(0.001, 0.1)) = 0.02
        _LineColor ("Line Color", Color) = (1,0,0,1)
        _CurveHeight ("Curve Height", Range(0, 1)) = 0.5
        _LineColor2 ("Line Color2", Color) = (0,1,0,1)
        _CurveHeight2 ("Curve Height2", Range(0, 1)) = 0.5
    }
    
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 200

        CGPROGRAM
        #pragma surface surf Standard fullforwardshadows
        #pragma target 3.0

        sampler2D _MainTex;

        struct Input
        {
            float2 uv_MainTex;
        };

        half _Glossiness;
        half _Metallic;
        fixed4 _Color;
        float _LineWidth;
        fixed4 _LineColor;
        fixed4 _LineColor2;
        float _CurveHeight;
        float _CurveHeight2;
        int one = 0;
        int two = 0;
        // 计算点到贝塞尔曲线的最短距离
        float DistanceToBezier(float2 p, float2 P0, float2 P1, float2 P2,out float2 pp1,out float2 pp2,float2 pp,bool me)
        {
            float minDist = 1000.0;
            for (int i = 0; i <= 100; i++)
            {
                if ( me == true)
                {
                    if (i >= one && i <= two && one > 0 && two > 0)
                    continue;  // 跳过，不参与渲染
                }
                float t = (float)i / 100.0;
                float u = 1.0 - t;
                float2 pt = u * u * P0 + 2.0 * u * t * P1 + t * t * P2;
                float epsilon = 0.01;

                if (distance(pp, pp1) > epsilon)  // pp != pp1
                {
                    if (abs(pp.y - pt.y) < epsilon)  // pp.y == pt.y
                    {
                        if (distance(pp1, pp2) < epsilon)  // pp1 == pp2
                        {
                            pp1 = pt;
                            one =i;
                        }
                        else
                        {
                            pp2 = pt;
                            two =i;
                        }
                    }
                }
                float d = length(p - pt);
                if (d < minDist) minDist = d;  
             }
             return minDist;
        }
        void surf (Input IN, inout SurfaceOutputStandard o)
        {
            fixed4 c = tex2D(_MainTex, IN.uv_MainTex) * _Color;
            o.Albedo = c.rgb;
            o.Metallic = _Metallic;
            o.Smoothness = _Glossiness;
            o.Alpha = c.a;
            
            float2 uv = IN.uv_MainTex;
            float j1 = _CurveHeight;
            
            // ===== 第一条曲线：下左到下右 =====
            float2 P0 = float2(0.0, 0.0);
            float2 P1 = float2(0.5, 1.0 - j1);
            float2 P2 = float2(1.0, 0.0);
            
            // ===== 第二条曲线：上左到上右 =====
            float2 P01 = float2(0.0, 1.0);
            float2 P11 = float2(0.5, 0.0 + j1);
            float2 P21 = float2(1.0, 1.0);
            float2 pp1 = float2(0.5,1.0 - _CurveHeight2);
            float2 pp11 = float2(0.5, 0.0 + _CurveHeight2);
            float2 a1 = float2(0, 0);
            float2 a2 = float2(0, 0);
            float2 a3 = float2(0, 0);
            float2 a4 = float2(0, 0);
            // 绘制曲线
            float minDist0 = DistanceToBezier(uv, P0, P1, P2, a1,a2,pp1,false);
            float minDist1 = DistanceToBezier(uv, P0, P1, P2, a1,a2,pp1,true);
            float minDist2 = DistanceToBezier(uv, P01, P11, P21,a3,a4,pp11,true);
            float minDist3 = DistanceToBezier(uv, P01, P11, P21,a3,a4,pp11,false);
            // 1. 先画绿色打底（全部）
            if (minDist0 < _LineWidth || minDist3 < _LineWidth)
            {
                float d = min(minDist0, minDist3);
                float s = 1.0 - smoothstep(0, _LineWidth, d);
                o.Albedo = lerp(o.Albedo, _LineColor.rgb, s);  // 绿色
                o.Emission = _LineColor.rgb * s;
            }
            float minDist4 = min(minDist1,minDist2); 
            // 2. 再画红色（只画两边，覆盖在上面）
            if (minDist4 < _LineWidth)
            {
                float s = 1.0 - smoothstep(0, _LineWidth, minDist4);
                o.Albedo = lerp(o.Albedo, _LineColor2.rgb, s);  // 红色覆盖
                o.Emission = max(o.Emission, _LineColor2.rgb * s);
            }   
        }
        ENDCG
    }
    FallBack "Diffuse"
}