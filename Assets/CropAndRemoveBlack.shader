Shader "Hidden/CropAndRemoveBlack"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _RectMin ("Rect Min", Vector) = (0,0,0,0)
        _RectMax ("Rect Max", Vector) = (1,1,0,0)
        _BlackThreshold ("Black Threshold", Range(0,1)) = 0.1
    }
    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" }
        Blend SrcAlpha OneMinusSrcAlpha
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _RectMin;
            float4 _RectMax;
            float _BlackThreshold;

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata_img v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.texcoord;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // 1. 将当前 UV 映射到裁剪矩形内
                float2 cropUV = _RectMin.xy + i.uv.xy * (_RectMax.xy - _RectMin.xy);
                // 防止浮点误差导致采样超出 [0,1] 范围（可选）
                cropUV = clamp(cropUV, 0.0, 1.0);
    
                // 2. 采样轮廓纹理
                fixed4 col = tex2D(_MainTex, cropUV);
    
                // 3. 判断是否为边缘（基于亮度阈值）
                float luminance = dot(col.rgb, float3(0.299, 0.587, 0.114));
    
                // 假设轮廓是白色或其他高亮度颜色，亮度 > 阈值即为边缘
                if (luminance > _BlackThreshold)
                {
                    // 输出边缘颜色，Alpha = 1（完全不透明);
                    return fixed4(1,1,1,1);
                }
                else
                {
                    // 非边缘像素完全透明（丢弃也可以，但丢弃可能导致 GPU 优化问题，返回透明更安全）
                    return fixed4(0, 0, 0, 0);
                }
            }
            
            ENDCG
        }
    }
}