Shader "Hidden/EdgeDetectionOutline"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _EdgeThreshold ("Edge Threshold", Range(0, 1)) = 0.2
        _OutlineColor ("Outline Color", Color) = (1,1,1,1)
        _OutlineIntensity ("Outline Intensity", Range(0, 1)) = 1
        _ShowOriginal ("Show Original", Range(0, 1)) = 1
    }
    SubShader
    {
        // 不写入深度，只做后处理
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_TexelSize;   // 包含纹理像素宽高以及偏移量
            float _EdgeThreshold;
            float4 _OutlineColor;
            float _OutlineIntensity;
            float _ShowOriginal;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            // 计算单点亮度（用于边缘检测）
            float luminance(float3 color)
            {
                return dot(color, float3(0.299, 0.587, 0.114));
            }

            // Sobel 算子 (3x3 邻域)
            float sobelEdgeDetection(sampler2D tex, float2 uv, float2 texelSize)
            {
                // 采样周围 9 个点的亮度
                float2 offsets[9] = {
                    float2(-1, -1), float2(0, -1), float2(1, -1),
                    float2(-1, 0),  float2(0, 0),  float2(1, 0),
                    float2(-1, 1),  float2(0, 1),  float2(1, 1)
                };

                float lum[9];
                for (int i = 0; i < 9; i++)
                {
                    float2 sampleUV = uv + offsets[i] * texelSize;
                    float3 col = tex2D(tex, sampleUV).rgb;
                    lum[i] = luminance(col);
                }

                // Sobel 卷积核 (Gx, Gy)
                float gx = 0.0;
                float gy = 0.0;

                // Gx
                gx += -1.0 * lum[0] + 0.0 * lum[1] + 1.0 * lum[2];
                gx += -2.0 * lum[3] + 0.0 * lum[4] + 2.0 * lum[5];
                gx += -1.0 * lum[6] + 0.0 * lum[7] + 1.0 * lum[8];

                // Gy
                gy += -1.0 * lum[0] + -2.0 * lum[1] + -1.0 * lum[2];
                gy +=  0.0 * lum[3] +  0.0 * lum[4] +  0.0 * lum[5];
                gy +=  1.0 * lum[6] +  2.0 * lum[7] +  1.0 * lum[8];

                return sqrt(gx*gx + gy*gy);
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // 原始颜色
                float4 original = tex2D(_MainTex, i.uv);

                // 边缘强度 0~1
                float edge = sobelEdgeDetection(_MainTex, i.uv, _MainTex_TexelSize.xy);
                edge = saturate(edge);
                // 应用阈值（平滑边缘避免硬切）
                float edgeStrength = smoothstep(0.0, _EdgeThreshold + 0.05, edge);
                edgeStrength = saturate(edgeStrength);

                // 轮廓颜色（带强度）
                float4 outline = _OutlineColor;
                outline.a *= _OutlineIntensity;

                // 混合：原图 + 边缘叠加
                float4 final;
                if (_ShowOriginal > 0.5)
                {
                    // 在原图上叠加轮廓
                    final = original + outline * edgeStrength;
                }
                else
                {
                    // 只显示轮廓（黑背景）
                    final = outline * edgeStrength;
                }

                // 确保颜色范围有效
                return saturate(final);
            }
            ENDCG
        }
    }
}