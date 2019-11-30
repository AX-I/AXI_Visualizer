// Vertex shader

__kernel void wave(__global float3 *XYZ,
                   __global float3 *VN,
                   const float ox, const float oy, const float oz,
                   const float wDirx, const float wDirz,
                   const float pScale,
                   const float time,
                   __constant float *wLen, __constant float *wAmp,
                   __constant float *wSpd, const char lenW,
                   const int lenP) {
    // Block index
    int ix = get_global_id(0);
    
    if (ix < lenP) {
     float3 origin = (float3)(ox, oy, oz);
     float3 coords = XYZ[ix];
     coords -= origin;
     coords /= pScale;
     float2 wxz = (float2)(coords.x, coords.z);
     float2 wdr = (float2)(wDirx, wDirz);
     
     float py = 0;
     float deriv = 0;
     for (char i = 0; i < lenW; i++) {
       float frq = 2.f / wLen[i];
       float dist = dot(wdr, wxz) * frq + time * wSpd[i];
       py += wAmp[i] * sin(dist);
       deriv += cos(dist);
     }
     
     float3 p = (float3)(wxz.x, py, wxz.y);
     float3 tg = (float3)(deriv * wdr.x, -1, deriv * wdr.y);
     tg = normalize(tg);
     
     p *= pScale;
     p += origin;
     
     //printf("coords x %f z %f \n dist %f \n final %f %f %f \n",
     //        coords.x, coords.z, dist, p.x, p.y, p.z);
     //printf("final %f %f %f \n", p.x, p.y, p.z);
     //if (ix == 0) printf("norm %f %f %f \n", tg.x, tg.y, tg.z);
     
     XYZ[ix] = p;
     VN[ix] = tg;
    }
}
