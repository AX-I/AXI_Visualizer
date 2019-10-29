// 0: Vertex shaders

__kernel void Trotate(__global float3 *XYZ,
                     __global float3 *VN,
                     __constant float3 *OldR,
                     __constant float3 *RR,
                     __constant float *O,
                     const int lStart, const int lEnd) {
    int ix = get_global_id(0);
    
    if (ix < (lEnd-lStart)) {
     float3 p = XYZ[ix+lStart];
     float3 n = VN[ix+lStart];
     
     float3 origin = (float3)(O[0], O[1], O[2]);
     p -= origin;
     float3 q = (float3)(dot(p, OldR[0]), dot(p, OldR[1]), dot(p, OldR[2]));
     float3 r = (float3)(dot(q, RR[0]), dot(q, RR[1]), dot(q, RR[2]));
     r += origin;
     
     float3 m = (float3)(dot(n, OldR[0]), dot(n, OldR[1]), dot(n, OldR[2]));
     float3 l = (float3)(dot(m, RR[0]), dot(m, RR[1]), dot(m, RR[2]));
     
     XYZ[ix+lStart] = r;
     VN[ix+lStart] = l;
    }
}
__kernel void Ttranslate(__global float3 *XYZ, __constant float *O,
                        const int lStart, const int lEnd) {
    int ix = get_global_id(0);
    if (ix < (lEnd-lStart)) {
     float3 p = XYZ[ix+lStart] + (float3)(O[0], O[1], O[2]);
     XYZ[ix+lStart] = p;
    }
}
__kernel void Tscale(__global float3 *XYZ, __constant float *O, const float S,
                     const int lStart, const int lEnd) {
    int ix = get_global_id(0);
    if (ix < (lEnd-lStart)) {
     float3 p = XYZ[ix+lStart] - (float3)(O[0], O[1], O[2]);
     XYZ[ix+lStart] = p*S + (float3)(O[0], O[1], O[2]);
    }
}

__kernel void vertL(//__global float3 *XYZ,
                    __global float3 *VN,
                    __global float *I,
                    __constant float *LInt, __constant float3 *LDir, // Directional
                    //__constant float *PInt, __constant float3 *PPos, // Point
                    //__constant float *SInt, __constant float3 *SDir, __constant float3 *SPos,
                    const float ambLight,
                    const int lenL, const int lenP) {
    
    // Block index
    int bx = get_group_id(0);
    int tx = get_local_id(0);

    if ((bx * BLOCK_SIZE + tx) < lenP) {
     int ci = bx * BLOCK_SIZE + tx;
     //float3 x1 = XYZ[ci];
     //float3 vp = (float3)(Vpos[0], Vpos[1], Vpos[2]);
     //x1 -= vp; x2 -= vp; x3 -= vp;
     
     float3 norm = VN[ci];
     float light = 0.f;
     for (char i = 0; i < lenL; i++) {
       light += max(0.f, dot(norm, LDir[i])) * LInt[i];
     }
     light = max(light, ambLight);
     I[ci] = light;
    }
}

