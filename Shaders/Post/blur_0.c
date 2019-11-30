__kernel void blurH(__global ushort *r2, __global ushort *g2, __global ushort *b2,
                    __global ushort *Ro, __global ushort *Go, __global ushort *Bo,
                    int wF, int hF, const int BS,
                    int stepW, float stepH) {

    int bx = get_group_id(0);
    int by = get_group_id(1);
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    int ci = bx * BS + tx;
    int cj = by * BS + ty;

    int h1 = stepH * cj;
    int h2 = stepH * (cj+1);
    
    wF = wF/2; hF = hF/2; stepH = stepH/2;
    h1 = stepH * cj;
    h2 = stepH * (cj+1);
    
    for (int cy = h1; cy < min(h2, hF-1); cy++) {
        for (int cx = ci; cx < wF; cx += stepW) {
          int y1 = wF*cy*4+1;
          int x1 = cx*2+1;
          float cr = r2[wF*cy*4 + cx*2]/4 + r2[y1 + cx*2]/4 + r2[wF*cy*4 + x1]/4 + r2[y1 + x1]/4;
          float cg = g2[wF*cy*4 + cx*2]/4 + g2[y1 + cx*2]/4 + g2[wF*cy*4 + x1]/4 + g2[y1 + x1]/4;
          float cb = b2[wF*cy*4 + cx*2]/4 + b2[y1 + cx*2]/4 + b2[wF*cy*4 + x1]/4 + b2[y1 + x1]/4;
          float lum = 0.3626*cr + 0.5152*cg + 0.1222*cb;
          Ro[wF * cy + cx] = cr*lum/256/256;
          Go[wF * cy + cx] = cg*lum/256/256;
          Bo[wF * cy + cx] = cb*lum/256/256;
        }
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    for (int cy = h1; cy < min(h2, hF); cy++) {
        for (int cx = ci; cx < wF; cx += stepW) {
            float3 a = (float3)0;
            if ((cx > 5) && (cx < (wF-6))) {
                a += 0.3f*(float3)(Ro[wF * cy + cx-6], Go[wF * cy + cx-6], Bo[wF * cy + cx-6]);
                a += 0.4f*(float3)(Ro[wF * cy + cx-5], Go[wF * cy + cx-5], Bo[wF * cy + cx-5]);
                a += 0.6f*(float3)(Ro[wF * cy + cx-4], Go[wF * cy + cx-4], Bo[wF * cy + cx-4]);
                a += (float3)(Ro[wF * cy + cx-3], Go[wF * cy + cx-3], Bo[wF * cy + cx-3]);
                a += 2*(float3)(Ro[wF * cy + cx-2], Go[wF * cy + cx-2], Bo[wF * cy + cx-2]);
                a += 4*(float3)(Ro[wF * cy + cx-1], Go[wF * cy + cx-1], Bo[wF * cy + cx-1]);
                a += 8*(float3)(Ro[wF * cy + cx], Go[wF * cy + cx], Bo[wF * cy + cx]);
                a += 4*(float3)(Ro[wF * cy + cx+1], Go[wF * cy + cx+1], Bo[wF * cy + cx+1]);
                a += 2*(float3)(Ro[wF * cy + cx+2], Go[wF * cy + cx+2], Bo[wF * cy + cx+2]);
                a += (float3)(Ro[wF * cy + cx+3], Go[wF * cy + cx+3], Bo[wF * cy + cx+3]);
                a += 0.6f*(float3)(Ro[wF * cy + cx+4], Go[wF * cy + cx+4], Bo[wF * cy + cx+4]);
                a += 0.4f*(float3)(Ro[wF * cy + cx+5], Go[wF * cy + cx+5], Bo[wF * cy + cx+5]);
                a += 0.3f*(float3)(Ro[wF * cy + cx+6], Go[wF * cy + cx+6], Bo[wF * cy + cx+6]);
                a /= 24.6f;
                //a /= 16;

                Ro[wF * cy + cx] = (ushort)a.x; //(0.5*a.x + 0.5*Ro[wF * cy + cx]);
                Go[wF * cy + cx] = (ushort)a.y; //(0.5*a.y + 0.5*Go[wF * cy + cx]);
                Bo[wF * cy + cx] = (ushort)a.z; //(0.5*a.z + 0.5*Bo[wF * cy + cx]);
            }
        }
    }
}
