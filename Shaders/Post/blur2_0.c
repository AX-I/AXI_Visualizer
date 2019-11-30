__kernel void blurV(__global ushort *Ro, __global ushort *Go, __global ushort *Bo,
                    __global ushort *r2, __global ushort *g2, __global ushort *b2,
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

    for (int cy = h1; cy < min(h2, hF); cy++) {
        for (int cx = ci; cx < wF; cx += stepW) {
            float3 a = (float3)0;
            if ((cy > 5) && (cy < (hF-6))) {
              a += 0.3f*(float3)(Ro[wF * (cy-6) + cx], Go[wF * (cy-6) + cx], Bo[wF * (cy-6) + cx]);
              a += 0.4f*(float3)(Ro[wF * (cy-5) + cx], Go[wF * (cy-5) + cx], Bo[wF * (cy-5) + cx]);
              a += 0.6f*(float3)(Ro[wF * (cy-4) + cx], Go[wF * (cy-4) + cx], Bo[wF * (cy-4) + cx]);
              a += (float3)(Ro[wF * (cy-3) + cx], Go[wF * (cy-3) + cx], Bo[wF * (cy-3) + cx]);
              a += 2*(float3)(Ro[wF * (cy-2) + cx], Go[wF * (cy-2) + cx], Bo[wF * (cy-2) + cx]);
              a += 4*(float3)(Ro[wF * (cy-1) + cx], Go[wF * (cy-1) + cx], Bo[wF * (cy-1) + cx]);
              a += 8*(float3)(Ro[wF * cy + cx], Go[wF * cy + cx], Bo[wF * cy + cx]);
              a += 4*(float3)(Ro[wF * (cy+1) + cx], Go[wF * (cy+1) + cx], Bo[wF * (cy+1) + cx]);
              a += 2*(float3)(Ro[wF * (cy+2) + cx], Go[wF * (cy+2) + cx], Bo[wF * (cy+2) + cx]);
              a += (float3)(Ro[wF * (cy+3) + cx], Go[wF * (cy+3) + cx], Bo[wF * (cy+3) + cx]);
              a += 0.6f*(float3)(Ro[wF * (cy+4) + cx], Go[wF * (cy+4) + cx], Bo[wF * (cy+4) + cx]);
              a += 0.4f*(float3)(Ro[wF * (cy+5) + cx], Go[wF * (cy+5) + cx], Bo[wF * (cy+5) + cx]);
              a += 0.3f*(float3)(Ro[wF * (cy+6) + cx], Go[wF * (cy+6) + cx], Bo[wF * (cy+6) + cx]);
              a /= 24.6f;
              //a /= 16;

              Ro[wF * cy + cx] = (ushort)a.x;
              Go[wF * cy + cx] = (ushort)a.y;
              Bo[wF * cy + cx] = (ushort)a.z;
            }
        }
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    wF = wF*2; hF = hF*2; stepH = stepH*2;
    h1 = stepH * cj;
    h2 = stepH * (cj+1);
    int lenT = wF/2;
    
    for (int cy = h1; cy < min(h2, hF); cy++) {
        for (int cx = ci; cx < wF; cx += stepW) {
                float texr1 = cx/2.f;
                int tex1 = (int)texr1;
                texr1 -= tex1;
                float texr2 = cy/2.f;
                int tex2 = (int)texr2;
                texr2 -= tex2;
                
                int tex = tex1 + lenT*tex2;
                int tex10 = min(tex1+1, lenT-1) + lenT*tex2;
                int tex01 = tex1 + lenT*min(tex2+1, hF/2-1);
                int tex11 = min(tex1+1, lenT-1) + lenT*min(tex2+1, hF/2-1);
                float texi1 = 1-texr1;
                float texi2 = 1-texr2;
                
          r2[wF * cy + cx] = r2[wF * cy + cx] + (texi1*texi2*Ro[tex] + texr1*texi2*Ro[tex10] + texi1*texr2*Ro[tex01] + texr1*texr2*Ro[tex11]);
          g2[wF * cy + cx] = g2[wF * cy + cx] + (texi1*texi2*Go[tex] + texr1*texi2*Go[tex10] + texi1*texr2*Go[tex01] + texr1*texr2*Go[tex11]);
          b2[wF * cy + cx] = b2[wF * cy + cx] + (texi1*texi2*Bo[tex] + texr1*texi2*Bo[tex10] + texi1*texr2*Bo[tex01] + texr1*texr2*Bo[tex11]);
        }
    }
}
