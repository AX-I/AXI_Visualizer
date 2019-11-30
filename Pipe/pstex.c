// Sprite textured particles

__kernel void ps(__global ushort *Ro, __global ushort *Go, __global ushort *Bo,
                 __global float *F,
                 __global float3 *XYZ, __global ushort3 *I,
                 const float opacity, const int size,
                 __constant ushort *TR, const int lenT,
                 __constant float *Vpos, __constant float3 *VV,
                 const float sScale, const int wF, const int hF,
                 const float cAX, const float cAY, const int lenP) {
    
    // Block index
    int bx = get_group_id(0);
    int tx = get_local_id(0);
    
    if ((bx * BLOCK_SIZE + tx) < lenP) {
    int ci = (bx * BLOCK_SIZE + tx);
    float3 x1 = XYZ[ci];
    ushort3 col = I[ci];
    
    float3 vp = (float3)(Vpos[0], Vpos[1], Vpos[2]);
    x1 -= vp;
    float3 SVd = VV[0];
    float3 SVx = VV[1];
    float3 SVy = VV[2];
    float dd = dot(x1, SVd);
    float dx = dot(x1, SVx);
    float dy = dot(x1, SVy);
    dx /= dd; dy /= dd;
    if ((dd > 0) && (fabs(dx) < cAX) && (fabs(dy) < cAY)) {
      dx = dx * -sScale + wF/2;
      dy = dy * sScale + hF/2;
      int dix = (int)(dx);
      int diy = (int)(dy);
      int vsize = size * 8 / dd;
      vsize = min(16, vsize);
      float vopacity = opacity;
      if (vsize <= 1) {
        vsize = 2; vopacity *= min(1.f, size * 8.f / dd);
      }
      for (int ay = max(0, diy-vsize); ay < min(hF-1, diy+vsize); ay++) {
        for (int ax = max(0, dix-vsize); ax < min(wF-1, dix+vsize); ax++) {
          if (F[wF * ay + ax] > dd) {
            float texr1 = (ax - (dix-vsize)) * lenT / (2.f*vsize);
            float texr2 = (ay - (diy-vsize)) * lenT / (2.f*vsize);
            int tex1 = (int)texr1;
            int tex2 = (int)texr2;
            texr1 -= tex1; texr2 -= tex2;
            
            int tex = tex1 + lenT*tex2;
            int tex10 = min(tex1+1, lenT-1) + lenT*tex2;
            int tex01 = tex1 + lenT*min(tex2+1, lenT-1);
            int tex11 = min(tex1+1, lenT-1) + lenT*min(tex2+1, lenT-1);
            float texi1 = 1-texr1;
            float texi2 = 1-texr2;
            
            float u = (texi1*texi2*convert_float3(TR[tex]) +
                        texr1*texi2*convert_float3(TR[tex10]) +
                        texi1*texr2*convert_float3(TR[tex01]) +
                        texr1*texr2*convert_float3(TR[tex11]));
            
            Ro[wF * ay + ax] += vopacity * u * col.x;
            Go[wF * ay + ax] += vopacity * u * col.y;
            Bo[wF * ay + ax] += vopacity * u * col.z;
          }
        }
      }
    }
    }
}
