// Particles

__kernel void ps(__global ushort *Ro, __global ushort *Go, __global ushort *Bo,
                 __global float *F,
                 __global float3 *XYZ, __global ushort3 *I,
                 const float opacity, const int size,
                 //__constant ushort *TR, __constant ushort *TG, __constant ushort *TB
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
      vsize = min(4, vsize);
      float vopacity = opacity;
      if (vsize == 0) {
        vsize = 1; vopacity /= 2.f;
      }
      for (int ay = max(0, diy-vsize); ay < min(hF-1, diy+vsize); ay++) {
        for (int ax = max(0, dix-vsize); ax < min(wF-1, dix+vsize); ax++) {
          if (F[wF * ay + ax] > dd) {
            
            Ro[wF * ay + ax] += 256*vopacity * col.x;
            Go[wF * ay + ax] += 256*vopacity * col.y;
            Bo[wF * ay + ax] += 256*vopacity * col.z;
          }
        }
      }
    }
    }
}
