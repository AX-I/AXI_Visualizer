// Cloud particles

#define shBias 4.f

__kernel void sh(__global float *F,
                 __global float3 *XYZ,
                 const int size,
                 __global ushort *TR, const int lenT,
                 __constant float *Vpos, __constant float3 *VV,
                 const float avgx, const float avgy, const float avgz,
                 const float devx, const float devy, const float devz,
                 const float sScale, const int wF, const int hF,
                 const float cAX, const float cAY, const int lenP) {

    // Block index
    int bx = get_group_id(0);
    int tx = get_local_id(0);

    if ((bx * BLOCK_SIZE + tx) < lenP) {
    int ci = (bx * BLOCK_SIZE + tx);
    float3 x1 = XYZ[ci];

    float3 avgpos = (float3)(avgx, avgy, avgz);
    float3 dev = (float3)(devx, devy, devz);
    float3 dpos = fabs(x1 - avgpos) / dev;
    float avdist = sqrt(dot(dpos, dpos));
    float sopacity = 1;
    if (avdist > 2.25f) sopacity *= max(0.f, 4*(2.5f-avdist));

    float3 vp = (float3)(Vpos[0], Vpos[1], Vpos[2]);
    x1 -= vp;
    float3 SVd = VV[0];
    float3 SVx = VV[1];
    float3 SVy = VV[2];
    float dd = dot(x1, SVd);
    float dx = dot(x1, SVx);
    float dy = dot(x1, SVy);
    dd += shBias;

    //if ((dd > 0) && (fabs(dx) < cAX) && (fabs(dy) < cAY)) {
      dx = dx * sScale + wF/2;
      dy = dy * -sScale + hF/2;
      int dix = (int)(dx);
      int diy = (int)(dy);
      int vsize = size;
      for (int ay = max(0, diy-vsize); ay < min(hF-1, diy+vsize); ay++) {
        for (int ax = max(0, dix-vsize); ax < min(wF-1, dix+vsize); ax++) {
          if (F[wF * ay + ax] > dd) {
            float texr1 = (ax - (dix-vsize)) * lenT / (2.f*vsize);
            float texr2 = (ay - (diy-vsize)) * lenT / (2.f*vsize);
            int tex1 = (int)texr1;
            int tex2 = (int)texr2;

            int tex = tex1 + lenT*tex2;

            float u = TR[tex];
            if ((u > 256*64) && (sopacity>0.5f)) F[wF * ay + ax] = dd;
          }
        }
      }
    //}
    }
}
