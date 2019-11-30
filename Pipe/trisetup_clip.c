// 1: Triangle setup, clipping

__kernel void setup(__global float3 *XYZ,
                    __global float2 *UV,// __global float *I, __global float3 *N,
                    __global int *TI,  // Indices out
                    __global int *TN,  // number of tris
                    __global int2 *XY,   // Screen coords out
                    __global float *ZZ,  // Depths out
                    __global int *bigIndex, // const int bigSize,
                    __constant float *Vpos, __constant float3 *VV,
                    const float sScale, const int wF, const int hF,
                    const float cAX, const float cAY, const int lenP,
                    const int numBlocks) {
    
    // Block index
    int bx = get_group_id(0);
    int tx = get_local_id(0);
    
    //if ((bx == 0) && (tx == 0)) printf("Trisetup \n");
    if (tx == 0) TN[bx] = 0;
    barrier(CLK_GLOBAL_MEM_FENCE);

    if ((bx * BLOCK_SIZE + tx) < lenP) {
    int ci = (bx * BLOCK_SIZE + tx) * 3;
    float3 x1 = XYZ[ci];
    float3 x2 = XYZ[ci+1];
    float3 x3 = XYZ[ci+2];
    
    float3 vp = (float3)(Vpos[0], Vpos[1], Vpos[2]);
    x1 -= vp; x2 -= vp; x3 -= vp;
    float3 SVd = VV[0];
    float3 SVx = VV[1];
    float3 SVy = VV[2];
    float3 dd = (float3)(dot(x1, SVd), dot(x2, SVd), dot(x3, SVd));
    float3 dx = (float3)(dot(x1, SVx), dot(x2, SVx), dot(x3, SVx));
    float3 dy = (float3)(dot(x1, SVy), dot(x2, SVy), dot(x3, SVy));
    dd = 1 / dd;
    dx *= dd; dy *= dd;
    float yes = false;
    if (all(dd > 0) && all(fabs(dx) < cAX) && all(fabs(dy) < cAY)) {
      yes = true;
    }
    if (yes) {
      dx = dx * -sScale + wF/2;
      dy = dy * sScale + hF/2;
      int3 dix = convert_int3(dx);
      int3 diy = convert_int3(dy);
      int nextI;
      if ((max(dix.x, max(dix.y, dix.z)) - min(dix.x, min(dix.y, dix.z)) > 100) ||
          (max(diy.x, max(diy.y, diy.z)) - min(diy.x, min(diy.y, diy.z)) > 100)) {
        printf("big");
        int3 dmx = (dix.xyz + dix.yzx) >> 1;
        int3 dmy = (diy.xyz + diy.yzx) >> 1;
        float3 dmz = (dd.xyz + dd.yzx) / 2;
        nextI = atomic_add(*bigIndex, 4);
        if (nextI < (BLOCK_SIZE - 4)) {
         int bInd = numBlocks*BLOCK_SIZE + nextI;
         XY[bInd*3] = (int2)(dix.x, diy.x);
         XY[bInd*3+1] = (int2)(dmx.x, dmy.x);
         XY[bInd*3+2] = (int2)(dmx.z, dmy.z);
         XY[bInd*3+3] = (int2)(dix.y, diy.y);
         XY[bInd*3+4] = (int2)(dmx.x, dmy.x);
         XY[bInd*3+5] = (int2)(dmx.y, dmy.y);
         XY[bInd*3+6] = (int2)(dix.z, diy.z);
         XY[bInd*3+7] = (int2)(dmx.z, dmy.z);
         XY[bInd*3+8] = (int2)(dmx.y, dmy.y);
         XY[bInd*3+9] = (int2)(dmx.x, dmy.x);
         XY[bInd*3+10] = (int2)(dmx.y, dmy.y);
         XY[bInd*3+11] = (int2)(dmx.z, dmy.z);
         ZZ[bInd*3] = dd.s0;
         ZZ[bInd*3+1] = dmz.s0;
         ZZ[bInd*3+2] = dmz.s2;
         ZZ[bInd*3+3] = dd.s1;
         ZZ[bInd*3+4] = dmz.s0;
         ZZ[bInd*3+5] = dmz.s1;
         ZZ[bInd*3+6] = dd.s2;
         ZZ[bInd*3+7] = dmz.s1;
         ZZ[bInd*3+8] = dmz.s2;
         ZZ[bInd*3+9] = dmz.s0;
         ZZ[bInd*3+10] = dmz.s1;
         ZZ[bInd*3+11] = dmz.s2;
         //TI[bx*BLOCK_SIZE + nextI] = bx * BLOCK_SIZE + tx;
        }
      }
      else {
        nextI = atomic_inc(&TN[bx]);
        XY[(bx*BLOCK_SIZE + tx)*3] = (int2)(dix.s0, diy.s0);
        XY[(bx*BLOCK_SIZE + tx)*3+1] = (int2)(dix.s1, diy.s1);
        XY[(bx*BLOCK_SIZE + tx)*3+2] = (int2)(dix.s2, diy.s2);
        ZZ[(bx*BLOCK_SIZE + tx)*3] = dd.s0;
        ZZ[(bx*BLOCK_SIZE + tx)*3+1] = dd.s1;
        ZZ[(bx*BLOCK_SIZE + tx)*3+2] = dd.s2;
        TI[bx*BLOCK_SIZE + nextI] = bx * BLOCK_SIZE + tx;
      }
    }
    }
}
