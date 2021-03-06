// Additive drawing

#define shDist 2

__kernel void drawTex(__global int *TO,
                      __global ushort *Ro, __global ushort *Go, __global ushort *Bo,
                      __global float *F, __global int2 *P, __global float *Z,
                      __global float2 *UV,
                      __global float3 *I,
                      __global float3 *PXYZ,
                      __global float *SD, const int wS, const float sScale,
                      __constant float3 *SV, __constant float *SPos,
                      const float ambLight,
                      __global ushort *TR, __global ushort *TG, __global ushort *TB,
                      const char useShadow,
                      const int wF, const int hF, const int lenP, const int lenT) {

    int bx = get_group_id(0);
    int tx = get_local_id(0);

    if ((bx * BLOCK_SIZE + tx) < lenP) {
    int txd = TO[bx*BLOCK_SIZE + tx];
    int ci = txd * 3;

    // get points
    float z1 = Z[ci];
    float z2 = Z[ci+1];
    float z3 = Z[ci+2];

    int2 xy1 = P[ci];
    int2 xy2 = P[ci+1];
    int2 xy3 = P[ci+2];
    float2 uv1 = UV[ci] * z1;
    float2 uv2 = UV[ci+1] * z2;
    float2 uv3 = UV[ci+2] * z3;
    float3 l1 = I[ci];
    float3 l2 = I[ci+1];
    float3 l3 = I[ci+2];

    float3 pos1 = PXYZ[ci] * z1;
    float3 pos2 = PXYZ[ci+1] * z2;
    float3 pos3 = PXYZ[ci+2] * z3;

    int xtemp; int ytemp;
    float2 uvt; float3 lt; float zt;
    float3 post;

    int x1 = xy1.x; int x2 = xy2.x; int x3 = xy3.x;
    int y1 = xy1.y; int y2 = xy2.y; int y3 = xy3.y;

    // bubble sort y1<y2<y3
    if (y1 > y2) {
      ytemp = y1; xtemp = x1; uvt = uv1; lt = l1; post = pos1; zt = z1;
      y1 = y2; x1 = x2; uv1 = uv2; l1 = l2; pos1 = pos2; z1 = z2;
      y2 = ytemp; x2 = xtemp; uv2 = uvt; l2 = lt; pos2 = post; z2 = zt;
    }
    if (y2 > y3) {
      ytemp = y2; xtemp = x2; uvt = uv2; lt = l2; post = pos2; zt = z2;
      y2 = y3; x2 = x3; uv2 = uv3; l2 = l3; pos2 = pos3; z2 = z3;
      y3 = ytemp; x3 = xtemp; uv3 = uvt; l3 = lt; pos3 = post; z3 = zt;
    }
    if (y1 > y2) {
      ytemp = y1; xtemp = x1; uvt = uv1; lt = l1; post = pos1; zt = z1;
      y1 = y2; x1 = x2; uv1 = uv2; l1 = l2; pos1 = pos2; z1 = z2;
      y2 = ytemp; x2 = xtemp; uv2 = uvt; l2 = lt; pos2 = post; z2 = zt;
    }

    if ((y1 < hF) && (y3 >= 0)) {

    float3 SP = (float3)(SPos[0], SPos[1], SPos[2]);
    float3 SVd = SV[0];
    float3 SVx = SV[1];
    float3 SVy = SV[2];

    float u1 = uv1.x; float u2 = uv2.x; float u3 = uv3.x;
    float v1 = uv1.y; float v2 = uv2.y; float v3 = uv3.y;
    int x4 = (int)(x1 + (float)(y2 - y1)/(float)(y3-y1) * (x3 - x1));
    int y4 = y2;
    float u4 = u1 + (float)(y2 - y1)/(float)(y3-y1) * (u3 - u1);
    float v4 = v1 + (float)(y2 - y1)/(float)(y3-y1) * (v3 - v1);
    float3 l4 = l1 + (float)(y2 - y1)/(float)(y3-y1) * (l3 - l1);
    float3 pos4 = pos1 + (float)(y2 - y1)/(float)(y3-y1) * (pos3 - pos1);
    float z4 = z1 + (float)(y2 - y1)/(float)(y3-y1) * (z3 - z1);

    // fill bottom flat triangle
    float slope1 = (float)(x2-x1) / (float)(y2-y1);
    float slope2 = (float)(x4-x1) / (float)(y4-y1);
    float slopeu1 = (u2-u1) / (float)(y2-y1);
    float slopev1 = (v2-v1) / (float)(y2-y1);
    float slopeu2 = (u4-u1) / (float)(y4-y1);
    float slopev2 = (v4-v1) / (float)(y4-y1);
    float3 slopel1 = (l2-l1) / (float)(y2-y1);
    float3 slopel2 = (l4-l1) / (float)(y4-y1);

    float3 slopepos1 = (pos2-pos1) / (float)(y2-y1);
    float3 slopepos2 = (pos4-pos1) / (float)(y4-y1);

    float slopez1 = (z2-z1) / (float)(y2-y1);
    float slopez2 = (z4-z1) / (float)(y4-y1);

    float cx1 = x1; float cx2 = x1;
    float cu1 = u1; float cv1 = v1; float3 cl1 = l1;
    float cu2 = u1; float cv2 = v1; float3 cl2 = l1;
    float3 cp1 = pos1; float3 cp2 = pos1;
    float cz1 = z1; float cz2 = z1;

    float slopet; float ut; float vt;
    if (slope1 < slope2) {
      slopet = slope1; ut = slopeu1; vt = slopev1; lt = slopel1; post = slopepos1; zt = slopez1;
      slope1 = slope2; slopeu1 = slopeu2; slopev1 = slopev2;
        slopel1 = slopel2; slopepos1 = slopepos2; slopez1 = slopez2;
      slope2 = slopet; slopeu2 = ut; slopev2 = vt; slopel2 = lt; slopepos2 = post; slopez2 = zt;
    }

    for (int cy = y1; cy <= y2; cy++) {
        for (int ax = (int)cx2; ax < (int)cx1; ax++) {
            if ((cy >= 0) && (cy < hF) && (ax >= 0) && (ax < wF)) {
              float t = (ax-cx2)/(cx1-cx2);
              t = max((float)0., min((float)1., t));
              float tz = 1 / ((1-t)*cz2 + t*cz1);
              if (F[wF * cy + ax] > tz) {

                float3 col = ((1-t)*cl2 + t*cl1);
                Ro[wF * cy + ax] += 256*64 * col.x;
                Go[wF * cy + ax] += 256*64 * col.y;
                Bo[wF * cy + ax] += 256*64 * col.z;
              }
            }
        }
        cx1 += slope1;
        cx2 += slope2;
        cu1 += slopeu1;
        cv1 += slopev1;
        cu2 += slopeu2;
        cv2 += slopev2;
        cl1 += slopel1;
        cl2 += slopel2;
        cp1 += slopepos1;
        cp2 += slopepos2;
        cz1 += slopez1;
        cz2 += slopez2;
    }

    // fill top flat triangle
    slope1 = (float)(x3-x2) / (float)(y3-y2);
    slope2 = (float)(x3-x4) / (float)(y3-y4);
    slopeu1 = (u3-u2) / (float)(y3-y2);
    slopev1 = (v3-v2) / (float)(y3-y2);
    slopeu2 = (u3-u4) / (float)(y3-y4);
    slopev2 = (v3-v4) / (float)(y3-y4);
    slopel1 = (l3-l2) / (float)(y3-y2);
    slopel2 = (l3-l4) / (float)(y3-y4);
    slopez1 = (z3-z2) / (float)(y3-y2);
    slopez2 = (z3-z4) / (float)(y3-y4);

    slopepos1 = (pos3-pos2) / (float)(y3-y2);
    slopepos2 = (pos3-pos4) / (float)(y3-y4);

    cx1 = x3; cx2 = x3;
    cu1 = u3; cv1 = v3; cl1 = l3;
    cu2 = u3; cv2 = v3; cl2 = l3;
    cp1 = pos3; cp2 = pos3;
    cz1 = z3; cz2 = z3;

    if (slope1 < slope2) {
      slopet = slope1; ut = slopeu1; vt = slopev1; lt = slopel1; post = slopepos1; zt = slopez1;
      slope1 = slope2; slopeu1 = slopeu2; slopev1 = slopev2;
        slopel1 = slopel2; slopepos1 = slopepos2; slopez1 = slopez2;
      slope2 = slopet; slopeu2 = ut; slopev2 = vt; slopel2 = lt; slopepos2 = post; slopez2 = zt;
    }

    for (int cy = y3; cy > y2; cy--) {
        for (int ax = (int)cx1; ax < (int)cx2; ax++) {
            if ((cy >= 0) && (cy < hF) && (ax >= 0) && (ax < wF)) {
              float t = (ax-cx2)/(cx1-cx2);
              t = max((float)0., min((float)1., t));
              float tz = 1 / ((1-t)*cz2 + t*cz1);
              if (F[wF * cy + ax] > tz) {

                float3 col = ((1-t)*cl2 + t*cl1);
                Ro[wF * cy + ax] += 256*64 * col.x;
                Go[wF * cy + ax] += 256*64 * col.y;
                Bo[wF * cy + ax] += 256*64 * col.z;
              }
            }
        }
        cx1 -= slope1;
        cx2 -= slope2;
        cu1 -= slopeu1;
        cv1 -= slopev1;
        cu2 -= slopeu2;
        cv2 -= slopev2;
        cl1 -= slopel1;
        cl2 -= slopel2;
        cp1 -= slopepos1;
        cp2 -= slopepos2;
        cz1 -= slopez1;
        cz2 -= slopez2;
    }
    }
    }
}
