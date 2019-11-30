// Perspective-correct triilinear texture drawing with smooth brightness and 2 PCF shadow maps
// Multitexture

#define shDist 2
#define mipBias 0.2f

__kernel void drawTex(__global int *TO,
                      __global ushort *Ro, __global ushort *Go, __global ushort *Bo,
                      __global float *F, __global int2 *P, __global float *Z,
                      __global float2 *UV,
                      __global float *I,
                      __global float3 *PXYZ,
                      __global float *SD, const int wS, const float sScale,
                      __constant float3 *SV, __constant float *SPos,
                      __global float *SD2, const int wS2, const float sScale2,
                      __constant float3 *SV2, __constant float *SPos2,
                      const float ambLight,
                      __global ushort *TR, __global ushort *TG, __global ushort *TB,
                      __global ushort *TR2, __global ushort *TG2, __global ushort *TB2,
                      const int wF, const int hF, const int lenP, const int numMip,
                      const float h1, const float h2) {

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
    float2 uv1 = UV[ci];
    float2 uv2 = UV[ci+1];
    float2 uv3 = UV[ci+2];
    float l1 = I[ci];
    float l2 = I[ci+1];
    float l3 = I[ci+2];

    int x1 = xy1.x; int x2 = xy2.x; int x3 = xy3.x;
    int y1 = xy1.y; int y2 = xy2.y; int y3 = xy3.y;

    float2 slopeu1 = fabs((uv2-uv1) / (float)((y2-y1)*(y2-y1)+(x2-x1)*(x2-x1)));
    float2 slopeu2 = fabs((uv3-uv1) / (float)((y3-y1)*(y3-y1)+(x3-x1)*(x3-x1)));

    //float d1 = sqrt(slopeu1.x * slopeu1.x + slopeu1.y * slopeu1.y);
    //float d2 = sqrt(slopeu2.x * slopeu2.x + slopeu2.y * slopeu2.y);
    //float dt = max(d1, d2);
    float dt = max(slopeu1.x, max(slopeu1.y, max(slopeu2.x, slopeu2.y)));
    float fmip = min((float)(numMip-1), fabs(log2(dt))/2 + mipBias);
    int mip = (int)fmip;
    fmip -= mip;
    int lenMip = 1 << mip;
    int lenMip2 = lenMip << 1;
    int startMip = 0;
    for (int i = 0; i < mip; i++) {
      startMip += (1 << i) * (1 << i);
    }
    int startMip2 = startMip + (lenMip * lenMip);

    float u1 = uv1.x; float u2 = uv2.x; float u3 = uv3.x;
    float v1 = uv1.y; float v2 = uv2.y; float v3 = uv3.y;
    u1 *= z1;
    v1 *= z1;
    u2 *= z2;
    v2 *= z2;
    u3 *= z3;
    v3 *= z3;

    float3 pos1 = PXYZ[ci] * z1;
    float3 pos2 = PXYZ[ci+1] * z2;
    float3 pos3 = PXYZ[ci+2] * z3;

    int ytemp; int xtemp;
    float ut; float vt; float lt; float zt;
    float3 post;

    // bubble sort y1<y2<y3
    if (y1 > y2) {
      ytemp = y1; xtemp = x1; ut = u1; vt = v1; lt = l1; post = pos1; zt = z1;
      y1 = y2; x1 = x2; u1 = u2; v1 = v2; l1 = l2; pos1 = pos2; z1 = z2;
      y2 = ytemp; x2 = xtemp; u2 = ut; v2 = vt; l2 = lt; pos2 = post; z2 = zt;
    }
    if (y2 > y3) {
      ytemp = y2; xtemp = x2; ut = u2; vt = v2; lt = l2; post = pos2; zt = z2;
      y2 = y3; x2 = x3; u2 = u3; v2 = v3; l2 = l3; pos2 = pos3; z2 = z3;
      y3 = ytemp; x3 = xtemp; u3 = ut; v3 = vt; l3 = lt; pos3 = post; z3 = zt;
    }
    if (y1 > y2) {
      ytemp = y1; xtemp = x1; ut = u1; vt = v1; lt = l1; post = pos1; zt = z1;
      y1 = y2; x1 = x2; u1 = u2; v1 = v2; l1 = l2; pos1 = pos2; z1 = z2;
      y2 = ytemp; x2 = xtemp; u2 = ut; v2 = vt; l2 = lt; pos2 = post; z2 = zt;
    }

    if ((y1 < hF) && (y3 >= 0)) {

    float3 SP = (float3)(SPos[0], SPos[1], SPos[2]);
    float3 SVd = SV[0];
    float3 SVx = SV[1];
    float3 SVy = SV[2];

    float3 SP2 = (float3)(SPos2[0], SPos2[1], SPos2[2]);
    float3 SVd2 = SV2[0];
    float3 SVx2 = SV2[1];
    float3 SVy2 = SV2[2];

    float ydiff1 = (float)(y2 - y1)/(float)(y3-y1);
    int x4 = (int)(x1 + ydiff1 * (x3 - x1));
    int y4 = y2;
    float u4 = u1 + ydiff1 * (u3 - u1);
    float v4 = v1 + ydiff1 * (v3 - v1);
    float l4 = l1 + ydiff1 * (l3 - l1);
    float3 pos4 = pos1 + ydiff1 * (pos3 - pos1);
    float z4 = z1 + ydiff1 * (z3 - z1);

    // fill bottom flat triangle
    ydiff1 = 1 / (float)(y2-y1);

    float slope1 = (float)(x2-x1) * ydiff1;
    float slopeu1 = (u2-u1) * ydiff1;
    float slopev1 = (v2-v1) * ydiff1;
    float slopel1 = (l2-l1) * ydiff1;
    float3 slopepos1 = (pos2-pos1) * ydiff1;
    float slopez1 = (z2-z1) * ydiff1;

    ydiff1 = 1 / (float)(y4-y1);
    float slope2 = (float)(x4-x1) * ydiff1;
    float slopeu2 = (u4-u1) * ydiff1;
    float slopev2 = (v4-v1) * ydiff1;
    float slopel2 = (l4-l1) * ydiff1;
    float3 slopepos2 = (pos4-pos1) * ydiff1;
    float slopez2 = (z4-z1) * ydiff1;

    float cx1 = x1; float cx2 = x1;
    float cu1 = u1; float cv1 = v1; float cl1 = l1;
    float cu2 = u1; float cv2 = v1; float cl2 = l1;
    float3 cp1 = pos1; float3 cp2 = pos1;
    float cz1 = z1; float cz2 = z1;

    float slopet;
    if (slope1 < slope2) {
      slopet = slope1; ut = slopeu1; vt = slopev1; lt = slopel1; post = slopepos1; zt = slopez1;
      slope1 = slope2; slopeu1 = slopeu2; slopev1 = slopev2;
        slopel1 = slopel2; slopepos1 = slopepos2; slopez1 = slopez2;
      slope2 = slopet; slopeu2 = ut; slopev2 = vt; slopel2 = lt; slopepos2 = post; slopez2 = zt;
    }

    for (int cy = y1; cy <= y2; cy++) {
        for (int ax = (int)cx2; ax <= (int)cx1; ax++) {
            if ((cy >= 0) && (cy < hF) && (ax >= 0) && (ax < wF)) {
              float t = (ax-cx2)/(cx1-cx2);
              t = max(0.f, min(1.f, t));
              float tz = 1 / ((1-t)*cz2 + t*cz1);
              if (F[wF * cy + ax] > tz) {
                F[wF * cy + ax] = tz;

                float3 pos = ((1-t)*cp2 + t*cp1) * tz - SP;
                float depth = dot(pos, SVd);
                int sx = (int)(dot(pos, SVx) * sScale) + wS;
                int sy = (int)(dot(pos, SVy) * -sScale) + wS;
                int shadow = 0;
                if ((sx >= shDist) && (sx < 2*wS-shDist) &&
                    (sy >= shDist) && (sy < 2*wS-shDist)) {
                  if (SD[2*wS * sy + sx] < depth) shadow += 1;
                  if (SD[2*wS * (sy+shDist) + sx] < depth) shadow += 1;
                  if (SD[2*wS * (sy-shDist) + sx] < depth) shadow += 1;
                  if (SD[2*wS * sy + (sx+shDist)] < depth) shadow += 1;
                  if (SD[2*wS * sy + (sx-shDist)] < depth) shadow += 1;
                }
                pos = ((1-t)*cp2 + t*cp1) * tz - SP2;
                depth = dot(pos, SVd2);
                sx = (int)(dot(pos, SVx2) * sScale2) + wS2;
                sy = (int)(dot(pos, SVy2) * -sScale2) + wS2;
                if ((sx >= shDist) && (sx < 2*wS2-shDist) &&
                    (sy >= shDist) && (sy < 2*wS2-shDist)) {
                  if (SD2[2*wS2 * sy + sx] < depth) shadow += 1;
                  if (SD2[2*wS2 * (sy+shDist) + sx] < depth) shadow += 1;
                  if (SD2[2*wS2 * (sy-shDist) + sx] < depth) shadow += 1;
                  if (SD2[2*wS2 * sy + (sx+shDist)] < depth) shadow += 1;
                  if (SD2[2*wS2 * sy + (sx-shDist)] < depth) shadow += 1;
                }
                shadow = min(5, shadow);
                float light = shadow * ambLight + (5-shadow) * ((1-t)*cl2 + t*cl1);
                light *= 0.2f;

                float texr1 = ((1-t)*cu2 + t*cu1) * tz * (lenMip-1);
                int tex1 = (int)texr1;
                texr1 -= tex1;
                tex1 = min(tex1, lenMip-1);
                float texr2 = (lenMip-1) * ((1-t)*cv2 + t*cv1) * tz;
                int tex2 = (int)texr2;
                texr2 -= tex2;
                tex2 = min(tex2, lenMip-1);

                int tex = startMip + tex1 + lenMip*tex2;
                int tex10 = startMip + min(tex1+1, lenMip-1) + lenMip*tex2;
                int tex01 = startMip + tex1 + lenMip*min(tex2+1, lenMip-1);
                int tex11 = startMip + min(tex1+1, lenMip-1) + lenMip*min(tex2+1, lenMip-1);

                float texi1 = 1-texr1;
                float texi2 = 1-texr2;

                pos = ((1-t)*cp2 + t*cp1) * tz;
                float blend = max(0.f, min(1.f, (pos.y - h1) / (h2 - h1)));

                ushort outR = (ushort)((texi1*texi2*TR[tex] + texr1*texi2*TR[tex10] +
                                   texi1*texr2*TR[tex01] + texr1*texr2*TR[tex11])
                                   * light * (1-fmip) * (1-blend));
                ushort outG = (ushort)((texi1*texi2*TG[tex] + texr1*texi2*TG[tex10] +
                                   texi1*texr2*TG[tex01] + texr1*texr2*TG[tex11])
                                   * light * (1-fmip) * (1-blend));
                ushort outB = (ushort)((texi1*texi2*TB[tex] + texr1*texi2*TB[tex10] +
                                   texi1*texr2*TB[tex01] + texr1*texr2*TB[tex11])
                                   * light * (1-fmip) * (1-blend));
                outR += (ushort)((texi1*texi2*TR2[tex] + texr1*texi2*TR2[tex10] +
                                   texi1*texr2*TR2[tex01] + texr1*texr2*TR2[tex11])
                                   * light * (1-fmip) * blend);
                outG += (ushort)((texi1*texi2*TG2[tex] + texr1*texi2*TG2[tex10] +
                                   texi1*texr2*TG2[tex01] + texr1*texr2*TG2[tex11])
                                   * light * (1-fmip) * blend);
                outB += (ushort)((texi1*texi2*TB[tex] + texr1*texi2*TB[tex10] +
                                   texi1*texr2*TB[tex01] + texr1*texr2*TB[tex11])
                                   * light * (1-fmip) * blend);


                texr1 = ((1-t)*cu2 + t*cu1) * tz * (lenMip2-1);
                tex1 = (int)texr1;
                texr1 -= tex1;
                tex1 = min(tex1, lenMip2-1);
                texr2 = (lenMip2-1) * ((1-t)*cv2 + t*cv1) * tz;
                tex2 = (int)texr2;
                texr2 -= tex2;
                tex2 = min(tex2, lenMip2-1);

                tex = startMip2 + tex1 + lenMip2*tex2;
                tex10 = startMip2 + min(tex1+1, lenMip2-1) + lenMip2*tex2;
                tex01 = startMip2 + tex1 + lenMip2*min(tex2+1, lenMip2-1);
                tex11 = startMip2 + min(tex1+1, lenMip2-1) + lenMip2*min(tex2+1, lenMip2-1);

                texi1 = 1-texr1;
                texi2 = 1-texr2;

                outR += (ushort)((texi1*texi2*TR[tex] + texr1*texi2*TR[tex10] +
                                   texi1*texr2*TR[tex01] + texr1*texr2*TR[tex11])
                                   * light * fmip * (1-blend));
                outG += (ushort)((texi1*texi2*TG[tex] + texr1*texi2*TG[tex10] +
                                   texi1*texr2*TG[tex01] + texr1*texr2*TG[tex11])
                                   * light * fmip * (1-blend));
                outB += (ushort)((texi1*texi2*TB[tex] + texr1*texi2*TB[tex10] +
                                   texi1*texr2*TB[tex01] + texr1*texr2*TB[tex11])
                                   * light * fmip * (1-blend));
                outR += (ushort)((texi1*texi2*TR2[tex] + texr1*texi2*TR2[tex10] +
                                   texi1*texr2*TR2[tex01] + texr1*texr2*TR2[tex11])
                                   * light * fmip * blend);
                outG += (ushort)((texi1*texi2*TG2[tex] + texr1*texi2*TG2[tex10] +
                                   texi1*texr2*TG2[tex01] + texr1*texr2*TG2[tex11])
                                   * light * fmip * blend);
                outB += (ushort)((texi1*texi2*TB[tex] + texr1*texi2*TB[tex10] +
                                   texi1*texr2*TB[tex01] + texr1*texr2*TB[tex11])
                   * light * fmip * blend);
                Ro[wF * cy + ax] = outR;
                Go[wF * cy + ax] = outG;
                Bo[wF * cy + ax] = outB;

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
    ydiff1 = 1 / (float)(y3-y2);
    slope1 = (x3-x2) * ydiff1;
    slopeu1 = (u3-u2) * ydiff1;
    slopev1 = (v3-v2) * ydiff1;
    slopel1 = (l3-l2) * ydiff1;
    slopez1 = (z3-z2) * ydiff1;
    slopepos1 = (pos3-pos2) * ydiff1;

    ydiff1 = 1 / (float)(y3-y4);
    slope2 = (x3-x4) * ydiff1;
    slopeu2 = (u3-u4) * ydiff1;
    slopev2 = (v3-v4) * ydiff1;
    slopel2 = (l3-l4) * ydiff1;
    slopez2 = (z3-z4) * ydiff1;
    slopepos2 = (pos3-pos4) * ydiff1;

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

    for (int cy = y3; cy >= y2; cy--) {
        for (int ax = (int)cx1; ax <= (int)cx2; ax++) {
            if ((cy >= 0) && (cy < hF) && (ax >= 0) && (ax < wF)) {
              float t = (ax-cx2)/(cx1-cx2);
              t = max((float)0., min((float)1., t));
              float tz = 1 / ((1-t)*cz2 + t*cz1);
              if (F[wF * cy + ax] > tz) {
                F[wF * cy + ax] = tz;

                float3 pos = ((1-t)*cp2 + t*cp1) * tz - SP;
                float depth = dot(pos, SVd);
                int sx = (int)(dot(pos, SVx) * sScale) + wS;
                int sy = (int)(dot(pos, SVy) * -sScale) + wS;
                int shadow = 0;
                if ((sx >= shDist) && (sx < 2*wS-shDist) &&
                    (sy >= shDist) && (sy < 2*wS-shDist)) {
                  if (SD[2*wS * sy + sx] < depth) shadow += 1;
                  if (SD[2*wS * (sy+shDist) + sx] < depth) shadow += 1;
                  if (SD[2*wS * (sy-shDist) + sx] < depth) shadow += 1;
                  if (SD[2*wS * sy + (sx+shDist)] < depth) shadow += 1;
                  if (SD[2*wS * sy + (sx-shDist)] < depth) shadow += 1;
                }
                pos = ((1-t)*cp2 + t*cp1) * tz - SP2;
                depth = dot(pos, SVd2);
                sx = (int)(dot(pos, SVx2) * sScale2) + wS2;
                sy = (int)(dot(pos, SVy2) * -sScale2) + wS2;
                if ((sx >= shDist) && (sx < 2*wS2-shDist) &&
                    (sy >= shDist) && (sy < 2*wS2-shDist)) {
                  if (SD2[2*wS2 * sy + sx] < depth) shadow += 1;
                  if (SD2[2*wS2 * (sy+shDist) + sx] < depth) shadow += 1;
                  if (SD2[2*wS2 * (sy-shDist) + sx] < depth) shadow += 1;
                  if (SD2[2*wS2 * sy + (sx+shDist)] < depth) shadow += 1;
                  if (SD2[2*wS2 * sy + (sx-shDist)] < depth) shadow += 1;
                }

                shadow = min(5, shadow);
                float light = shadow * ambLight + (5-shadow) * ((1-t)*cl2 + t*cl1);
                light *= 0.2f;

                float texr1 = ((1-t)*cu2 + t*cu1) * tz * (lenMip-1);
                int tex1 = (int)texr1;
                texr1 -= tex1;
                tex1 = min(tex1, lenMip-1);
                float texr2 = (lenMip-1) * ((1-t)*cv2 + t*cv1) * tz;
                int tex2 = (int)texr2;
                texr2 -= tex2;
                tex2 = min(tex2, lenMip-1);

                int tex = startMip + tex1 + lenMip*tex2;
                int tex10 = startMip + min(tex1+1, lenMip-1) + lenMip*tex2;
                int tex01 = startMip + tex1 + lenMip*min(tex2+1, lenMip-1);
                int tex11 = startMip + min(tex1+1, lenMip-1) + lenMip*min(tex2+1, lenMip-1);

                float texi1 = 1-texr1;
                float texi2 = 1-texr2;

                pos = ((1-t)*cp2 + t*cp1) * tz;
                float blend = max(0.f, min(1.f, (pos.y - h1) / (h2 - h1)));

                ushort outR = (ushort)((texi1*texi2*TR[tex] + texr1*texi2*TR[tex10] +
                                   texi1*texr2*TR[tex01] + texr1*texr2*TR[tex11])
                                   * light * (1-fmip) * (1-blend));
                ushort outG = (ushort)((texi1*texi2*TG[tex] + texr1*texi2*TG[tex10] +
                                   texi1*texr2*TG[tex01] + texr1*texr2*TG[tex11])
                                   * light * (1-fmip) * (1-blend));
                ushort outB = (ushort)((texi1*texi2*TB[tex] + texr1*texi2*TB[tex10] +
                                   texi1*texr2*TB[tex01] + texr1*texr2*TB[tex11])
                                   * light * (1-fmip) * (1-blend));
                outR += (ushort)((texi1*texi2*TR2[tex] + texr1*texi2*TR2[tex10] +
                                   texi1*texr2*TR2[tex01] + texr1*texr2*TR2[tex11])
                                   * light * (1-fmip) * blend);
                outG += (ushort)((texi1*texi2*TG2[tex] + texr1*texi2*TG2[tex10] +
                                   texi1*texr2*TG2[tex01] + texr1*texr2*TG2[tex11])
                                   * light * (1-fmip) * blend);
                outB += (ushort)((texi1*texi2*TB[tex] + texr1*texi2*TB[tex10] +
                                   texi1*texr2*TB[tex01] + texr1*texr2*TB[tex11])
                                   * light * (1-fmip) * blend);


                texr1 = ((1-t)*cu2 + t*cu1) * tz * (lenMip2-1);
                tex1 = (int)texr1;
                texr1 -= tex1;
                tex1 = min(tex1, lenMip2-1);
                texr2 = (lenMip2-1) * ((1-t)*cv2 + t*cv1) * tz;
                tex2 = (int)texr2;
                texr2 -= tex2;
                tex2 = min(tex2, lenMip2-1);

                tex = startMip2 + tex1 + lenMip2*tex2;
                tex10 = startMip2 + min(tex1+1, lenMip2-1) + lenMip2*tex2;
                tex01 = startMip2 + tex1 + lenMip2*min(tex2+1, lenMip2-1);
                tex11 = startMip2 + min(tex1+1, lenMip2-1) + lenMip2*min(tex2+1, lenMip2-1);

                texi1 = 1-texr1;
                texi2 = 1-texr2;

                outR += (ushort)((texi1*texi2*TR[tex] + texr1*texi2*TR[tex10] +
                                   texi1*texr2*TR[tex01] + texr1*texr2*TR[tex11])
                                   * light * fmip * (1-blend));
                outG += (ushort)((texi1*texi2*TG[tex] + texr1*texi2*TG[tex10] +
                                   texi1*texr2*TG[tex01] + texr1*texr2*TG[tex11])
                                   * light * fmip * (1-blend));
                outB += (ushort)((texi1*texi2*TB[tex] + texr1*texi2*TB[tex10] +
                                   texi1*texr2*TB[tex01] + texr1*texr2*TB[tex11])
                                   * light * fmip * (1-blend));
                outR += (ushort)((texi1*texi2*TR2[tex] + texr1*texi2*TR2[tex10] +
                                   texi1*texr2*TR2[tex01] + texr1*texr2*TR2[tex11])
                                   * light * fmip * blend);
                outG += (ushort)((texi1*texi2*TG2[tex] + texr1*texi2*TG2[tex10] +
                                   texi1*texr2*TG2[tex01] + texr1*texr2*TG2[tex11])
                                   * light * fmip * blend);
                outB += (ushort)((texi1*texi2*TB[tex] + texr1*texi2*TB[tex10] +
                                   texi1*texr2*TB[tex01] + texr1*texr2*TB[tex11])
                   * light * fmip * blend);
                Ro[wF * cy + ax] = outR;
                Go[wF * cy + ax] = outG;
                Bo[wF * cy + ax] = outB;

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
