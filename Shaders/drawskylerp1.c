// Texture drawing with linear interpolation

__kernel void drawSky(__global ushort *Ro, __global ushort *Go, __global ushort *Bo,
                      __global int *P, __global float *U, __global float *V,
                      __global ushort *TR, __global ushort *TG, __global ushort *TB,
                      const int wF, const int hF,
                      const int lenP, const int lenT) {

    // P => screenpoints, wF, hF => width, height
    // lenT => size of side (ex 512 x 512)
    
    int bx = get_group_id(0);
    int tx = get_local_id(0);

    if ((bx * BLOCK_SIZE + tx) < lenP) {

    int ti = (bx * BLOCK_SIZE + tx) * 3 * 2;
    int ci = (bx * BLOCK_SIZE + tx) * 3;

    int x1 = P[ti];
    int y1 = P[ti + 1];
    float u1 = U[ci];
    float v1 = V[ci];
    int x2 = P[ti + 2];
    int y2 = P[ti + 3];
    float u2 = U[ci+1];
    float v2 = V[ci+1];
    int x3 = P[ti + 4];
    int y3 = P[ti + 5];
    float u3 = U[ci+2];
    float v3 = V[ci+2];
    int ytemp; int xtemp;
    float ut; float vt;

    // bubble sort y1<y2<y3
    if (y1 > y2) {
      ytemp = y1; xtemp = x1; ut = u1; vt = v1;
      y1 = y2; x1 = x2; u1 = u2; v1 = v2;
      y2 = ytemp; x2 = xtemp; u2 = ut; v2 = vt;
    }
    if (y2 > y3) {
      ytemp = y2; xtemp = x2; ut = u2; vt = v2;
      y2 = y3; x2 = x3; u2 = u3; v2 = v3;
      y3 = ytemp; x3 = xtemp; u3 = ut; v3 = vt;
    }
    if (y1 > y2) {
      ytemp = y1; xtemp = x1; ut = u1; vt = v1;
      y1 = y2; x1 = x2; u1 = u2; v1 = v2;
      y2 = ytemp; x2 = xtemp; u2 = ut; v2 = vt;
    }

    if ((y1 < hF) && (y3 >= 0)) {

    int x4 = (int)(x1 + (float)(y2 - y1)/(float)(y3-y1) * (x3 - x1));
    int y4 = y2;
    float u4 = u1 + (float)(y2 - y1)/(float)(y3-y1) * (u3 - u1);
    float v4 = v1 + (float)(y2 - y1)/(float)(y3-y1) * (v3 - v1);

    // fill bottom flat triangle
    float slope1 = (float)(x2-x1) / (float)(y2-y1);
    float slope2 = (float)(x4-x1) / (float)(y4-y1);
    float slopeu1 = (u2-u1) / (float)(y2-y1);
    float slopev1 = (v2-v1) / (float)(y2-y1);
    float slopeu2 = (u4-u1) / (float)(y4-y1);
    float slopev2 = (v4-v1) / (float)(y4-y1);

    float cx1 = x1; float cx2 = x1;
    float cu1 = u1; float cv1 = v1;
    float cu2 = u1; float cv2 = v1;
    
    float slopet;
    if (slope1 < slope2) {
      slopet = slope1; ut = slopeu1; vt = slopev1;
      slope1 = slope2; slopeu1 = slopeu2; slopev1 = slopev2;
      slope2 = slopet; slopeu2 = ut; slopev2 = vt;
    }
    for (int cy = y1; cy <= y2; cy++) {
        for (int ax = (int)cx2; ax <= (int)cx1; ax++) {
            if ((cy >= 0) && (cy < hF) && (ax >= 0) && (ax < wF)) {
                
                float t = (ax-cx2)/(cx1-cx2);
                t = max((float)0., min((float)1., t));
                float texr1 = ((1-t)*cu2 + t*cu1) * lenT;
                int tex1 = (int)texr1;
                texr1 -= tex1;
                tex1 = min(tex1, lenT-1);
                float texr2 = lenT * ((1-t)*cv2 + t*cv1);
                int tex2 = (int)texr2;
                texr2 -= tex2;
                tex2 = min(tex2, lenT-1);
                int tex = tex1 + lenT*tex2;
                int tex10 = min(tex1+1, lenT-1) + lenT*tex2;
                int tex01 = tex1 + lenT*min(tex2+1, lenT-1);
                int tex11 = min(tex1+1, lenT-1) + lenT*min(tex2+1, lenT-1);
                float texi1 = 1-texr1;
                float texi2 = 1-texr2;
                
                Ro[wF * cy + ax] = (ushort)(texi1*texi2*TR[tex] + texr1*texi2*TR[tex10] +
                                   texi1*texr2*TR[tex01] + texr1*texr2*TR[tex11]);
                Go[wF * cy + ax] = (ushort)(texi1*texi2*TG[tex] + texr1*texi2*TG[tex10] +
                                   texi1*texr2*TG[tex01] + texr1*texr2*TG[tex11]);
                Bo[wF * cy + ax] = (ushort)(texi1*texi2*TB[tex] + texr1*texi2*TB[tex10] +
                                   texi1*texr2*TB[tex01] + texr1*texr2*TB[tex11]);
            }
        }
        cx1 += slope1;
        cx2 += slope2;
        cu1 += slopeu1;
        cv1 += slopev1;
        cu2 += slopeu2;
        cv2 += slopev2;
    }

    // fill top flat triangle
    slope1 = (float)(x3-x2) / (float)(y3-y2);
    slope2 = (float)(x3-x4) / (float)(y3-y4);
    slopeu1 = (u3-u2) / (float)(y3-y2);
    slopev1 = (v3-v2) / (float)(y3-y2);
    slopeu2 = (u3-u4) / (float)(y3-y4);
    slopev2 = (v3-v4) / (float)(y3-y4);
    cx1 = x3; cx2 = x3;
    cu1 = u3; cv1 = v3;
    cu2 = u3; cv2 = v3;
    
    if (slope1 < slope2) {
      slopet = slope1; ut = slopeu1; vt = slopev1;
      slope1 = slope2; slopeu1 = slopeu2; slopev1 = slopev2;
      slope2 = slopet; slopeu2 = ut; slopev2 = vt;
    }
    for (int cy = y3; cy >= y2; cy--) {
        for (int ax = (int)cx1; ax <= (int)cx2; ax++) {
            if ((cy >= 0) && (cy < hF) && (ax >= 0) && (ax < wF)) {
                float t = (ax-cx2)/(cx1-cx2);
                t = max((float)0., min((float)1., t));
                float texr1 = ((1-t)*cu2 + t*cu1) * (lenT);
                int tex1 = (int)texr1;
                texr1 -= tex1;
                tex1 = min(tex1, lenT-1);
                float texr2 = (lenT) * ((1-t)*cv2 + t*cv1);
                int tex2 = (int)texr2;
                texr2 -= tex2;
                tex2 = min(tex2, lenT-1);
                int tex = tex1 + lenT*tex2;
                int tex10 = min(tex1+1, lenT-1) + lenT*tex2;
                int tex01 = tex1 + lenT*min(tex2+1, lenT-1);
                int tex11 = min(tex1+1, lenT-1) + lenT*min(tex2+1, lenT-1);
                float texi1 = 1-texr1;
                float texi2 = 1-texr2;
                
                Ro[wF * cy + ax] = (ushort)(texi1*texi2*TR[tex] + texr1*texi2*TR[tex10] +
                                   texi1*texr2*TR[tex01] + texr1*texr2*TR[tex11]);
                Go[wF * cy + ax] = (ushort)(texi1*texi2*TG[tex] + texr1*texi2*TG[tex10] +
                                   texi1*texr2*TG[tex01] + texr1*texr2*TG[tex11]);
                Bo[wF * cy + ax] = (ushort)(texi1*texi2*TB[tex] + texr1*texi2*TB[tex10] +
                                   texi1*texr2*TB[tex01] + texr1*texr2*TB[tex11]);
            }
        }
        cx1 -= slope1;
        cx2 -= slope2;
        cu1 -= slopeu1;
        cv1 -= slopev1;
        cu2 -= slopeu2;
        cv2 -= slopev2;
    }
    }
    }
}
