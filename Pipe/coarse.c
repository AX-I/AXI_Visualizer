#define TILE_SIZE 8.f
#define TILE_BUF 64

__kernel void draw(__global int *TO,
                   __global int *F, __global int *FN,
                   __global float2 *P,
                   int wF, int hF, int lenP) {

  // wF and hF should be divided by tile size
  // F => index buffer, FN => num buffer, P => screenpoints

  int bx = get_group_id(0);
  int tx = get_local_id(0);

  if ((bx * BLOCK_SIZE + tx) < lenP) {
    int txd = TO[bx*BLOCK_SIZE + tx];

    int ci = txd * 3;

    float2 xy1 = P[ci] / TILE_SIZE;
    float2 xy2 = P[ci+1] / TILE_SIZE;
    float2 xy3 = P[ci+2] / TILE_SIZE;

    float ytemp; float xtemp;
    float x1 = xy1.x; float x2 = xy2.x; float x3 = xy3.x;
    float y1 = xy1.y; float y2 = xy2.y; float y3 = xy3.y;

    // sort y1<y2<y3
    if (y1 > y2) {
      ytemp = y1; xtemp = x1;
      y1 = y2; x1 = x2;
      y2 = ytemp; x2 = xtemp;
    }
    if (y2 > y3) {
      ytemp = y2; xtemp = x2;
      y2 = y3; x2 = x3;
      y3 = ytemp; x3 = xtemp;
    }
    if (y1 > y2) {
      ytemp = y1; xtemp = x1;
      y1 = y2; x1 = x2;
      y2 = ytemp; x2 = xtemp;
    }


    float x4 = (x1 + ((y2-y1)/(y3-y1)) * (x3-x1));
    float y4 = y2;

    // fill bottom flat triangle
    float slope1 = (x2-x1) / (y2-y1);
    float slope2 = (x4-x1) / (y4-y1);

    float cx1 = x1; float cx2 = x1;

    float slopet;
    if (slope1 < slope2) {
      slopet = slope1;
      slope1 = slope2;
      slope2 = slopet;
    }

    for (int cy = floor(y1); cy <= ceil(y2); cy++) {
        for (int ax = floor(cx2); ax <= ceil(cx1); ax++) {
            if ((cy >= 0) && (cy < hF) && (ax >= 0) && (ax < wF)) {
                int nextI = atomic_inc(&FN[wF * cy + ax]);
                F[(wF * cy + ax) * TILE_BUF + nextI] = txd;
            }
        }
        cx1 += slope1;
        cx2 += slope2;
    }

    // fill top flat triangle
    slope1 = (float)(x3-x2) / (float)(y3-y2);
    slope2 = (float)(x3-x4) / (float)(y3-y4);
    cx1 = x3; cx2 = x3;

    if (slope1 < slope2) {
      slopet = slope1;
      slope1 = slope2;
      slope2 = slopet;
    }

    for (int cy = ceil(y3); cy >= floor(y2); cy--) {
        for (int ax = floor(cx1); ax <= ceil(cx2); ax++) {
            if ((cy >= 0) && (cy < hF) && (ax >= 0) && (ax < wF)) {
                int nextI = atomic_inc(&FN[wF * cy + ax]);
                F[(wF * cy + ax) * TILE_BUF + nextI] = txd;
            }
        }
        cx1 -= slope1;
        cx2 -= slope2;
    }
  }
}
