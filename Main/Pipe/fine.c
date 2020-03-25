// One block of threads per tile

#define TILE_SIZE 16.f
#define TILE_BUF 128

__kernel void draw(__global int *TO, __global char *TN,
                   __global float *F, __global float2 *P, __global float *Z,
                   int wF, int hF) {

    // TO => index buffer, TN => num buffer
    // F => depth buffer, P => screenpoints, Z => 1/depths

    int bx = get_group_id(0);
    int tx = get_local_id(0);

  if (TN[bx] > tx) {

    int tileX = bx % (int)(wF / TILE_SIZE);
    int tileY = bx / (wF / TILE_SIZE);

    float xMin = (tileX-0.5f) * TILE_SIZE;
    float xMax = (tileX+0.5f) * TILE_SIZE - 1.f;
    float yMin = (tileY-0.5f) * TILE_SIZE;
    float yMax = (tileY+0.5f) * TILE_SIZE - 1.f;

    int txd = TO[bx * TILE_BUF + tx];

    int ci = txd * 3;

    float z1 = Z[ci];
    float z2 = Z[ci+1];
    float z3 = Z[ci+2];

    float2 xy1 = P[ci];
    float2 xy2 = P[ci+1];
    float2 xy3 = P[ci+2];

    float ytemp; float xtemp;
    float zt;
    float x1 = xy1.x; float x2 = xy2.x; float x3 = xy3.x;
    float y1 = xy1.y; float y2 = xy2.y; float y3 = xy3.y;

    // sort y1<y2<y3
    if (y1 > y2) {
      ytemp = y1; xtemp = x1; zt = z1;
      y1 = y2; x1 = x2; z1 = z2;
      y2 = ytemp; x2 = xtemp; z2 = zt;
    }
    if (y2 > y3) {
      ytemp = y2; xtemp = x2; zt = z2;
      y2 = y3; x2 = x3; z2 = z3;
      y3 = ytemp; x3 = xtemp; z3 = zt;
    }
    if (y1 > y2) {
      ytemp = y1; xtemp = x1; zt = z1;
      y1 = y2; x1 = x2; z1 = z2;
      y2 = ytemp; x2 = xtemp; z2 = zt;
    }

    float x4 = (x1 + ((y2 - y1)/(y3-y1)) * (x3-x1));
    float y4 = y2;
    float z4 = z1 + (y2 - y1)/(y3-y1) * (z3 - z1);

    // fill bottom flat triangle
    float slope1 = (x2-x1) / (y2-y1);
    float slope2 = (x4-x1) / (y4-y1);
    float slopez1 = (z2-z1) / (y2-y1);
    float slopez2 = (z4-z1) / (y4-y1);
    float cx1 = x1; float cx2 = x1;
    float cz1 = z1; float cz2 = z1;

    float slopet;
    if (slope1 < slope2) {
      slopet = slope1; zt = slopez1;
      slope1 = slope2; slopez1 = slopez2;
      slope2 = slopet; slopez2 = zt;
    }

	int cy = clamp(y1, yMin, yMax);

	cx1 = x1 + (cy-y1) * slope1;
    cx2 = x1 + (cy-y1) * slope2;
	cz1 = z1 + (cy-y1) * slopez1;
    cz2 = z1 + (cy-y1) * slopez2;

  if (y2 > yMin) {
    for (cy; cy <= clamp(y2, yMin, yMax); cy++) {
	  if ((cx2 < xMax) && (cx1 > xMin)) {
        for (int ax = clamp(cx2, xMin, xMax); ax <= clamp(cx1, xMin, xMax); ax++) {
            float t = (ax-cx2)/(cx1-cx2);
            t = max(0.f, min(1.f, t));
            float tz = 1 / ((1-t)*cz2 + t*cz1);

            if (F[wF * cy + ax] > tz) {
                F[wF * cy + ax] = tz;
            }
        }
      }
        cx1 += slope1;
		cx2 += slope2;
		cz1 += slopez1;
        cz2 += slopez2;
    }
  }

    // fill top flat triangle
    slope1 = (x3-x2) / (y3-y2);
    slope2 = (x3-x4) / (y3-y4);
    slopez1 = (z3-z2) / (y3-y2);
    slopez2 = (z3-z4) / (y3-y4);
    cx1 = x3; cx2 = x3;
    cz1 = z3; cz2 = z3;

    if (slope1 < slope2) {
      slopet = slope1; zt = slopez1;
      slope1 = slope2; slopez1 = slopez2;
      slope2 = slopet; slopez2 = zt;
    }

	cy = clamp(y3, yMin, yMax);
	cx1 = x3 + (cy-y3) * slope1;
    cx2 = x3 + (cy-y3) * slope2;
	cz1 = z3 + (cy-y3) * slopez1;
    cz2 = z3 + (cy-y3) * slopez2;

  if (y2 < yMax) {
    for (cy; cy >= clamp(y2, yMin, yMax); cy--) {
	  if ((cx1 < xMax) && (cx2 > xMin)) {
        for (int ax = clamp(cx1, xMin, xMax); ax <= clamp(cx2, xMin, xMax); ax++) {
            float t = (ax-cx2)/(cx1-cx2);
            t = max(0.f, min(1.f, t));
            float tz = 1 / ((1-t)*cz2 + t*cz1);

            if (F[wF * cy + ax] > tz) {
                F[wF * cy + ax] = tz;
            }
        }
	  }
        cx1 -= slope1;
        cx2 -= slope2;
        cz1 -= slopez1;
        cz2 -= slopez2;
    }
  }
  }
  if (tx == 0) TN[bx] = 0;
  if (tx < TILE_BUF) TO[bx * TILE_BUF + tx] = -1;
}
