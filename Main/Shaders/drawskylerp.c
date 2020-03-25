
#define TILE_SIZE 16.f
#define TILE_BUF 128
#define TILE_AREA 256

__kernel void draw(__global int *TO, __global int *TN,
                   __global ushort *Ro, __global ushort *Go, __global ushort *Bo,
                   __global float *F, __global float2 *P, __global float *Z,
                   __global float2 *UV,
__global ushort *TR, __global ushort *TG, __global ushort *TB, const int lT,

                   const int wF, const int hF) {

    int bx = get_group_id(0);
    int tx = get_local_id(0);

    __local float ZBuf[TILE_AREA];
    __local int ZAccess[TILE_AREA];

    int tileX = bx % (int)(wF / TILE_SIZE);
    int tileY = bx / (wF / TILE_SIZE);

    float xMin = (tileX) * TILE_SIZE;
    float xMax = (tileX+1.f) * TILE_SIZE - 1.f;
    float yMin = (tileY) * TILE_SIZE;
    float yMax = (tileY+1.f) * TILE_SIZE - 1.f;

    if (tx < TILE_SIZE) {
        for (int i=0; i < (TILE_SIZE); i++) {
            ZBuf[(int)(tx * TILE_SIZE + i)] = F[(int)(wF * (yMin + tx) + xMin + i)];
            ZAccess[(int)(tx * TILE_SIZE + i)] = 0;
        }
    }

  if (TN[bx] > tx) {

    int txd = TO[bx * TILE_BUF + tx];

  //if (true) {//((txd == 0) && (bx == 1232)) {

    int ci = txd * 3;

    float z1 = Z[ci];
    float z2 = Z[ci+1];
    float z3 = Z[ci+2];

    float2 xy1 = P[ci];
    float2 xy2 = P[ci+1];
    float2 xy3 = P[ci+2];

    float2 vertUV1 = UV[ci+0];
float2 vertUV2 = UV[ci+1];
float2 vertUV3 = UV[ci+2];
//printf("bx %d 1 %f %f 2 %f %f 3 %f %f \n", bx, vertUV1.x, vertUV1.y, vertUV2.x, vertUV2.y, vertUV3.x, vertUV3.y);


    float x1 = xy1.x; float x2 = xy2.x; float x3 = xy3.x;
    float y1 = xy1.y; float y2 = xy2.y; float y3 = xy3.y;

    float ytemp; float xtemp;
    float zt;

    float2 vertUVtemp;


    // sort y1<y2<y3
    if (y1 > y2) {
    ytemp = y1; xtemp = x1; zt = z1; vertUVtemp = vertUV1;
    y1 = y2; x1 = x2; z1 = z2; vertUV1 = vertUV2;
    y2 = ytemp; x2 = xtemp; z2 = zt; vertUV2 = vertUVtemp; }
if (y2 > y3) {
    ytemp = y2; xtemp = x2; zt = z2; vertUVtemp = vertUV2;
    y2 = y3; x2 = x3; z2 = z3; vertUV2 = vertUV3;
    y3 = ytemp; x3 = xtemp; z3 = zt; vertUV3 = vertUVtemp; }
if (y1 > y2) {
    ytemp = y1; xtemp = x1; zt = z1; vertUVtemp = vertUV1;
    y1 = y2; x1 = x2; z1 = z2; vertUV1 = vertUV2;
    y2 = ytemp; x2 = xtemp; z2 = zt; vertUV2 = vertUVtemp; }


    float ydiff1 = (y2-y1)/(y3-y1);
    float x4 = x1 + ydiff1 * (x3 - x1);
    float y4 = y2;
    float z4 = z1 + ydiff1 * (z3 - z1);

    float2 vertUV4 = vertUV1 + ydiff1 * (vertUV3 - vertUV1);



int hT = 2;


    // fill bottom flat triangle
    ydiff1 = 1 / (y2-y1);
    float slope1 = (x2-x1) * ydiff1;
    float slope2 = (x4-x1) * ydiff1;
    float slopez1 = (z2-z1) * ydiff1;
    float slopez2 = (z4-z1) * ydiff1;

    float2 slopevertUV1 = (vertUV2 - vertUV1) * ydiff1;
float2 slopevertUV2 = (vertUV4 - vertUV1) * ydiff1;


    float cx1 = x1; float cx2 = x1;
    float cz1 = z1; float cz2 = z1;

    float2 currvertUV1 = vertUV1; float2 currvertUV2 = vertUV1;


    float slopet;
    if (slope1 < slope2) {
      slopet = slope1; zt = slopez1; vertUVtemp = slopevertUV1;
 slope1 = slope2; slopez1 = slopez2; slopevertUV1 = slopevertUV2;
 slope2 = slopet; slopez2 = zt; slopevertUV2 = vertUVtemp;
    }

    int cy = clamp(y1, yMin, yMax);

    cx1 = x1 + (cy-y1) * slope1;
    cx2 = x1 + (cy-y1) * slope2;
    cz1 = z1 + (cy-y1) * slopez1;
    cz2 = z1 + (cy-y1) * slopez2;

    currvertUV1 = vertUV1 + (cy-y1) * slopevertUV1;
currvertUV2 = vertUV1 + (cy-y1) * slopevertUV2;


    if (y2 > yMin) {
      for (; cy <= clamp(y2, yMin, yMax); cy++) {
        if ((cx2 < xMax) && (cx1 > xMin)) {
          for (int ax = clamp(cx2, xMin, xMax); ax <= clamp(cx1, xMin, xMax); ax++) {
            float t = (ax-cx2)/(cx1-cx2);
            t = max(0.f, min(1.f, t));
            float tz = 1 / ((1-t)*cz2 + t*cz1);


	float2 texr12 = ((1-t)*currvertUV2 + t*currvertUV1);
	//printf("texr %f %f \n", texr12.x, texr12.y);

	float texr1 = texr12.x * lT;
	int tex1 = (int)texr1;
	texr1 -= tex1;
	tex1 = min(tex1, lT-1);

	float texr2 = texr12.y * hT;
	int tex2 = (int)texr2;
	texr2 -= tex2;
	tex2 = min(tex2, hT-1);

	int tex = tex1 + lT*tex2;
	int tex10 = min(tex1+1, lT-1) + lT*tex2;
	int tex01 = tex1 + lT*min(tex2+1, hT-1);
	int tex11 = min(tex1+1, lT-1) + lT*min(tex2+1, hT-1);
	float texi1 = 1-texr1;
	float texi2 = 1-texr2;


	Ro[wF * cy + ax] = TR[tex];//convert_ushort(texi1*texi2*TR[tex] + texr1*texi2*TR[tex10] +
					   //texi1*texr2*TR[tex01] + texr1*texr2*TR[tex11]);
	Go[wF * cy + ax] = TG[tex];//convert_ushort(texi1*texi2*TG[tex] + texr1*texi2*TG[tex10] +
					   //texi1*texr2*TG[tex01] + texr1*texr2*TG[tex11]);
	Bo[wF * cy + ax] = TB[tex];//convert_ushort(texi1*texi2*TB[tex] + texr1*texi2*TB[tex10] +
					   //texi1*texr2*TB[tex01] + texr1*texr2*TB[tex11]);



          }
        }
        cx1 += slope1; cx2 += slope2;
        cz1 += slopez1; cz2 += slopez2;
        currvertUV1 += slopevertUV1; currvertUV2 += slopevertUV2;

      }
    }

    // fill top flat triangle
    ydiff1 = 1 / (y3-y2);
    slope1 = (x3-x2) * ydiff1;
    slope2 = (x3-x4) * ydiff1;
    slopez1 = (z3-z2) * ydiff1;
    slopez2 = (z3-z4) * ydiff1;

     slopevertUV1 = -(vertUV2 - vertUV3) * ydiff1;
 slopevertUV2 = -(vertUV4 - vertUV3) * ydiff1;


    cx1 = x3; cx2 = x3;
    cz1 = z3; cz2 = z3;

    currvertUV1 = vertUV3; currvertUV2 = vertUV3;


    if (slope1 < slope2) {
      slopet = slope1; zt = slopez1; vertUVtemp = slopevertUV1;
 slope1 = slope2; slopez1 = slopez2; slopevertUV1 = slopevertUV2;
 slope2 = slopet; slopez2 = zt; slopevertUV2 = vertUVtemp;
    }

    cy = clamp(y3, yMin, yMax);
    cx1 = x3 + (cy-y3) * slope1;
    cx2 = x3 + (cy-y3) * slope2;
    cz1 = z3 + (cy-y3) * slopez1;
    cz2 = z3 + (cy-y3) * slopez2;

    currvertUV1 = vertUV3 + (cy-y3) * slopevertUV1;
currvertUV2 = vertUV3 + (cy-y3) * slopevertUV2;


    if (y2 < yMax) {
      for (; cy >= clamp(y2, yMin, yMax); cy--) {
        if ((cx1 < xMax) && (cx2 > xMin)) {
          for (int ax = clamp(cx1, xMin, xMax); ax <= clamp(cx2, xMin, xMax); ax++) {
            float t = (ax-cx2)/(cx1-cx2);
            t = max(0.f, min(1.f, t));
            float tz = 1 / ((1-t)*cz2 + t*cz1);


	float2 texr12 = ((1-t)*currvertUV2 + t*currvertUV1);
	//printf("texr %f %f \n", texr12.x, texr12.y);
	float texr1 = texr12.x * lT;
	int tex1 = (int)texr1;
	texr1 -= tex1;
	tex1 = min(tex1, lT-1);

	float texr2 = texr12.y * hT;
	int tex2 = (int)texr2;
	texr2 -= tex2;
	tex2 = min(tex2, hT-1);

	int tex = tex1 + lT*tex2;

	//printf("tex %d \n", tex);


	Ro[wF * cy + ax] = TR[tex];//convert_ushort(texi1*texi2*TR[tex] + texr1*texi2*TR[tex10] +
					   //texi1*texr2*TR[tex01] + texr1*texr2*TR[tex11]);
	Go[wF * cy + ax] = TG[tex];//convert_ushort(texi1*texi2*TG[tex] + texr1*texi2*TG[tex10] +
					   //texi1*texr2*TG[tex01] + texr1*texr2*TG[tex11]);
	Bo[wF * cy + ax] = TB[tex];//convert_ushort(texi1*texi2*TB[tex] + texr1*texi2*TB[tex10] +
					   //texi1*texr2*TB[tex01] + texr1*texr2*TB[tex11]);



          }
        }
        cx1 -= slope1; cx2 -= slope2;
        cz1 -= slopez1; cz2 -= slopez2;
        currvertUV1 -= slopevertUV1; currvertUV2 -= slopevertUV2;

      }
    }

//} // debug
  } // if (TN[bx] > tx)

  if (tx == 0) TN[bx] = 0;
  if (tx < TILE_BUF) TO[bx * TILE_BUF + tx] = -1;

} // __kernel void draw()
