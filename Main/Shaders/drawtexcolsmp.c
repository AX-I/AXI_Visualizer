
#define TILE_SIZE 16.f
#define TILE_BUF 128
#define TILE_AREA 256

__kernel void draw(__global int *TO, __global int *TN,
                   __global ushort *Ro, __global ushort *Go, __global ushort *Bo,
                   __global float *F, __global float2 *P, __global float *Z,
                   __global float2 *UV, 
__global float3 *I, 
__global float3 *N, 
__global float3 *PXYZ, 
__constant float3 *LInt,
__constant float3 *LDir,
__global ushort *TR, __global ushort *TG, __global ushort *TB, const int lenT,

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

    int ci = txd * 3;

    float z1 = Z[ci];
    float z2 = Z[ci+1];
    float z3 = Z[ci+2];

    float2 xy1 = P[ci];
    float2 xy2 = P[ci+1];
    float2 xy3 = P[ci+2];

    float2 vertUV1 = UV[ci+0] * z1; 
float2 vertUV2 = UV[ci+1] * z2; 
float2 vertUV3 = UV[ci+2] * z3; 
float3 vertI1 = I[ci+0] * z1; 
float3 vertI2 = I[ci+1] * z2; 
float3 vertI3 = I[ci+2] * z3; 
float3 vertN1 = N[ci+0] * z1; 
float3 vertN2 = N[ci+1] * z2; 
float3 vertN3 = N[ci+2] * z3; 
float3 vertPXYZ1 = PXYZ[ci+0] * z1; 
float3 vertPXYZ2 = PXYZ[ci+1] * z2; 
float3 vertPXYZ3 = PXYZ[ci+2] * z3; 


    float x1 = xy1.x; float x2 = xy2.x; float x3 = xy3.x;
    float y1 = xy1.y; float y2 = xy2.y; float y3 = xy3.y;
    
    float ytemp; float xtemp;
    float zt;

    float2 vertUVtemp; 
float3 vertItemp; 
float3 vertNtemp; 
float3 vertPXYZtemp; 


    // sort y1<y2<y3
    if (y1 > y2) { 
    ytemp = y1; xtemp = x1; zt = z1; vertUVtemp = vertUV1; vertItemp = vertI1; vertNtemp = vertN1; vertPXYZtemp = vertPXYZ1; 
    y1 = y2; x1 = x2; z1 = z2; vertUV1 = vertUV2; vertI1 = vertI2; vertN1 = vertN2; vertPXYZ1 = vertPXYZ2; 
    y2 = ytemp; x2 = xtemp; z2 = zt; vertUV2 = vertUVtemp; vertI2 = vertItemp; vertN2 = vertNtemp; vertPXYZ2 = vertPXYZtemp; }
if (y2 > y3) { 
    ytemp = y2; xtemp = x2; zt = z2; vertUVtemp = vertUV2; vertItemp = vertI2; vertNtemp = vertN2; vertPXYZtemp = vertPXYZ2; 
    y2 = y3; x2 = x3; z2 = z3; vertUV2 = vertUV3; vertI2 = vertI3; vertN2 = vertN3; vertPXYZ2 = vertPXYZ3; 
    y3 = ytemp; x3 = xtemp; z3 = zt; vertUV3 = vertUVtemp; vertI3 = vertItemp; vertN3 = vertNtemp; vertPXYZ3 = vertPXYZtemp; }
if (y1 > y2) { 
    ytemp = y1; xtemp = x1; zt = z1; vertUVtemp = vertUV1; vertItemp = vertI1; vertNtemp = vertN1; vertPXYZtemp = vertPXYZ1; 
    y1 = y2; x1 = x2; z1 = z2; vertUV1 = vertUV2; vertI1 = vertI2; vertN1 = vertN2; vertPXYZ1 = vertPXYZ2; 
    y2 = ytemp; x2 = xtemp; z2 = zt; vertUV2 = vertUVtemp; vertI2 = vertItemp; vertN2 = vertNtemp; vertPXYZ2 = vertPXYZtemp; }

    
    float ydiff1 = (y2-y1)/(y3-y1);
    float x4 = x1 + ydiff1 * (x3 - x1);
    float y4 = y2;
    float z4 = z1 + ydiff1 * (z3 - z1);

    float2 vertUV4 = vertUV1 + ydiff1 * (vertUV3 - vertUV1); 
float3 vertI4 = vertI1 + ydiff1 * (vertI3 - vertI1); 
float3 vertN4 = vertN1 + ydiff1 * (vertN3 - vertN1); 
float3 vertPXYZ4 = vertPXYZ1 + ydiff1 * (vertPXYZ3 - vertPXYZ1); 



    // fill bottom flat triangle
    ydiff1 = 1 / (y2-y1);
    float slope1 = (x2-x1) * ydiff1;
    float slope2 = (x4-x1) * ydiff1;
    float slopez1 = (z2-z1) * ydiff1;
    float slopez2 = (z4-z1) * ydiff1;
    
    float2 slopevertUV1 = (vertUV2 - vertUV1) * ydiff1; 
float2 slopevertUV2 = (vertUV4 - vertUV1) * ydiff1; 
float3 slopevertI1 = (vertI2 - vertI1) * ydiff1; 
float3 slopevertI2 = (vertI4 - vertI1) * ydiff1; 
float3 slopevertN1 = (vertN2 - vertN1) * ydiff1; 
float3 slopevertN2 = (vertN4 - vertN1) * ydiff1; 
float3 slopevertPXYZ1 = (vertPXYZ2 - vertPXYZ1) * ydiff1; 
float3 slopevertPXYZ2 = (vertPXYZ4 - vertPXYZ1) * ydiff1; 

    
    float cx1 = x1; float cx2 = x1;
    float cz1 = z1; float cz2 = z1;

    float2 currvertUV1 = vertUV1; float2 currvertUV2 = vertUV1; 
float3 currvertI1 = vertI1; float3 currvertI2 = vertI1; 
float3 currvertN1 = vertN1; float3 currvertN2 = vertN1; 
float3 currvertPXYZ1 = vertPXYZ1; float3 currvertPXYZ2 = vertPXYZ1; 

    
    float slopet;
    if (slope1 < slope2) {
      slopet = slope1; zt = slopez1; vertUVtemp = slopevertUV1; vertItemp = slopevertI1; vertNtemp = slopevertN1; vertPXYZtemp = slopevertPXYZ1; 
 slope1 = slope2; slopez1 = slopez2; slopevertUV1 = slopevertUV2; slopevertI1 = slopevertI2; slopevertN1 = slopevertN2; slopevertPXYZ1 = slopevertPXYZ2; 
 slope2 = slopet; slopez2 = zt; slopevertUV2 = vertUVtemp; slopevertI2 = vertItemp; slopevertN2 = vertNtemp; slopevertPXYZ2 = vertPXYZtemp; 
    }

    int cy = clamp(y1, yMin, yMax);

    cx1 = x1 + (cy-y1) * slope1;
    cx2 = x1 + (cy-y1) * slope2;
    cz1 = z1 + (cy-y1) * slopez1;
    cz2 = z1 + (cy-y1) * slopez2;

    currvertUV1 = vertUV1 + (cy-y1) * slopevertUV1;
currvertUV2 = vertUV1 + (cy-y1) * slopevertUV2;
currvertI1 = vertI1 + (cy-y1) * slopevertI1;
currvertI2 = vertI1 + (cy-y1) * slopevertI2;
currvertN1 = vertN1 + (cy-y1) * slopevertN1;
currvertN2 = vertN1 + (cy-y1) * slopevertN2;
currvertPXYZ1 = vertPXYZ1 + (cy-y1) * slopevertPXYZ1;
currvertPXYZ2 = vertPXYZ1 + (cy-y1) * slopevertPXYZ2;


    if (y2 > yMin) {
      for (; cy <= clamp(y2, yMin, yMax); cy++) {
        if ((cx2 < xMax) && (cx1 > xMin)) {
          for (int ax = clamp(cx2, xMin, xMax); ax <= clamp(cx1, xMin, xMax); ax++) {
            float t = (ax-cx2)/(cx1-cx2);
            t = max(0.f, min(1.f, t));
            float tz = 1 / ((1-t)*cz2 + t*cz1);
            
            

int localCoord = TILE_SIZE * (cy - yMin) + ax - xMin;

ZAccess[localCoord] = 0;
//barrier(CLK_LOCAL_MEM_FENCE);

int zId = atomic_inc(&ZAccess[localCoord]);

//barrier(CLK_LOCAL_MEM_FENCE);
int maxZid = ZAccess[localCoord];

for (int zr = 0; zr < maxZid; zr++) {
    if (zr == zId) {
        if (ZBuf[localCoord] > tz) ZBuf[localCoord] = tz;
        //barrier(CLK_LOCAL_MEM_FENCE);
    }
}


if (ZBuf[localCoord] >= tz) {

	F[wF * cy + ax] = tz;

	float2 texr12 = ((1-t)*currvertUV2 + t*currvertUV1) * tz * lenT;
	int tex1 = (int)texr12.x;
	texr12.x -= tex1;
	tex1 = abs(tex1) & (lenT - 1);
	int tex2 = (int)texr12.y;
	texr12.y -= tex2;
	tex2 = abs(tex2) & (lenT - 1);

	int tex = tex1 + lenT*tex2;
	int tex10 = min(tex1+1, lenT-1) + lenT*tex2;
	int tex01 = tex1 + lenT*min(tex2+1, lenT-1);
	int tex11 = min(tex1+1, lenT-1) + lenT*min(tex2+1, lenT-1);
	float texr1 = texr12.x; float texr2 = texr12.y;
	float texi1 = 1-texr1; float texi2 = 1-texr2;

	float light = 1;
	float3 col = ((1-t)*currvertI2 + t*currvertI1) * tz;

	float3 norm = fast_normalize(((1-t)*currvertN2 + t*currvertN1) * tz);
	float3 dirCol = max(0.f, dot(norm, LDir[0])) * LInt[0];

	Ro[wF * cy + ax] = convert_ushort((texi1*texi2*TR[tex] + texr1*texi2*TR[tex10] +
					   texi1*texr2*TR[tex01] + texr1*texr2*TR[tex11]) * (light * (dirCol.x + col.x) + (1-light) * col.x));
	Go[wF * cy + ax] = convert_ushort((texi1*texi2*TG[tex] + texr1*texi2*TG[tex10] +
					   texi1*texr2*TG[tex01] + texr1*texr2*TG[tex11]) * (light * (dirCol.y + col.y) + (1-light) * col.y));
	Bo[wF * cy + ax] = convert_ushort((texi1*texi2*TB[tex] + texr1*texi2*TB[tex10] +
					   texi1*texr2*TB[tex01] + texr1*texr2*TB[tex11]) * (light * (dirCol.z + col.z) + (1-light) * col.z));
}


          }
        }
        cx1 += slope1; cx2 += slope2;
        cz1 += slopez1; cz2 += slopez2;
        currvertUV1 += slopevertUV1; currvertUV2 += slopevertUV2; 
currvertI1 += slopevertI1; currvertI2 += slopevertI2; 
currvertN1 += slopevertN1; currvertN2 += slopevertN2; 
currvertPXYZ1 += slopevertPXYZ1; currvertPXYZ2 += slopevertPXYZ2; 

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
 slopevertI1 = -(vertI2 - vertI3) * ydiff1; 
 slopevertI2 = -(vertI4 - vertI3) * ydiff1; 
 slopevertN1 = -(vertN2 - vertN3) * ydiff1; 
 slopevertN2 = -(vertN4 - vertN3) * ydiff1; 
 slopevertPXYZ1 = -(vertPXYZ2 - vertPXYZ3) * ydiff1; 
 slopevertPXYZ2 = -(vertPXYZ4 - vertPXYZ3) * ydiff1; 

    
    cx1 = x3; cx2 = x3;
    cz1 = z3; cz2 = z3;

    currvertUV1 = vertUV3; currvertUV2 = vertUV3; 
currvertI1 = vertI3; currvertI2 = vertI3; 
currvertN1 = vertN3; currvertN2 = vertN3; 
currvertPXYZ1 = vertPXYZ3; currvertPXYZ2 = vertPXYZ3; 


    if (slope1 < slope2) {
      slopet = slope1; zt = slopez1; vertUVtemp = slopevertUV1; vertItemp = slopevertI1; vertNtemp = slopevertN1; vertPXYZtemp = slopevertPXYZ1; 
 slope1 = slope2; slopez1 = slopez2; slopevertUV1 = slopevertUV2; slopevertI1 = slopevertI2; slopevertN1 = slopevertN2; slopevertPXYZ1 = slopevertPXYZ2; 
 slope2 = slopet; slopez2 = zt; slopevertUV2 = vertUVtemp; slopevertI2 = vertItemp; slopevertN2 = vertNtemp; slopevertPXYZ2 = vertPXYZtemp; 
    }

    cy = clamp(y3, yMin, yMax);
    cx1 = x3 + (cy-y3) * slope1;
    cx2 = x3 + (cy-y3) * slope2;
    cz1 = z3 + (cy-y3) * slopez1;
    cz2 = z3 + (cy-y3) * slopez2;

    currvertUV1 = vertUV3 + (cy-y3) * slopevertUV1;
currvertUV2 = vertUV3 + (cy-y3) * slopevertUV2;
currvertI1 = vertI3 + (cy-y3) * slopevertI1;
currvertI2 = vertI3 + (cy-y3) * slopevertI2;
currvertN1 = vertN3 + (cy-y3) * slopevertN1;
currvertN2 = vertN3 + (cy-y3) * slopevertN2;
currvertPXYZ1 = vertPXYZ3 + (cy-y3) * slopevertPXYZ1;
currvertPXYZ2 = vertPXYZ3 + (cy-y3) * slopevertPXYZ2;


    if (y2 < yMax) {
      for (; cy >= clamp(y2, yMin, yMax); cy--) {
        if ((cx1 < xMax) && (cx2 > xMin)) {
          for (int ax = clamp(cx1, xMin, xMax); ax <= clamp(cx2, xMin, xMax); ax++) {
            float t = (ax-cx2)/(cx1-cx2);
            t = max(0.f, min(1.f, t));
            float tz = 1 / ((1-t)*cz2 + t*cz1);
            
            

int localCoord = TILE_SIZE * (cy - yMin) + ax - xMin;

ZAccess[localCoord] = 0;
//barrier(CLK_LOCAL_MEM_FENCE);

int zId = atomic_inc(&ZAccess[localCoord]);

//barrier(CLK_LOCAL_MEM_FENCE);
int maxZid = ZAccess[localCoord];

for (int zr = 0; zr < maxZid; zr++) {
    if (zr == zId) {
        if (ZBuf[localCoord] > tz) ZBuf[localCoord] = tz;
        //barrier(CLK_LOCAL_MEM_FENCE);
    }
}


if (ZBuf[localCoord] >= tz) {

	F[wF * cy + ax] = tz;

	float2 texr12 = ((1-t)*currvertUV2 + t*currvertUV1) * tz * lenT;
	int tex1 = (int)texr12.x;
	texr12.x -= tex1;
	tex1 = abs(tex1) & (lenT - 1);
	int tex2 = (int)texr12.y;
	texr12.y -= tex2;
	tex2 = abs(tex2) & (lenT - 1);

	int tex = tex1 + lenT*tex2;
	int tex10 = min(tex1+1, lenT-1) + lenT*tex2;
	int tex01 = tex1 + lenT*min(tex2+1, lenT-1);
	int tex11 = min(tex1+1, lenT-1) + lenT*min(tex2+1, lenT-1);
	float texr1 = texr12.x; float texr2 = texr12.y;
	float texi1 = 1-texr1; float texi2 = 1-texr2;

	float light = 1;
	float3 col = ((1-t)*currvertI2 + t*currvertI1) * tz;

	float3 norm = fast_normalize(((1-t)*currvertN2 + t*currvertN1) * tz);
	float3 dirCol = max(0.f, dot(norm, LDir[0])) * LInt[0];

	Ro[wF * cy + ax] = convert_ushort((texi1*texi2*TR[tex] + texr1*texi2*TR[tex10] +
					   texi1*texr2*TR[tex01] + texr1*texr2*TR[tex11]) * (light * (dirCol.x + col.x) + (1-light) * col.x));
	Go[wF * cy + ax] = convert_ushort((texi1*texi2*TG[tex] + texr1*texi2*TG[tex10] +
					   texi1*texr2*TG[tex01] + texr1*texr2*TG[tex11]) * (light * (dirCol.y + col.y) + (1-light) * col.y));
	Bo[wF * cy + ax] = convert_ushort((texi1*texi2*TB[tex] + texr1*texi2*TB[tex10] +
					   texi1*texr2*TB[tex01] + texr1*texr2*TB[tex11]) * (light * (dirCol.z + col.z) + (1-light) * col.z));
}

            
          }
        }
        cx1 -= slope1; cx2 -= slope2;
        cz1 -= slopez1; cz2 -= slopez2;
        currvertUV1 -= slopevertUV1; currvertUV2 -= slopevertUV2; 
currvertI1 -= slopevertI1; currvertI2 -= slopevertI2; 
currvertN1 -= slopevertN1; currvertN2 -= slopevertN2; 
currvertPXYZ1 -= slopevertPXYZ1; currvertPXYZ2 -= slopevertPXYZ2; 

      }
    }
  
  } // if (TN[bx] > tx)

  if (tx == 0) TN[bx] = 0;
  if (tx < TILE_BUF) TO[bx * TILE_BUF + tx] = -1;

} // __kernel void draw()




__kernel void drawSmall(__global int *TO,
                        __global ushort *Ro, __global ushort *Go, __global ushort *Bo,
                        __global float *F, __global float2 *P, __global float *Z,
                        __global float2 *UV, 
__global float3 *I, 
__global float3 *N, 
__global float3 *PXYZ, 
__constant float3 *LInt,
__constant float3 *LDir,
__global ushort *TR, __global ushort *TG, __global ushort *TB, const int lenT,

                        const int wF, const int hF, const int lenP) {

    int bx = get_group_id(0);
    int tx = get_local_id(0);

  if ((bx * BLOCK_SIZE + tx) < lenP) {

    int txd = TO[bx * BLOCK_SIZE + tx];

    int ci = txd * 3;

    float z1 = Z[ci];
    float z2 = Z[ci+1];
    float z3 = Z[ci+2];

    float2 xy1 = P[ci];
    float2 xy2 = P[ci+1];
    float2 xy3 = P[ci+2];

    float2 vertUV1 = UV[ci+0] * z1; 
float2 vertUV2 = UV[ci+1] * z2; 
float2 vertUV3 = UV[ci+2] * z3; 
float3 vertI1 = I[ci+0] * z1; 
float3 vertI2 = I[ci+1] * z2; 
float3 vertI3 = I[ci+2] * z3; 
float3 vertN1 = N[ci+0] * z1; 
float3 vertN2 = N[ci+1] * z2; 
float3 vertN3 = N[ci+2] * z3; 
float3 vertPXYZ1 = PXYZ[ci+0] * z1; 
float3 vertPXYZ2 = PXYZ[ci+1] * z2; 
float3 vertPXYZ3 = PXYZ[ci+2] * z3; 


    int x1 = xy1.x; int x2 = xy2.x; int x3 = xy3.x;
    int y1 = xy1.y; int y2 = xy2.y; int y3 = xy3.y;
    
    int ytemp; int xtemp;
    float zt;

    float2 vertUVtemp; 
float3 vertItemp; 
float3 vertNtemp; 
float3 vertPXYZtemp; 


    // sort y1<y2<y3
    if (y1 > y2) { 
    ytemp = y1; xtemp = x1; zt = z1; vertUVtemp = vertUV1; vertItemp = vertI1; vertNtemp = vertN1; vertPXYZtemp = vertPXYZ1; 
    y1 = y2; x1 = x2; z1 = z2; vertUV1 = vertUV2; vertI1 = vertI2; vertN1 = vertN2; vertPXYZ1 = vertPXYZ2; 
    y2 = ytemp; x2 = xtemp; z2 = zt; vertUV2 = vertUVtemp; vertI2 = vertItemp; vertN2 = vertNtemp; vertPXYZ2 = vertPXYZtemp; }
if (y2 > y3) { 
    ytemp = y2; xtemp = x2; zt = z2; vertUVtemp = vertUV2; vertItemp = vertI2; vertNtemp = vertN2; vertPXYZtemp = vertPXYZ2; 
    y2 = y3; x2 = x3; z2 = z3; vertUV2 = vertUV3; vertI2 = vertI3; vertN2 = vertN3; vertPXYZ2 = vertPXYZ3; 
    y3 = ytemp; x3 = xtemp; z3 = zt; vertUV3 = vertUVtemp; vertI3 = vertItemp; vertN3 = vertNtemp; vertPXYZ3 = vertPXYZtemp; }
if (y1 > y2) { 
    ytemp = y1; xtemp = x1; zt = z1; vertUVtemp = vertUV1; vertItemp = vertI1; vertNtemp = vertN1; vertPXYZtemp = vertPXYZ1; 
    y1 = y2; x1 = x2; z1 = z2; vertUV1 = vertUV2; vertI1 = vertI2; vertN1 = vertN2; vertPXYZ1 = vertPXYZ2; 
    y2 = ytemp; x2 = xtemp; z2 = zt; vertUV2 = vertUVtemp; vertI2 = vertItemp; vertN2 = vertNtemp; vertPXYZ2 = vertPXYZtemp; }

    
    float ydiff1 = (y2-y1)/(float)(y3-y1);
    float x4 = x1 + ydiff1 * (x3 - x1);
    float y4 = y2;
    float z4 = z1 + ydiff1 * (z3 - z1);

    float2 vertUV4 = vertUV1 + ydiff1 * (vertUV3 - vertUV1); 
float3 vertI4 = vertI1 + ydiff1 * (vertI3 - vertI1); 
float3 vertN4 = vertN1 + ydiff1 * (vertN3 - vertN1); 
float3 vertPXYZ4 = vertPXYZ1 + ydiff1 * (vertPXYZ3 - vertPXYZ1); 


    // fill bottom flat triangle
    ydiff1 = 1 / (float)(y2-y1);
    float slope1 = (x2-x1) * ydiff1;
    float slope2 = (x4-x1) * ydiff1;
    float slopez1 = (z2-z1) * ydiff1;
    float slopez2 = (z4-z1) * ydiff1;
    
    float2 slopevertUV1 = (vertUV2 - vertUV1) * ydiff1; 
float2 slopevertUV2 = (vertUV4 - vertUV1) * ydiff1; 
float3 slopevertI1 = (vertI2 - vertI1) * ydiff1; 
float3 slopevertI2 = (vertI4 - vertI1) * ydiff1; 
float3 slopevertN1 = (vertN2 - vertN1) * ydiff1; 
float3 slopevertN2 = (vertN4 - vertN1) * ydiff1; 
float3 slopevertPXYZ1 = (vertPXYZ2 - vertPXYZ1) * ydiff1; 
float3 slopevertPXYZ2 = (vertPXYZ4 - vertPXYZ1) * ydiff1; 

    
    float cx1 = x1; float cx2 = x1;
    float cz1 = z1; float cz2 = z1;

    float2 currvertUV1 = vertUV1; float2 currvertUV2 = vertUV1; 
float3 currvertI1 = vertI1; float3 currvertI2 = vertI1; 
float3 currvertN1 = vertN1; float3 currvertN2 = vertN1; 
float3 currvertPXYZ1 = vertPXYZ1; float3 currvertPXYZ2 = vertPXYZ1; 

    
    float slopet;
    if (slope1 < slope2) {
      slopet = slope1; zt = slopez1; vertUVtemp = slopevertUV1; vertItemp = slopevertI1; vertNtemp = slopevertN1; vertPXYZtemp = slopevertPXYZ1; 
 slope1 = slope2; slopez1 = slopez2; slopevertUV1 = slopevertUV2; slopevertI1 = slopevertI2; slopevertN1 = slopevertN2; slopevertPXYZ1 = slopevertPXYZ2; 
 slope2 = slopet; slopez2 = zt; slopevertUV2 = vertUVtemp; slopevertI2 = vertItemp; slopevertN2 = vertNtemp; slopevertPXYZ2 = vertPXYZtemp; 
    }

    int cy = y1;

    for (; cy <= y2; cy++) {
        for (int ax = cx2; ax <= cx1; ax++) {
          if ((cy >= 0) && (cy < hF) && (ax >= 0) && (ax < wF)) {
            float t = (ax-cx2)/(cx1-cx2);
            t = max(0.f, min(1.f, t));
            float tz = 1 / ((1-t)*cz2 + t*cz1);
            
            



if (F[wF * cy + ax] >= tz) {

	F[wF * cy + ax] = tz;

	float2 texr12 = ((1-t)*currvertUV2 + t*currvertUV1) * tz * lenT;
	int tex1 = (int)texr12.x;
	texr12.x -= tex1;
	tex1 = abs(tex1) & (lenT - 1);
	int tex2 = (int)texr12.y;
	texr12.y -= tex2;
	tex2 = abs(tex2) & (lenT - 1);

	int tex = tex1 + lenT*tex2;
	int tex10 = min(tex1+1, lenT-1) + lenT*tex2;
	int tex01 = tex1 + lenT*min(tex2+1, lenT-1);
	int tex11 = min(tex1+1, lenT-1) + lenT*min(tex2+1, lenT-1);
	float texr1 = texr12.x; float texr2 = texr12.y;
	float texi1 = 1-texr1; float texi2 = 1-texr2;

	float light = 1;
	float3 col = ((1-t)*currvertI2 + t*currvertI1) * tz;

	float3 norm = fast_normalize(((1-t)*currvertN2 + t*currvertN1) * tz);
	float3 dirCol = max(0.f, dot(norm, LDir[0])) * LInt[0];

	Ro[wF * cy + ax] = convert_ushort((texi1*texi2*TR[tex] + texr1*texi2*TR[tex10] +
					   texi1*texr2*TR[tex01] + texr1*texr2*TR[tex11]) * (light * (dirCol.x + col.x) + (1-light) * col.x));
	Go[wF * cy + ax] = convert_ushort((texi1*texi2*TG[tex] + texr1*texi2*TG[tex10] +
					   texi1*texr2*TG[tex01] + texr1*texr2*TG[tex11]) * (light * (dirCol.y + col.y) + (1-light) * col.y));
	Bo[wF * cy + ax] = convert_ushort((texi1*texi2*TB[tex] + texr1*texi2*TB[tex10] +
					   texi1*texr2*TB[tex01] + texr1*texr2*TB[tex11]) * (light * (dirCol.z + col.z) + (1-light) * col.z));
}

          }
        }
        cx1 += slope1; cx2 += slope2;
        cz1 += slopez1; cz2 += slopez2;
        currvertUV1 += slopevertUV1; currvertUV2 += slopevertUV2; 
currvertI1 += slopevertI1; currvertI2 += slopevertI2; 
currvertN1 += slopevertN1; currvertN2 += slopevertN2; 
currvertPXYZ1 += slopevertPXYZ1; currvertPXYZ2 += slopevertPXYZ2; 

    }

    // fill top flat triangle
    ydiff1 = 1 / (float)(y3-y2);
    slope1 = (x3-x2) * ydiff1;
    slope2 = (x3-x4) * ydiff1;
    slopez1 = (z3-z2) * ydiff1;
    slopez2 = (z3-z4) * ydiff1;

     slopevertUV1 = -(vertUV2 - vertUV3) * ydiff1; 
 slopevertUV2 = -(vertUV4 - vertUV3) * ydiff1; 
 slopevertI1 = -(vertI2 - vertI3) * ydiff1; 
 slopevertI2 = -(vertI4 - vertI3) * ydiff1; 
 slopevertN1 = -(vertN2 - vertN3) * ydiff1; 
 slopevertN2 = -(vertN4 - vertN3) * ydiff1; 
 slopevertPXYZ1 = -(vertPXYZ2 - vertPXYZ3) * ydiff1; 
 slopevertPXYZ2 = -(vertPXYZ4 - vertPXYZ3) * ydiff1; 

    
    cx1 = x3; cx2 = x3;
    cz1 = z3; cz2 = z3;

    currvertUV1 = vertUV3; currvertUV2 = vertUV3; 
currvertI1 = vertI3; currvertI2 = vertI3; 
currvertN1 = vertN3; currvertN2 = vertN3; 
currvertPXYZ1 = vertPXYZ3; currvertPXYZ2 = vertPXYZ3; 


    if (slope1 < slope2) {
      slopet = slope1; zt = slopez1; vertUVtemp = slopevertUV1; vertItemp = slopevertI1; vertNtemp = slopevertN1; vertPXYZtemp = slopevertPXYZ1; 
 slope1 = slope2; slopez1 = slopez2; slopevertUV1 = slopevertUV2; slopevertI1 = slopevertI2; slopevertN1 = slopevertN2; slopevertPXYZ1 = slopevertPXYZ2; 
 slope2 = slopet; slopez2 = zt; slopevertUV2 = vertUVtemp; slopevertI2 = vertItemp; slopevertN2 = vertNtemp; slopevertPXYZ2 = vertPXYZtemp; 
    }

    cy = y3;
    
    for (; cy >= y2; cy--) {
        for (int ax = cx1; ax <= cx2; ax++) {
          if ((cy >= 0) && (cy < hF) && (ax >= 0) && (ax < wF)) {
            float t = (ax-cx2)/(cx1-cx2);
            t = max(0.f, min(1.f, t));
            float tz = 1 / ((1-t)*cz2 + t*cz1);
            
            



if (F[wF * cy + ax] >= tz) {

	F[wF * cy + ax] = tz;

	float2 texr12 = ((1-t)*currvertUV2 + t*currvertUV1) * tz * lenT;
	int tex1 = (int)texr12.x;
	texr12.x -= tex1;
	tex1 = abs(tex1) & (lenT - 1);
	int tex2 = (int)texr12.y;
	texr12.y -= tex2;
	tex2 = abs(tex2) & (lenT - 1);

	int tex = tex1 + lenT*tex2;
	int tex10 = min(tex1+1, lenT-1) + lenT*tex2;
	int tex01 = tex1 + lenT*min(tex2+1, lenT-1);
	int tex11 = min(tex1+1, lenT-1) + lenT*min(tex2+1, lenT-1);
	float texr1 = texr12.x; float texr2 = texr12.y;
	float texi1 = 1-texr1; float texi2 = 1-texr2;

	float light = 1;
	float3 col = ((1-t)*currvertI2 + t*currvertI1) * tz;

	float3 norm = fast_normalize(((1-t)*currvertN2 + t*currvertN1) * tz);
	float3 dirCol = max(0.f, dot(norm, LDir[0])) * LInt[0];

	Ro[wF * cy + ax] = convert_ushort((texi1*texi2*TR[tex] + texr1*texi2*TR[tex10] +
					   texi1*texr2*TR[tex01] + texr1*texr2*TR[tex11]) * (light * (dirCol.x + col.x) + (1-light) * col.x));
	Go[wF * cy + ax] = convert_ushort((texi1*texi2*TG[tex] + texr1*texi2*TG[tex10] +
					   texi1*texr2*TG[tex01] + texr1*texr2*TG[tex11]) * (light * (dirCol.y + col.y) + (1-light) * col.y));
	Bo[wF * cy + ax] = convert_ushort((texi1*texi2*TB[tex] + texr1*texi2*TB[tex10] +
					   texi1*texr2*TB[tex01] + texr1*texr2*TB[tex11]) * (light * (dirCol.z + col.z) + (1-light) * col.z));
}

          }
        }
        cx1 -= slope1; cx2 -= slope2;
        cz1 -= slopez1; cz2 -= slopez2;
        currvertUV1 -= slopevertUV1; currvertUV2 -= slopevertUV2; 
currvertI1 -= slopevertI1; currvertI2 -= slopevertI2; 
currvertN1 -= slopevertN1; currvertN2 -= slopevertN2; 
currvertPXYZ1 -= slopevertPXYZ1; currvertPXYZ2 -= slopevertPXYZ2; 

    }
  
  } // if ((bx * BLOCK_SIZE + tx) < lenP)

} // __kernel void drawSmall()