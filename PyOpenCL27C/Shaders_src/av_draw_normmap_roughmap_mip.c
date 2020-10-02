// Normal + roughness mapping

!shader_args

:Vertex float2 UV
:Vertex float3 I
:Vertex float3 N
:Vertex float3 PXYZ

:Texture ushort TR TG TB numMip
//:Texture ushort TR TG TB lenT

__global uchar3 *NM,
__global uchar *RGH,

__constant float *Vpos,

__constant float3 *LInt,
__constant float3 *LDir,
__global float *SD, const int wS, const float sScale,
__constant float3 *SV, __constant float *SPos,

const float mipBias,
!

!shader_setup

	vertUV1 /= z1; vertUV2 /= z2; vertUV3 /= z3;
	vertPXYZ1 /= z1; vertPXYZ2 /= z2; vertPXYZ3 /= z3;

	float2 dUV1 = vertUV2 - vertUV1;
	float2 dUV2 = vertUV3 - vertUV1;
	float3 dPos1 = vertPXYZ2 - vertPXYZ1;
	float3 dPos2 = vertPXYZ3 - vertPXYZ1;

	float rdet = 1.f / (dUV1.x * dUV2.y - dUV1.y * dUV2.x);

	float3 tangent = fast_normalize((dPos1 * dUV2.y - dPos2 * dUV1.y) * rdet);
	float3 bitangent = fast_normalize((dPos2 * dUV1.x - dPos1 * dUV2.x) * rdet);


	float2 slopeu1 = fabs(dUV1 / (float)((y2-y1)*(y2-y1)+(x2-x1)*(x2-x1)));
	float2 slopeu2 = fabs(dUV2 / (float)((y3-y1)*(y3-y1)+(x3-x1)*(x3-x1)));
	float2 slopeu3 = fabs((vertUV3-vertUV2) / (float)((y3-y2)*(y3-y2)+(x3-x2)*(x3-x2)));

	float dt = max(slopeu1.x, max(slopeu1.y, max(slopeu2.x, slopeu2.y)));
	dt = max(dt, max(slopeu3.x, slopeu3.y));
	float fmip = max(0.f, min((float)(numMip-1), fabs(log2(dt))/2 + mipBias));
	int mip = (int)fmip;
	fmip -= mip;
	int lenMip = 1 << mip;
	int lenMip2 = lenMip << 1;
	int startMip = 0;
	for (int i = 0; i < mip; i++) {
	  startMip += (1 << i) * (1 << i);
	}
    int startMip2 = startMip + (lenMip * lenMip);


	vertUV1 *= z1; vertUV2 *= z2; vertUV3 *= z3;
	vertPXYZ1 *= z1; vertPXYZ2 *= z2; vertPXYZ3 *= z3;

    float3 SP = (float3)(SPos[0], SPos[1], SPos[2]);
    float3 VP = (float3)(Vpos[0], Vpos[1], Vpos[2]);
!

!shader_core

:DEPTH_COMPARE

:IF_DEPTH_TEST {

	F[wF * cy + ax] = tz;

	float2 texr12 = ((1-t)*currvertUV2 + t*currvertUV1) * tz * lenMip;
	int tex1 = (int)texr12.x;
	texr12.x -= tex1;
	tex1 = abs(tex1) & (lenMip - 1);
	int tex2 = (int)texr12.y;
	texr12.y -= tex2;
	tex2 = abs(tex2) & (lenMip - 1);

	int tex = startMip + tex1 + lenMip*tex2;
	int tex10 = startMip + min(tex1+1, lenMip-1) + lenMip*tex2;
	int tex01 = startMip + tex1 + lenMip*min(tex2+1, lenMip-1);
	int tex11 = startMip + min(tex1+1, lenMip-1) + lenMip*min(tex2+1, lenMip-1);
	float texr1 = texr12.x; float texr2 = texr12.y;
	float texi1 = 1-texr1; float texi2 = 1-texr2;

	float outR = texi1*texi2*TR[tex] + texr1*texi2*TR[tex10] +
					 texi1*texr2*TR[tex01] + texr1*texr2*TR[tex11];
	float outG = texi1*texi2*TG[tex] + texr1*texi2*TG[tex10] +
					 texi1*texr2*TG[tex01] + texr1*texr2*TG[tex11];
	float outB = texi1*texi2*TB[tex] + texr1*texi2*TB[tex10] +
				     texi1*texr2*TB[tex01] + texr1*texr2*TB[tex11];
	float3 tgvec = texi1*texi2*convert_float3(NM[tex]) +
	               texr1*texi2*convert_float3(NM[tex10]) +
				   texi1*texr2*convert_float3(NM[tex01]) +
				   texr1*texr2*convert_float3(NM[tex11]);

	texr12 = ((1-t)*currvertUV2 + t*currvertUV1) * tz * lenMip2;
	tex1 = (int)texr12.x;
	texr12.x -= tex1;
	tex1 = abs(tex1) & (lenMip2 - 1);
	tex2 = (int)texr12.y;
	texr12.y -= tex2;
	tex2 = abs(tex2) & (lenMip2 - 1);

	tex = startMip2 + tex1 + lenMip2*tex2;
	tex10 = startMip2 + min(tex1+1, lenMip2-1) + lenMip2*tex2;
	tex01 = startMip2 + tex1 + lenMip2*min(tex2+1, lenMip2-1);
	tex11 = startMip2 + min(tex1+1, lenMip2-1) + lenMip2*min(tex2+1, lenMip2-1);
	texr1 = texr12.x; texr2 = texr12.y;
	texi1 = 1-texr1; texi2 = 1-texr2;

	outR = (1-fmip) * outR + fmip * (texi1*texi2*TR[tex] + texr1*texi2*TR[tex10] +
					 texi1*texr2*TR[tex01] + texr1*texr2*TR[tex11]);
	outG = (1-fmip) * outG + fmip * (texi1*texi2*TG[tex] + texr1*texi2*TG[tex10] +
					 texi1*texr2*TG[tex01] + texr1*texr2*TG[tex11]);
	outB = (1-fmip) * outB + fmip * (texi1*texi2*TB[tex] + texr1*texi2*TB[tex10] +
				 	 texi1*texr2*TB[tex01] + texr1*texr2*TB[tex11]);
	tgvec = (1-fmip) * tgvec + fmip * (texi1*texi2*convert_float3(NM[tex]) +
	               texr1*texi2*convert_float3(NM[tex10]) +
				   texi1*texr2*convert_float3(NM[tex01]) +
				   texr1*texr2*convert_float3(NM[tex11]));

	float3 pos = ((1-t)*currvertPXYZ2 + t*currvertPXYZ1) * tz - SP;
	float depth = dot(pos, SV[0]);
	float sfx = (dot(pos, SV[1]) * sScale) + wS;
	float sfy = (dot(pos, SV[2]) * -sScale) + wS;
	float shadow = sampleShadow(depth, sfx, sfy, SD, wS, 0.f);

	float light = 1 - shadow;
	float3 col = ((1-t)*currvertI2 + t*currvertI1) * tz;

	float3 norm = fast_normalize(((1-t)*currvertN2 + t*currvertN1) * tz);

	//float3 tgvec = convert_float3(NM[tex]) / 255.f * 2.f - 1.f;

	tgvec = tgvec / 255.f * 2.f - 1.f;
	tgvec = fast_normalize(tgvec);
	norm = fast_normalize(tgvec.z * norm + -tgvec.y * tangent + -tgvec.x * bitangent);

	pos = ((1-t)*currvertPXYZ2 + t*currvertPXYZ1) * tz;
	float3 a = pos - VP;
	float3 h = fast_normalize(fast_normalize(a) + LDir[0]);
	float theta = max(0.f, dot(h, norm));

	float rgh = 1.f - RGH[tex]/256.f;
	rgh = exp2(12*rgh);

	int specPow = rgh * ((mip + fmip)*(mip + fmip)) / numMip / numMip;
	float3 spec = pown(theta, specPow) * 256 * light * LInt[0];
	spec *= (specPow + 2) / (8 * 3.1416f);

	// Metallic
	//spec.x *= TR[tex]/1024;
	//spec.y *= TG[tex]/1024;
	//spec.z *= TB[tex]/1024;

	float3 dirCol = max(0.f, dot(norm, LDir[0])) * LInt[0];

	Ro[wF * cy + ax] = convert_ushort(spec.x + outR * (light * dirCol.x + col.x));
	Go[wF * cy + ax] = convert_ushort(spec.y + outG * (light * dirCol.y + col.y));
	Bo[wF * cy + ax] = convert_ushort(spec.z + outB * (light * dirCol.z + col.z));
}
!