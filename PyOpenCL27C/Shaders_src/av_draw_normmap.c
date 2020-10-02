// Test normal mapping

!shader_args

:Vertex float2 UV
:Vertex float3 I
:Vertex float3 N
:Vertex float3 PXYZ

:Texture ushort TR TG TB lenT

__global uchar3 *NM, float roughness, char isMetal,
__constant float *Vpos,

__constant float3 *LInt,
__constant float3 *LDir,
__global float *SD, const int wS, const float sScale,
__constant float3 *SV, __constant float *SPos,
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

	vertUV1 *= z1; vertUV2 *= z2; vertUV3 *= z3;
	vertPXYZ1 *= z1; vertPXYZ2 *= z2; vertPXYZ3 *= z3;

    float3 SP = (float3)(SPos[0], SPos[1], SPos[2]);
    float3 VP = (float3)(Vpos[0], Vpos[1], Vpos[2]);
!

!shader_core

:DEPTH_COMPARE

:IF_DEPTH_TEST {

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

	float3 pos = ((1-t)*currvertPXYZ2 + t*currvertPXYZ1) * tz - SP;
	float depth = dot(pos, SV[0]);
	float sfx = (dot(pos, SV[1]) * sScale) + wS;
	float sfy = (dot(pos, SV[2]) * -sScale) + wS;
	float shadow = sampleShadow(depth, sfx, sfy, SD, wS, 0.f);

	float light = 1 - shadow;
	float3 col = ((1-t)*currvertI2 + t*currvertI1) * tz;

	float3 norm = fast_normalize(((1-t)*currvertN2 + t*currvertN1) * tz);

	//float3 tgvec = convert_float3(NM[tex]) / 255.f * 2.f - 1.f;

	float3 tgvec = (texi1*texi2*convert_float3(NM[tex]) +
	                texr1*texi2*convert_float3(NM[tex10]) +
					texi1*texr2*convert_float3(NM[tex01]) +
					texr1*texr2*convert_float3(NM[tex11])) / 255.f * 2.f - 1.f;

	tgvec = fast_normalize(tgvec);
	norm = fast_normalize(tgvec.z * norm + -tgvec.y * tangent + -tgvec.x * bitangent);

	/*pos = ((1-t)*currvertPXYZ2 + t*currvertPXYZ1) * tz;
	float3 a = pos - VP;

	float nd = dot(a, norm);
	float3 refl = fast_normalize(a - 2*nd*norm);
	float theta = max(0.f, dot(refl, -LDir[0]));
	int specPow = 64;
	float sunPow = 3.f;
	float3 spec = pown(theta, specPow) * 256 * light * sunPow;
	spec *= (specPow + 1) / (2 * 3.1416f);


	float3 dirCol = max(0.f, dot(norm, LDir[0])) * LInt[0];

	Ro[wF * cy + ax] = convert_ushort(spec.x + TR[tex] * (light * dirCol.x + col.x));
	Go[wF * cy + ax] = convert_ushort(spec.y + TG[tex] * (light * dirCol.y + col.y));
	Bo[wF * cy + ax] = convert_ushort(spec.z + TB[tex] * (light * dirCol.z + col.z));
*/

	/*Ro[wF * cy + ax] = convert_ushort(spec.x + (texi1*texi2*TR[tex] + texr1*texi2*TR[tex10] +
					   texi1*texr2*TR[tex01] + texr1*texr2*TR[tex11]) * (light * dirCol.x + col.x));
	Go[wF * cy + ax] = convert_ushort(spec.y + (texi1*texi2*TG[tex] + texr1*texi2*TG[tex10] +
					   texi1*texr2*TG[tex01] + texr1*texr2*TG[tex11]) * (light * dirCol.y + col.y));
	Bo[wF * cy + ax] = convert_ushort(spec.z + (texi1*texi2*TB[tex] + texr1*texi2*TB[tex10] +
					   texi1*texr2*TB[tex01] + texr1*texr2*TB[tex11]) * (light * dirCol.z + col.z));
*/

	pos = ((1-t)*currvertPXYZ2 + t*currvertPXYZ1) * tz;
	float3 a = fast_normalize(pos - VP);

	float3 h = fast_normalize(a + LDir[0]);
	float theta = max(0.f, dot(h, norm));

	int specPow = exp2(12*(1.f - roughness));
	float3 spec = pown(theta, specPow) * 256 * 4.f * light * LInt[0];
	spec *= (specPow + 2) / (8 * 3.1416f);

	a = pos - VP;
	float nd = dot(a, norm);
	float3 refl = fast_normalize(a - 2 * nd * norm);
    theta = max(0.f, dot(refl, (float3)(0,1,0)));
	spec += pown(theta, 2) * 128 * (1+isMetal) * (float3)(0.1f,0.3f,0.5f);

	theta = max(0.f, dot(fast_normalize(a), refl));
	spec += pown(theta, (int)half_sqrt((float)specPow)) * 256 *
	                     (float3)(0.2f,0.2f,0.2f);

	if (isMetal) {
		spec.x *= TR[tex]/1024.f;
		spec.y *= TG[tex]/1024.f;
		spec.z *= TB[tex]/1024.f;
		light = 0.f; col = 0.f;
	}

	float3 dirCol = max(0.f, dot(norm, LDir[0])) * LInt[0];

	Ro[wF * cy + ax] = convert_ushort(spec.x + (texi1*texi2*TR[tex] + texr1*texi2*TR[tex10] +
					   texi1*texr2*TR[tex01] + texr1*texr2*TR[tex11]) * (light * dirCol.x + col.x));
	Go[wF * cy + ax] = convert_ushort(spec.y + (texi1*texi2*TG[tex] + texr1*texi2*TG[tex10] +
					   texi1*texr2*TG[tex01] + texr1*texr2*TG[tex11]) * (light * dirCol.y + col.y));
	Bo[wF * cy + ax] = convert_ushort(spec.z + (texi1*texi2*TB[tex] + texr1*texi2*TB[tex10] +
					   texi1*texr2*TB[tex01] + texr1*texr2*TB[tex11]) * (light * dirCol.z + col.z));

}
!