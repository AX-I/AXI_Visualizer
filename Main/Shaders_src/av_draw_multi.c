// 4-way multitexturing
// Grass stone snow sand

!shader_args

:Vertex float2 UV
:Vertex float3 I
:Vertex float3 N
:Vertex float3 PXYZ

__constant float3 *LInt,
__constant float3 *LDir,

:Texture ushort TR TG TB lenT
:Texture ushort TR2 TG2 TB2 lenT2
:Texture ushort TR3 TG3 TB3 lenT3
:Texture ushort TR4 TG4 TB4 lenT4

!

!shader_setup
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

	float light = 1;
	float3 col = ((1-t)*currvertI2 + t*currvertI1) * tz;

	float3 norm = fast_normalize(((1-t)*currvertN2 + t*currvertN1) * tz);
	float3 dirCol = max(0.f, dot(norm, LDir[0])) * LInt[0];

	float t1r = (texi1*texi2*TR[tex] + texr1*texi2*TR[tex10] +
					   texi1*texr2*TR[tex01] + texr1*texr2*TR[tex11]);
	float t1g = (texi1*texi2*TG[tex] + texr1*texi2*TG[tex10] +
					   texi1*texr2*TG[tex01] + texr1*texr2*TG[tex11]);
	float t1b = (texi1*texi2*TB[tex] + texr1*texi2*TB[tex10] +
					   texi1*texr2*TB[tex01] + texr1*texr2*TB[tex11]);

	float t2r = (texi1*texi2*TR2[tex] + texr1*texi2*TR2[tex10] +
					   texi1*texr2*TR2[tex01] + texr1*texr2*TR2[tex11]);
	float t2g = (texi1*texi2*TG2[tex] + texr1*texi2*TG2[tex10] +
					   texi1*texr2*TG2[tex01] + texr1*texr2*TG2[tex11]);
	float t2b = (texi1*texi2*TB2[tex] + texr1*texi2*TB2[tex10] +
					   texi1*texr2*TB2[tex01] + texr1*texr2*TB2[tex11]);

	float t3r = (texi1*texi2*TR3[tex] + texr1*texi2*TR3[tex10] +
					   texi1*texr2*TR3[tex01] + texr1*texr2*TR3[tex11]);
	float t3g = (texi1*texi2*TG3[tex] + texr1*texi2*TG3[tex10] +
					   texi1*texr2*TG3[tex01] + texr1*texr2*TG3[tex11]);
	float t3b = (texi1*texi2*TB3[tex] + texr1*texi2*TB3[tex10] +
				   	   texi1*texr2*TB3[tex01] + texr1*texr2*TB3[tex11]);
	float t4r = (texi1*texi2*TR4[tex] + texr1*texi2*TR4[tex10] +
					   texi1*texr2*TR4[tex01] + texr1*texr2*TR4[tex11]);
	float t4g = (texi1*texi2*TG4[tex] + texr1*texi2*TG4[tex10] +
					   texi1*texr2*TG4[tex01] + texr1*texr2*TG4[tex11]);
	float t4b = (texi1*texi2*TB4[tex] + texr1*texi2*TB4[tex10] +
				   	   texi1*texr2*TB4[tex01] + texr1*texr2*TB4[tex11]);

	float3 cpos = ((1-t)*currvertPXYZ2 + t*currvertPXYZ1) * tz;
	float m = (cpos.y - 15.f) / 4.f;
	m = clamp(m, 0.f, 1.f);

	float ny = max(0.f, fabs(norm.y) - 0.33f) * 1.5f;
	m -= 1.5f * (ny * ny - 0.4f);

	m = clamp(m, 0.f, 1.f);

	float3 q = cpos / 3.f;
	float threshy = 16.f + 2*fabs(q.x - round(q.x)) + 2*fabs(q.z - round(q.z));
	float n = clamp((cpos.y - threshy) / 2.f, 0.f, 4.f);
	n = n * n * n;
	n += 1.5f * (ny * ny - 0.4f);
	n = clamp(n, 0.f, 32.f);

	float o = clamp((5.f - cpos.y) / 4.f, 0.f, 1.f);

	if (m + n + o > 1.f) {
		n = n / (n+m+o);
		m = m / (n+m+o);
		o = o / (n+m+o);
	}
	Ro[wF * cy + ax] = convert_ushort(m*t2r + n*t3r + o*t4r + (1-m-n-o)*t1r) * (light * dirCol.x + col.x);
	Go[wF * cy + ax] = convert_ushort(m*t2g + n*t3g + o*t4g + (1-m-n-o)*t1g) * (light * dirCol.y + col.y);
	Bo[wF * cy + ax] = convert_ushort(m*t2b + n*t3b + o*t4b + (1-m-n-o)*t1b) * (light * dirCol.z + col.z);
}
!