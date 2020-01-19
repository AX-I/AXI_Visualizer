# AXI Visualizer Shading Language
# v0.1

def compile_file(fn, fo):
    with open(fn) as j:
        with open(fo, "w") as k:
            k.write(compile_AVSL(j.readlines()))

def compile_AVSL(s):
    global a
    a = {"h":"", "a":"", "t":"", "c":""}
    g = None
    cstart = 0
    for line in s:
        cstart += 1
        if not len(line): continue
        if line[0] == "#":
            a["h"] += line
        elif line == "!shader_args\n": g = "a"
        elif line == "!shader_setup\n": g = "t"
        elif line == "!shader_core\n": g = "c"
        elif line[0] == "!": g = None
        elif g is not None: a[g] += line

    c = "".join(s[cstart:])

    out = a["h"]
    out += template_func.replace("[SHADER_ARGS]", a["a"])
    out += template_setup.replace("[SHADER_SETUP]", a["t"])
    out += template_draw.replace("[SHADER_CORE]", a["c"]) + template_end
    return out

template_func = """
__kernel void drawTex(__global int *TO,
        __global ushort *Ro, __global ushort *Go, __global ushort *Bo,
        __global float *F, __global float2 *P, __global float *Z,
        __global float2 *UV,
        __global float3 *I,
        __global float3 *N,
        __global float3 *PXYZ,
        __constant float3 *LInt, __constant float3 *LDir,
        [SHADER_ARGS]
        const float ambLight,
        __global ushort *TR, __global ushort *TG, __global ushort *TB,
        const char useShadow,
        const int wF, const int hF, const int lenP, const int lenT) {
"""

template_setup = """
    int bx = get_group_id(0);
    int tx = get_local_id(0);

    if ((bx * BLOCK_SIZE + tx) < lenP) {
    int txd = TO[bx*BLOCK_SIZE + tx];
    int ci = txd * 3;

    [SHADER_SETUP]

    float z1 = Z[ci];
    float z2 = Z[ci+1];
    float z3 = Z[ci+2];

    float2 xy1 = P[ci];
    float2 xy2 = P[ci+1];
    float2 xy3 = P[ci+2];
    float2 uv1 = UV[ci] * z1;
    float2 uv2 = UV[ci+1] * z2;
    float2 uv3 = UV[ci+2] * z3;
    float3 l1 = I[ci];
    float3 l2 = I[ci+1];
    float3 l3 = I[ci+2];
    float3 n1 = N[ci] * z1;
    float3 n2 = N[ci+1] * z2;
    float3 n3 = N[ci+2] * z3;
    float3 pos1 = PXYZ[ci] * z1;
    float3 pos2 = PXYZ[ci+1] * z2;
    float3 pos3 = PXYZ[ci+2] * z3;

    float xtemp; float ytemp;
    float2 uvt; float3 lt; float zt;
    float3 post; float3 nt;

    float x1 = xy1.x; float x2 = xy2.x; float x3 = xy3.x;
    float y1 = xy1.y; float y2 = xy2.y; float y3 = xy3.y;

    // bubble sort y1<y2<y3
    if (y1 > y2) {
      ytemp = y1; xtemp = x1; uvt = uv1; lt = l1; nt = n1; post = pos1; zt = z1;
      y1 = y2; x1 = x2; uv1 = uv2; l1 = l2; n1 = n2; pos1 = pos2; z1 = z2;
      y2 = ytemp; x2 = xtemp; uv2 = uvt; l2 = lt; n2 = nt; pos2 = post; z2 = zt;
    }
    if (y2 > y3) {
      ytemp = y2; xtemp = x2; uvt = uv2; lt = l2; nt = n2; post = pos2; zt = z2;
      y2 = y3; x2 = x3; uv2 = uv3; l2 = l3; n2 = n3; pos2 = pos3; z2 = z3;
      y3 = ytemp; x3 = xtemp; uv3 = uvt; l3 = lt; n3 = nt; pos3 = post; z3 = zt;
    }
    if (y1 > y2) {
      ytemp = y1; xtemp = x1; uvt = uv1; lt = l1; nt = n1; post = pos1; zt = z1;
      y1 = y2; x1 = x2; uv1 = uv2; l1 = l2; n1 = n2; pos1 = pos2; z1 = z2;
      y2 = ytemp; x2 = xtemp; uv2 = uvt; l2 = lt; n2 = nt; pos2 = post; z2 = zt;
    }

    if ((y1 < hF) && (y3 >= 0)) {

    float u1 = uv1.x; float u2 = uv2.x; float u3 = uv3.x;
    float v1 = uv1.y; float v2 = uv2.y; float v3 = uv3.y;

    float ydiff1 = (y2 - y1)/(y3-y1);
    float x4 = (x1 + ydiff1 * (x3 - x1));
    float y4 = y2;
    float u4 = u1 + ydiff1 * (u3 - u1);
    float v4 = v1 + ydiff1 * (v3 - v1);
    float3 l4 = l1 + ydiff1 * (l3 - l1);
    float3 n4 = n1 + ydiff1 * (n3 - n1);
    float3 pos4 = pos1 + ydiff1 * (pos3 - pos1);
    float z4 = z1 + ydiff1 * (z3 - z1);
"""

template_draw = """
    // fill bottom flat triangle
    ydiff1 = 1 / (y2-y1);

    float slope1 = (x2-x1) * ydiff1;
    float slopeu1 = (u2-u1) * ydiff1;
    float slopev1 = (v2-v1) * ydiff1;
    float3 slopel1 = (l2-l1) * ydiff1;
    float3 slopen1 = (n2-n1) * ydiff1;
    float3 slopepos1 = (pos2-pos1) * ydiff1;
    float slopez1 = (z2-z1) * ydiff1;

    ydiff1 = 1 / (y4-y1);
    float slope2 = (x4-x1) * ydiff1;
    float slopeu2 = (u4-u1) * ydiff1;
    float slopev2 = (v4-v1) * ydiff1;
    float3 slopel2 = (l4-l1) * ydiff1;
    float3 slopen2 = (n4-n1) * ydiff1;
    float3 slopepos2 = (pos4-pos1) * ydiff1;
    float slopez2 = (z4-z1) * ydiff1;

    float cx1 = x1; float cx2 = x1;
    float cu1 = u1; float cv1 = v1; float3 cl1 = l1;
    float cu2 = u1; float cv2 = v1; float3 cl2 = l1;
    float3 cn1 = n1; float3 cn2 = n1;
    float3 cp1 = pos1; float3 cp2 = pos1;
    float cz1 = z1; float cz2 = z1;

    float slopet; float ut; float vt;
    if (slope1 < slope2) {
      slopet = slope1; ut = slopeu1; vt = slopev1; lt = slopel1; nt = slopen1; post = slopepos1; zt = slopez1;
      slope1 = slope2; slopeu1 = slopeu2; slopev1 = slopev2;  slopen1 = slopen2;
        slopel1 = slopel2; slopepos1 = slopepos2; slopez1 = slopez2;
      slope2 = slopet; slopeu2 = ut; slopev2 = vt; slopel2 = lt; slopen2 = nt; slopepos2 = post; slopez2 = zt;
    }

    for (int cy = y1; cy <= y2; cy++) {
        for (int ax = (int)cx2; ax <= (int)cx1; ax++) {
            if ((cy >= 0) && (cy < hF) && (ax >= 0) && (ax < wF)) {
              float t = (ax-cx2)/(cx1-cx2);
              t = max((float)0.f, min((float)1.f, t));
              float tz = 1 / ((1-t)*cz2 + t*cz1);
              [SHADER_CORE]
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
        cn1 += slopen1;
        cn2 += slopen2;
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
    slopen1 = (n3-n2) * ydiff1;
    slopez1 = (z3-z2) * ydiff1;
    slopepos1 = (pos3-pos2) * ydiff1;

    ydiff1 = 1 / (float)(y3-y4);
    slope2 = (x3-x4) * ydiff1;
    slopeu2 = (u3-u4) * ydiff1;
    slopev2 = (v3-v4) * ydiff1;
    slopel2 = (l3-l4) * ydiff1;
    slopen2 = (n3-n4) * ydiff1;
    slopez2 = (z3-z4) * ydiff1;
    slopepos2 = (pos3-pos4) * ydiff1;

    cx1 = x3; cx2 = x3;
    cu1 = u3; cv1 = v3; cl1 = l3;
    cu2 = u3; cv2 = v3; cl2 = l3;
    cn1 = n3; cn2 = n3;
    cp1 = pos3; cp2 = pos3;
    cz1 = z3; cz2 = z3;

    if (slope1 < slope2) {
      slopet = slope1; ut = slopeu1; vt = slopev1; lt = slopel1; nt = slopen1; post = slopepos1; zt = slopez1;
      slope1 = slope2; slopeu1 = slopeu2; slopev1 = slopev2;  slopen1 = slopen2;
        slopel1 = slopel2; slopepos1 = slopepos2; slopez1 = slopez2;
      slope2 = slopet; slopeu2 = ut; slopev2 = vt; slopel2 = lt; slopen2 = nt; slopepos2 = post; slopez2 = zt;
    }

    for (int cy = y3; cy >= y2; cy--) {
        for (int ax = (int)cx1; ax <= (int)cx2; ax++) {
            if ((cy >= 0) && (cy < hF) && (ax >= 0) && (ax < wF)) {
              float t = (ax-cx2)/(cx1-cx2);
              t = max((float)0., min((float)1., t));
              float tz = 1 / ((1-t)*cz2 + t*cz1);
              [SHADER_CORE]
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
        cn1 -= slopen1;
        cn2 -= slopen2;
        cp1 -= slopepos1;
        cp2 -= slopepos2;
        cz1 -= slopez1;
        cz2 -= slopez2;
    }
    }
    }
"""

template_end = "}"

def compileAll():
    p = "Shaders_src/"
    q = "Shaders/"
    with open(p + "Config.txt") as f:
        for line in f:
            if line[0] != "#":
                n = line.replace("\n","").split(" ")
                compile_file(p + n[0], q + n[1])

if __name__ == "__main__":
    compileAll()
