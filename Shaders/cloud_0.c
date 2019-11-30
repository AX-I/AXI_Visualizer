// Cloud particles with shadow

#define shDist 1
#define ambLight 0.02f
#define ambLight2 0.4f
#define xopacity 0.9f

#define shmax 12.f

__kernel void ps(__global ushort *Ro, __global ushort *Go, __global ushort *Bo,
                 __global float *F,
                 __global float3 *XYZ,
                 const float avgx, const float avgy, const float avgz,
                 const float opacity, const int size,
                 __constant ushort *TR, const int lenT,
                 __constant float *LInt,
                 const float HR, const float HG, const float HB,
                 __global float *SD, const int wS, const float sScale,
                 __constant float3 *SV, __constant float *SPos,
                 __constant float *Vpos, __constant float3 *VV,
                 const float scale, const int wF, const int hF,
                 const float cAX, const float cAY, const int lenP) {
    
    // Block index
    int bx = get_group_id(0);
    int tx = get_local_id(0);
    
    if ((bx * BLOCK_SIZE + tx) < lenP) {
    int ci = (bx * BLOCK_SIZE + tx);
    float3 x1 = XYZ[ci];
    float3 col = (float3)(256*28);
    //float3 avgpos = (float3)(avgx, avgy, avgz);
    //float pdist = fast_length(x1-avgpos);
    //if (pdist > 70) x1 = avgpos + (x1-avgpos)/pdist*(pdist-70);
    
    float3 vp = (float3)(Vpos[0], Vpos[1], Vpos[2]);
    x1 -= vp;
    float3 Vd = VV[0];
    float3 Vx = VV[1];
    float3 Vy = VV[2];
    float dd = dot(x1, Vd);
    float dx = dot(x1, Vx);
    float dy = dot(x1, Vy);
    dx /= dd; dy /= dd;

    float3 SP = (float3)(SPos[0], SPos[1], SPos[2]);
    float3 SVd = SV[0];
    float3 SVx = SV[1];
    float3 SVy = SV[2];
    
    if ((dd > 0) && (fabs(dx) < cAX) && (fabs(dy) < cAY)) {
      dx = dx * -scale + wF/2;
      dy = dy * scale + hF/2;
      int dix = (int)(dx);
      int diy = (int)(dy);
      int vsize = size * 8 / dd;
      //if (pdist > 60) vsize /= max(1.1f, (pdist-60)/4);
      vsize = min(16, vsize);
      float sopacity = opacity;
      if (vsize <= 1) {
        vsize = 2; sopacity *= min(1.f, size * 8.f / dd);
      }
      float vopacity = sopacity;

      float3 light = (float3)0.f;
      float3 pos = x1 + vp - SP;
      float depth = dot(pos, SVd);
      int sx = (int)(dot(pos, SVx) * sScale) + wS;
      int sy = (int)(dot(pos, SVy) * -sScale) + wS;
      
      float shadow = 0;
      if ((sx >= shDist) && (sx < 2*wS-shDist) &&
          (sy >= shDist) && (sy < 2*wS-shDist)) {
        
        shadow += max(0.f, min(shmax, depth - SD[2*wS * sy + sx])) / shmax;
        shadow += max(0.f, min(shmax, depth - SD[2*wS * (sy+shDist) + sx])) / shmax;
        shadow += max(0.f, min(shmax, depth - SD[2*wS * (sy-shDist) + sx])) / shmax;
        shadow += max(0.f, min(shmax, depth - SD[2*wS * sy + (sx+shDist)])) / shmax;
        shadow += max(0.f, min(shmax, depth - SD[2*wS * sy + (sx-shDist)])) / shmax;
      }
      shadow = sqrt(shadow / 5.f);
      light = 1.f - (1.f-ambLight)*shadow;
      light *= 2*(float3)(LInt[0], LInt[1], LInt[2]);
      float scatter = pow(max(0.f, -dot(fast_normalize(x1), SVd)), 8) * 1.4f;
      scatter += max(0.f, dot(fast_normalize(x1), SVd))*0.8f + ambLight2;
      light *= scatter;
      //light = (1.f - (1.f-ambLight)*shadow) * (float3)(LInt[0], LInt[1], LInt[2]);
      light += (float3)(HR, HG, HB);

      for (int ay = max(0, diy-vsize); ay < min(hF-1, diy+vsize); ay++) {
        for (int ax = max(0, dix-vsize); ax < min(wF-1, dix+vsize); ax++) {
          if (F[wF * ay + ax] > dd) {
              
            float texr1 = (ax - (dix-vsize)) * lenT / (2.f*vsize);
            float texr2 = (ay - (diy-vsize)) * lenT / (2.f*vsize);
            int tex1 = (int)texr1;
            int tex2 = (int)texr2;
            texr1 -= tex1; texr2 -= tex2;
            
            int tex = tex1 + lenT*tex2;
            int tex10 = min(tex1+1, lenT-1) + lenT*tex2;
            int tex01 = tex1 + lenT*min(tex2+1, lenT-1);
            int tex11 = min(tex1+1, lenT-1) + lenT*min(tex2+1, lenT-1);
            float texi1 = 1-texr1;
            float texi2 = 1-texr2;
            
            float u = (texi1*texi2*convert_float3(TR[tex]) +
                        texr1*texi2*convert_float3(TR[tex10]) +
                        texi1*texr2*convert_float3(TR[tex01]) +
                        texr1*texr2*convert_float3(TR[tex11]));
            vopacity = opacity * u / 256.f / 256.f;

            //if (vopacity > 0.1f) {
            //  F[wF * ay + ax] = dd;
              Ro[wF * ay + ax] = vopacity * col.x * light.x + (1-vopacity) * Ro[wF * ay + ax];
              Go[wF * ay + ax] = vopacity * col.y * light.y + (1-vopacity) * Go[wF * ay + ax];
              Bo[wF * ay + ax] = vopacity * col.z * light.z + (1-vopacity) * Bo[wF * ay + ax];
            //}
            //Ro[wF * ay + ax] = vopacity * col.x * light + (1-vopacity) * Ro[wF * ay + ax];
            //Go[wF * ay + ax] = vopacity * col.y * light + (1-vopacity) * Go[wF * ay + ax];
            //Bo[wF * ay + ax] = vopacity * col.z * light + (1-vopacity) * Bo[wF * ay + ax];
          }
        }
      }
    }
    }
}
