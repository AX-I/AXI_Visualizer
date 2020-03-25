#define WHITE 2.f

float3 ACESFilm(float3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return min(1.f, max(0.f, (x*(a*x+b))/(x*(c*x+d)+e)));
}

float3 ACESFilmLum(float3 x)
{
    float lum = dot(x, (float3)(0.2126f, 0.7152f, 0.0722f));
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return (x / lum) * min(1.f, max(0.f, (lum*(a*lum+b))/(lum*(c*lum+d)+e)));
}

__kernel void g(__global ushort *Ro, __global ushort *Go, __global ushort *Bo,
                const int wF, const int hF, const int BS,
                const int stepW, const int stepH) {

    int bx = get_group_id(0);
    int by = get_group_id(1);
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    int ci = bx * BS + tx;
    int cj = by * BS + ty;

    int h1 = stepH * cj;
    int h2 = stepH * (cj+1);

    for (int cy = h1; cy < min(h2, hF); cy++) {
        for (int cx = ci; cx < wF; cx += stepW) {
            float3 i = (float3)(Ro[wF * cy + cx], Go[wF * cy + cx], Bo[wF * cy + cx]);
            float3 j = i * 8.f / 256.f / 256.f;

            // Reinhard luminance with white point
            j = 2*j; float lum = dot(j, (float3)(0.2126f, 0.7152f, 0.0722f));
            j = j * (1.f + lum / (WHITE*WHITE)) / (lum + 1.f);
            j = sqrt(j);

            // ACES
            //j = 0.5f * ACESFilmLum(j) + 0.5f * ACESFilm(j);
            //j = sqrt(j);


            // Hejl / Burgess-Dawson Filmic
            //j = max(0,j-0.004f);
            //j = (j*(6.2f*j+0.5f))/(j*(6.2f*j+1.7f)+0.06f);

            j *= 255.f;

            Ro[wF * cy + cx] = (ushort)min(255.f, j.x);
            Go[wF * cy + cx] = (ushort)min(255.f, j.y);
            Bo[wF * cy + cx] = (ushort)min(255.f, j.z);
        }
    }
}
