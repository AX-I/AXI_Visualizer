
!shader_args

:Vertex float2 UV
:Vertex float3 I
:Vertex float3 N
:Vertex float3 PXYZ

float emPow,

!

!shader_setup
!

!shader_core

:DEPTH_COMPARE

:IF_DEPTH_TEST {

    Ro[wF * cy + ax] = convert_ushort_sat(Ro[wF * cy + ax] * emPow);
    Go[wF * cy + ax] = convert_ushort_sat(Go[wF * cy + ax] * emPow);
    Bo[wF * cy + ax] = convert_ushort_sat(Bo[wF * cy + ax] * emPow);

}

!