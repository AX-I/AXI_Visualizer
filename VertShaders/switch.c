// Switch bone nums

__kernel void switchBones(__global char *BN,
                     char s1, char s2,
                     const int lenV) {
    int ix = get_global_id(0);

    if (ix < lenV) {
      if (BN[ix] == s1) {
       BN[ix] = s2;
      }
      else if (BN[ix] == s2) {
       BN[ix] = s1;
      }
    }
}
