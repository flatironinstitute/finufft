// this is code I was messing with timing using time2d2interp.cpp
// around May 3, 2018, to figure how wrapping was slowing down spreading.

void interp_square_nowrap(FLT *out, FLT *du, FLT *ker1, FLT *ker2, BIGINT i1, BIGINT i2,
                          BIGINT N1, BIGINT N2, int ns)
// *************** don't periodic wrap, avoid ptrs. correct if no NU pts nr edge
{
  out[0] = 0.0;
  out[1] = 0.0;
  if (0) { // plain
    for (int dy = 0; dy < ns; dy++) {
      BIGINT j = N1 * (i2 + dy) + i1;
      for (int dx = 0; dx < ns; dx++) {
        FLT k = ker1[dx] * ker2[dy];
        out[0] += du[2 * j] * k;
        out[1] += du[2 * j + 1] * k;
        ++j;
      }
    }
  } else {
    for (int dy = 0; dy < ns; dy++) {
      BIGINT j = N1 * (i2 + dy) + i1;
      // #pragma omp simd
      for (int dx = 0; dx < ns; dx++) {
        FLT k = ker1[dx] * ker2[dy];
        out[0] += du[2 * j] * k;
        out[1] += du[2 * j + 1] * k;
        ++j;
      }
    }
  }
}
