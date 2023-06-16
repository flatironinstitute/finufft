// unit tests for utils & utils_precindep modules.

// June 2023: switched to pass-fail tests within the executable (more clear,
// and platform-indep, than having to compare the text output)

#include <finufft/test_defs.h>
using namespace finufft::utils;

int main(int argc, char* argv[])
{
  // test next235even...
  // Barnett 2/9/17, made smaller range 3/28/17. pass-fail 6/16/23
  // The true outputs from {0,1,..,99}:
  const BIGINT next235even_true[100] = {2, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 16, 16, 16, 16, 18, 18, 20, 20, 24, 24, 24, 24, 30, 30, 30, 30, 30, 30, 32, 32, 36, 36, 36, 36, 40, 40, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 50, 50, 54, 54, 54, 54, 60, 60, 60, 60, 60, 60, 64, 64, 64, 64, 72, 72, 72, 72, 72, 72, 72, 72, 80, 80, 80, 80, 80, 80, 80, 80, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 96, 96, 96, 96, 96, 96, 100, 100, 100};
  for (BIGINT n=0;n<100;++n) {
    BIGINT o = next235even(n);
    BIGINT t = next235even_true[n];
    if (o!=t) {
      fprintf(stderr,"next235even(%lld) =\t%lld, error should be %lld!\n",(long long)n, (long long)o, (long long)t);
      return 1;
    }
  }
  
  // various old devel expts and comments...
  //printf("starting huge next235even...\n");   // 1e11 takes 1 sec
  //BIGINT n=(BIGINT)120573851963;
  //printf("next235even(%ld) =\t%ld\n",n,next235even(n));
  //double* a; printf("%g\n",a[0]);  // do deliberate segfault for bash debug!

  // test the vector norms and norm difference routines... pass-fail 6/16/23
  BIGINT M = 1e4;
  std::vector<CPX> a(M), b(M);
  for (BIGINT j=0; j<M; ++j) {
    a[j] = CPX(1.0,0.0);
    b[j] = a[j];
  }
  if (abs(infnorm(M,&a[0]) - 1.0) > EPSILON) return 1;
  if (abs(twonorm(M,&a[0]) - sqrt((FLT)M)) > EPSILON) return 1;
  b[0] = CPX(0.0,0.0);  // perturb b from a
  if (abs(errtwonorm(M,&a[0],&b[0]) - 1.0) > EPSILON) return 1;
  if (abs(relerrtwonorm(M,&a[0],&b[0]) - 1.0/sqrt((FLT)M)) > EPSILON) return 1;

#ifdef SINGLE
  printf("testutilsf passed.\n");
#else
  printf("testutils passed.\n");
#endif
  return 0;
}
