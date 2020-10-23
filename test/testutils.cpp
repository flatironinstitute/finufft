// tests for utils/utils_precindep module.

// No pass/fail tests are actually done here; check_finufft.sh will check the
// std output against its reference text file.

#include <finufft_eitherprec.h>
#include <utils.h>
#include <utils_precindep.h>
#include <stdio.h>
#include <vector>

int main(int argc, char* argv[])
{
  int ier = 0;
  // test next235even. Barnett 2/9/17, made smaller range 3/28/17  ............
  for (BIGINT n=90;n<100;++n)
    printf("next235even(%lld) =\t%lld\n",(long long)n,(long long)next235even(n));

  // various devel expts and comments...
  //printf("starting huge next235even...\n");   // 1e11 takes 1 sec
  //BIGINT n=(BIGINT)120573851963;
  //printf("next235even(%ld) =\t%ld\n",n,next235even(n));
  //double* a; printf("%g\n",a[0]);  // do deliberate segfault for bash debug!

  // test the vector norms and norm difference routines...
  BIGINT M = 1e4;
  std::vector<CPX> a(M), b(M);
  for (BIGINT j=0; j<M; ++j) {
    a[j] = CPX(1.0,0.0);
    b[j] = CPX(1.0,0.0);
  }
  b[0] = CPX(0.0,0.0);
  printf("relerrtwonorm: %.6g\n", relerrtwonorm(M,&a[0],&b[0]));  // 1/sqrt(M)
  printf("errtwonorm: %.6g\n", errtwonorm(M,&a[0],&b[0]));      // should be 1
  printf("twonorm: %.6g\n", twonorm(M,&a[0]));      // should be sqrt(M)
  printf("infnorm: %.6g\n", infnorm(M,&a[0]));      // should be 1

  // test omp helper which ended up not being used in v2.0 since OMP nested
  // not in the end used...
  int nth_used = get_num_threads_parallel_block();
  // don't report to stdout since won't match testutils{f}.refout
  if (nth_used==0) ier = 1;                         // v crude validation
  
  return ier;
}
