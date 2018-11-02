#include "../src/utils.h"
#include <stdio.h>

int main(int argc, char* argv[])
// test next235even. Barnett 2/9/17, made smaller range 3/28/17
{
  for (BIGINT n=90;n<100;++n)
    printf("next235even(%lld) =\t%lld\n",(long long)n,(long long)next235even(n));

  //printf("starting huge next235even...\n");   // 1e11 takes 1 sec
  //BIGINT n=(BIGINT)120573851963;
  //printf("next235even(%ld) =\t%ld\n",n,next235even(n));

  return 0;
}
