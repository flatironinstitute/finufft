#include <utils.h>
#include <finufft_j.h>

/* C++ layer for calling FINUFFT from Julia.
   The ptr to finufft_plan is passed as an "opaque" pointer.
   
   Note our typedefs:
   FLT = double (or float, depending on compilation precision)
   CPX = double complex (or float complex, depending on compilation precision)
   BIGINT = int64

   Make sure you call this library with matching julia types
*/

#ifdef __cplusplus
extern "C" {
#endif

finufft_plan* finufft_plan_alloc()
{
  finufft_plan* plan = (finufft_plan*)malloc(sizeof(finufft_plan));
  return plan;
}

void finufft_plan_free(finufft_plan* plan)
{
  if(plan)
    free(plan);
}

#ifdef __cplusplus
}
#endif
