#include <dataTypes.h>
#include <finufft_plan.h>
#include <finufftjulia.h>

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

int get_type(finufft_plan* plan)
{
  return plan->type;
}

int get_ntransf(finufft_plan* plan)
{
  return plan->ntrans;
}

int get_ndims(finufft_plan* plan)
{
  return plan->dim;
}

void get_nmodes(finufft_plan* plan, BIGINT* n_modes)
{
  n_modes[0] = plan->ms;
  n_modes[1] = plan->mt;
  n_modes[2] = plan->mu;
}

BIGINT get_nj(finufft_plan* plan)
{
  return plan->nj;
}

BIGINT get_nk(finufft_plan* plan)
{
  return plan->nk;
}

#ifdef __cplusplus
}
#endif
