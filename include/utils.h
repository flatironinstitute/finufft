// Header for utils.cpp, a little library of low-level array and timer stuff.
// (rest of finufft defs and types are now in defs.h)
#ifdef T

#include <defs.h>
#include <templates.h>


// ahb's low-level array helpers
T TEMPLATE(relerrtwonorm,T)(BIGINT n, TEMPLATE(CPX,T)* a, TEMPLATE(CPX,T)* b);
T TEMPLATE(errtwonorm,T)(BIGINT n, TEMPLATE(CPX,T)* a, TEMPLATE(CPX,T)* b);
T TEMPLATE(twonorm,T)(BIGINT n, TEMPLATE(CPX,T)* a);
T TEMPLATE(infnorm,T)(BIGINT n, TEMPLATE(CPX,T)* a);
void TEMPLATE(arrayrange,T)(BIGINT n, T* a, T *lo, T *hi);
void TEMPLATE(indexedarrayrange,T)(BIGINT n, BIGINT* i, T* a, T *lo, T *hi);
void TEMPLATE(arraywidcen,T)(BIGINT n, T* a, T *w, T *c);
BIGINT next235even(BIGINT n);

#ifndef ONCE_CNTIME
#define ONCE_CNTIME
// jfm's timer class
#include <sys/time.h>
class CNTime {
 public:
  void start();
  double restart();
  double elapsedsec();
 private:
  struct timeval initial;
};
#endif

#endif //def T

