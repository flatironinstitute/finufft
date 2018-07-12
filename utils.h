#ifndef __UTILS_H__
#define __UTILS_H__

#include <complex>
#include <cuComplex.h>

using namespace std;

typedef double FLT;
typedef complex<double> CPX;
typedef cuDoubleComplex gpuComplex;

#define rand01() ((FLT)rand()/RAND_MAX)
// unif[-1,1]:
#define IMA complex<FLT>(0.0,1.0)
#define randm11() (2*rand01() - (FLT)1.0)
#define crandm11() (randm11() + IMA*randm11())
#define PI (FLT)M_PI
#define M_1_2PI 0.159154943091895336
#define RESCALE(x,N,p) (p ? \
             ((x*M_1_2PI + (x<-PI ? 1.5 : (x>PI ? -0.5 : 0.5)))*N) : \
             (x<0 ? x+N : (x>N ? x-N : x)))
#define max_shared_mem 6000

// jfm timer class
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

FLT relerrtwonorm(int n, CPX* a, CPX* b);
