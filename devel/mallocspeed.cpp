#include <stdlib.h>
#include <vector>
#include <math.h>

int main(int argc, char* argv[])
/* time some mallocs in C and C++ STL
  
g++ -Ofast -fPIC mallocspeed.cpp -o mallocspeed
time ./mallocspeed 

 Barnett 6/15/17
*/
{
  int N=1e9;
  bool stl = true;
  
  if (stl) {
    std::vector<double> x(N);                    // takes 0.57 sec
    x[N] = 1.0;
    
  } else {
    double *x = (double*)malloc(sizeof(double)*N);  // takes 0.002 sec
    x[N] = 1.0;
    free(x);
  }
  return 0;
}

