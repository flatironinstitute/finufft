// this is all you must include for the finufft lib...
#include "finufft.h"
#include <complex>

// also needed for this example...
#include <iostream>
#include <iomanip>
#include <vector>
using namespace std;

int main(int argc, char *argv[]){

/* Simple 2D type-1 example of calling the FINUFFT library from C++, using plain
   arrays of C++ complex numbers, with a math test. Double precision version. 

   Compile multithreaded with
   g++ -fopenmp example2d1.cpp -I ../src ../lib-static/libfinufft.a -o example2d1 -lfftw3 -lfftw3_omp -lm
   single core with:
   g++ example2d1.cpp -I ../src ../lib-static/libfinufft.a -o example2d1 -lfftw3 -lm
   
   Usage:  ./example2d1
*/

  int M = 1e6; // number of nonuniform points
  int N = 1e6; // approximate total number of modes (N1*N2)
  double acc = 1e-9; //desired accuracy
  nufft_opts opts; finufft_default_opts(&opts);
  complex<double> I(0.0, 1.0); //the imaginary unit

  //generate random non-uniform points on (x,y) and complex strengths (c):
  vector<double> x(M);
  vector<double> y(M);
  vector<complex<double> > c(M, std::complex<double>(0,0));

  for(int i = 0; i < M; i++){
    x[i] = M_PI*(2*(double)rand()/RAND_MAX-1); //uniform random in [-pi, pi)
    y[i] = M_PI*(2*(double)rand()/RAND_MAX-1); //uniform random in [-pi, pi)

    //each component uniform random in [-1,1]
    c[i] = 2*((double)rand()/RAND_MAX-1) + I*(2*((double)rand()/RAND_MAX)-1); 
  }

  int N1 = round(2.0*sqrt(N));
  int N2 = round(N/N1);
  
  //output array for the Fourier modes
  vector<complex<double> > F(int(N1*N2) + 1, std::complex<double>(0,0));

  //call the NUFFT (with iflag += 1): note N and M are typecast to BIGINT
  int ier = finufft2d1(M,&x[0],&y[0], &c[0], 1, acc, N1, N2, &F[0], opts);

  double n_x = round(0.45*N1); //check the answer for this arbitrary mode
  double n_y = round(-0.35*N2);
  
  complex<double> Ftest(0,0);
  for(int j = 0; j < M; j++){
    Ftest += c[j]*exp(I*(n_x*x[j]+n_y*y[j]));
  }
  
  //indicies in output array for this frequency pair?
  int n_out_x = n_x + (int)N1/2; 
  int n_out_y = n_y + (int)N2/2;
  int indexOut = n_out_x + n_out_y*(N1);  

  //compute inf norm of F 
  double Fmax = 0.0;
  for (int m=0; m<N1*N2; m++) {
    double aF = abs(F[m]);
    if (aF>Fmax) Fmax=aF;
  }

  //compute relative error
  double err = abs(F[indexOut] - Ftest)/Fmax; 

  std::cout << "2D type-1 NUFFT done. ier=" << ier << ", err in F[" << indexOut << "] rel to max(F) is " << std::setprecision(2) << err << std::endl;

  return ier;

}
