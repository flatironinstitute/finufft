#include "cnufftspread.h"

#include <vector>

double rand01();

int main(int argc, char* argv[])
{

    //Q_UNUSED(argc)
    //Q_UNUSED(argv)

    long N=100;
    long M=1e5;

    std::vector<double> d_uniform(N*N*N*2);
    std::vector<double> kx(M),ky(M),kz(M),d_nonuniform(M*2);
    for (long i=0; i<M; i++) {
        kx[i]=rand01()*N;
        ky[i]=rand01()*N;
        kz[i]=rand01()*N;
        d_nonuniform[i*2]=rand01()*2-1;
        d_nonuniform[i*2+1]=rand01()*2-1;
    }

    cnufftspread_opts opts;
    set_kb_opts_from_eps(opts,1e-6);

    CNTime timer; timer.start();
    cnufftspread(N,N,N,d_uniform.data(),M,kx.data(),ky.data(),kz.data(),d_nonuniform.data(),opts);
    printf("Elapsed time for cnufftspread (ms): %d\n",timer.elapsed());

    return 0;
}

double rand01() {
    return (rand()%RAND_MAX)*1.0/RAND_MAX;
}
