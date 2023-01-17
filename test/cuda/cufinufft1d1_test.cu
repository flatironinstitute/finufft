#include <cmath>
#include <complex>
#include <helper_cuda.h>
#include <iomanip>
#include <iostream>
#include <random>

#include <cufinufft/utils.h>
#include <cufinufft_eitherprec.h>
using cufinufft::utils::infnorm;

int main(int argc, char *argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Usage: cufinufft1d1_test method N M tol checktol\n"
                        "Arguments:\n"
                        "  method: One of\n"
                        "    1: nupts driven, or\n"
                        "    2: sub-problem\n"
                        "  N: The size of the 1D array\n"
                        "  M: The number of non-uniform points\n"
                        "  tol: NUFFT tolerance.\n"
                        "  checktol: relative error to pass test\n");
        return 1;
    }
    int method = atoi(argv[1]);
    int N = atof(argv[2]);
    int M = atof(argv[3]);
    CUFINUFFT_FLT tol = atof(argv[4]);
    CUFINUFFT_FLT checktol = atof(argv[5]);

    int iflag = 1;

    std::cout << std::scientific << std::setprecision(3);
    int ier;

    CUFINUFFT_FLT *x;
    CUFINUFFT_CPX *c, *fk;
    cudaMallocHost(&x, M * sizeof(CUFINUFFT_FLT));
    cudaMallocHost(&c, M * sizeof(CUFINUFFT_CPX));
    cudaMallocHost(&fk, N * sizeof(CUFINUFFT_CPX));

    CUFINUFFT_FLT *d_x;
    CUCPX *d_c, *d_fk;
    checkCudaErrors(cudaMalloc(&d_x, M * sizeof(CUFINUFFT_FLT)));
    checkCudaErrors(cudaMalloc(&d_c, M * sizeof(CUCPX)));
    checkCudaErrors(cudaMalloc(&d_fk, N * sizeof(CUCPX)));

    std::default_random_engine eng(1);
    std::uniform_real_distribution<CUFINUFFT_FLT> dist11(-1, 1);
    auto randm11 = [&eng, &dist11]() { return dist11(eng); };

    // Making data
    for (int i = 0; i < M; i++) {
        x[i] = M_PI * randm11(); // x in [-pi,pi)
        c[i].real(randm11());
        c[i].imag(randm11());
    }

    checkCudaErrors(cudaMemcpy(d_x, x, M * sizeof(CUFINUFFT_FLT), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_c, c, M * sizeof(CUFINUFFT_CPX), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    float milliseconds = 0;
    float totaltime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warm up CUFFT (is slow, takes around 0.2 sec... )
    cudaEventRecord(start);
    {
        int nf1 = 1;
        cufftHandle fftplan;
        cufftPlan1d(&fftplan, nf1, CUFFT_TYPE, 1);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[time  ] dummy warmup call to CUFFT\t %.3g s\n", milliseconds / 1000);

    // now to our tests...
    CUFINUFFT_PLAN dplan;
    int dim = 1;
    int type = 1;

    // Here we setup our own opts, for gpu_method.
    cufinufft_opts opts;
    ier = CUFINUFFT_DEFAULT_OPTS(type, dim, &opts);
    if (ier != 0) {
        printf("err %d: CUFINUFFT_DEFAULT_OPTS\n", ier);
        return ier;
    }

    opts.gpu_method = method;

    int nmodes[3];
    int ntransf = 1;
    int maxbatchsize = 1;
    nmodes[0] = N;
    nmodes[1] = 1;
    nmodes[2] = 1;
    cudaEventRecord(start);
    ier = CUFINUFFT_MAKEPLAN(type, dim, nmodes, iflag, ntransf, tol, maxbatchsize, &dplan, &opts);
    if (ier != 0) {
        printf("err: cufinufft1d_plan\n");
        return ier;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    totaltime += milliseconds;
    printf("[time  ] cufinufft plan:\t\t %.3g s\n", milliseconds / 1000);

    cudaEventRecord(start);
    ier = CUFINUFFT_SETPTS(M, d_x, NULL, NULL, 0, NULL, NULL, NULL, dplan);
    if (ier != 0) {
        printf("err: cufinufft_setpts\n");
        return ier;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    totaltime += milliseconds;
    printf("[time  ] cufinufft setNUpts:\t\t %.3g s\n", milliseconds / 1000);

    cudaEventRecord(start);
    ier = CUFINUFFT_EXECUTE(d_c, d_fk, dplan);
    if (ier != 0) {
        printf("err: cufinufft1d1_exec\n");
        return ier;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    totaltime += milliseconds;
    float exec_ms = milliseconds;
    printf("[time  ] cufinufft exec:\t\t %.3g s\n", milliseconds / 1000);

    cudaEventRecord(start);
    ier = CUFINUFFT_DESTROY(dplan);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    totaltime += milliseconds;
    printf("[time  ] cufinufft destroy:\t\t %.3g s\n", milliseconds / 1000);

    checkCudaErrors(cudaMemcpy(fk, d_fk, N * sizeof(CUCPX), cudaMemcpyDeviceToHost));

    printf("[Method %d] %d NU pts to %d U pts in %.3g s:      %.3g NU pts/s\n", opts.gpu_method, M, N,
           totaltime / 1000, M / totaltime * 1000);
    printf("\t\t\t\t\t(exec-only thoughput: %.3g NU pts/s)\n", M / exec_ms * 1000);

    int nt1 = (int)(0.37 * N); // choose some mode index to check
    CUFINUFFT_CPX Ft = CUFINUFFT_CPX(0, 0), J = IMA * (CUFINUFFT_FLT)iflag;
    for (int j = 0; j < M; ++j)
        Ft += c[j] * exp(J * (nt1 * x[j])); // crude direct
    int it = N / 2 + nt1;                  // index in complex F as 1d array

    CUFINUFFT_FLT rel_error = abs(Ft - fk[it]) / infnorm(N, fk);
    printf("[gpu   ] one mode: rel err in F[%ld] is %.3g\n", (long)nt1, rel_error);

    cudaFreeHost(x);
    cudaFreeHost(c);
    cudaFreeHost(fk);
    cudaFree(d_x);
    cudaFree(d_c);
    cudaFree(d_fk);

    return rel_error > checktol;
}
