
#include <cmath>
#include <complex>
#include <cstdio>
#include <helper_cuda.h>
#include <iomanip>
#include <iostream>
#include <random>

#include <cufft.h>

#include <cufinufft.h>
#include <cufinufft/utils.h>
using cufinufft::utils::infnorm;

template <typename T>
int run_test(int method) {
    int N1 = 100;
    int N2 = 100;
    int N = N1 * N2;
    int M1 = N1 * N2;
    int M2 = 2 * N1 * N2;

    T tol = 1e-5;
    int iflag = 1;

    std::cout << std::scientific << std::setprecision(3);
    int ier;

    T *x1, *y1;
    std::complex<T> *c1, *fk1;
    cudaMallocHost(&x1, M1 * sizeof(T));
    cudaMallocHost(&y1, M1 * sizeof(T));
    cudaMallocHost(&c1, M1 * sizeof(std::complex<T>));
    cudaMallocHost(&fk1, N1 * N2 * sizeof(std::complex<T>));

    T *d_x1, *d_y1;
    cuda_complex<T> *d_c1, *d_fk1;
    checkCudaErrors(cudaMalloc(&d_x1, M1 * sizeof(T)));
    checkCudaErrors(cudaMalloc(&d_y1, M1 * sizeof(T)));
    checkCudaErrors(cudaMalloc(&d_c1, M1 * sizeof(cuda_complex<T>)));
    checkCudaErrors(cudaMalloc(&d_fk1, N1 * N2 * sizeof(cuda_complex<T>)));

    T *x2, *y2;
    std::complex<T> *c2, *fk2;
    cudaMallocHost(&x2, M2 * sizeof(T));
    cudaMallocHost(&y2, M2 * sizeof(T));
    cudaMallocHost(&c2, M2 * sizeof(std::complex<T>));
    cudaMallocHost(&fk2, N1 * N2 * sizeof(std::complex<T>));

    T *d_x2, *d_y2;
    cuda_complex<T> *d_c2, *d_fk2;
    checkCudaErrors(cudaMalloc(&d_x2, M2 * sizeof(T)));
    checkCudaErrors(cudaMalloc(&d_y2, M2 * sizeof(T)));
    checkCudaErrors(cudaMalloc(&d_c2, M2 * sizeof(cuda_complex<T>)));
    checkCudaErrors(cudaMalloc(&d_fk2, N1 * N2 * sizeof(cuda_complex<T>)));

    std::default_random_engine eng(1);
    std::uniform_real_distribution<T> dist11(-1, 1);
    auto randm11 = [&eng, &dist11]() { return dist11(eng); };
    
    // Making data
    for (int i = 0; i < M1; i++) {
        x1[i] = M_PI * randm11(); // x in [-pi,pi)
        y1[i] = M_PI * randm11();
        c1[i].real(randm11());
        c1[i].imag(randm11());
    }

    for (int i = 0; i < M2; i++) {
        x2[i] = M_PI * randm11(); // x in [-pi,pi)
        y2[i] = M_PI * randm11();
        c2[i].real(randm11());
        c2[i].imag(randm11());
    }

    checkCudaErrors(cudaMemcpy(d_x1, x1, M1 * sizeof(T), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y1, y1, M1 * sizeof(T), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_c1, c1, M1 * sizeof(cuda_complex<T>), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x2, x2, M2 * sizeof(T), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y2, y2, M2 * sizeof(T), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_c2, c2, M2 * sizeof(cuda_complex<T>), cudaMemcpyHostToDevice));

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
    cufinufft_plan_template<T> dplan;
    int dim = 2;
    int type = 1;

    // Here we setup our own opts, for gpu_method.
    cufinufft_opts opts;
    ier = cufinufft_default_opts(type, dim, &opts);
    if (ier != 0) {
        printf("err %d: CUFINUFFT_DEFAULT_OPTS\n", ier);
        return ier;
    }

    opts.gpu_method = method;

    int nmodes[3];
    int ntransf = 1;
    int maxbatchsize = 1;
    nmodes[0] = N1;
    nmodes[1] = N2;
    nmodes[2] = 1;
    cudaEventRecord(start);
    ier = cufinufft_makeplan_impl<T>(type, dim, nmodes, iflag, ntransf, tol, maxbatchsize, &dplan, &opts);
    if (ier != 0) {
        printf("err: cufinufft2d_plan\n");
        return ier;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    totaltime += milliseconds;
    printf("[time  ] cufinufft plan:\t\t %.3g s\n", milliseconds / 1000);

    cudaEventRecord(start);
    ier = cufinufft_setpts_impl<T>(M1, d_x1, d_y1, NULL, 0, NULL, NULL, NULL, dplan);
    if (ier != 0) {
        printf("err: cufinufft_setpts (set 1)\n");
        return ier;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    totaltime += milliseconds;
    printf("[time  ] cufinufft setNUpts (set 1):\t %.3g s\n", milliseconds / 1000);

    cudaEventRecord(start);
    ier = cufinufft_execute_impl<T>(d_c1, d_fk1, dplan);
    if (ier != 0) {
        printf("err: cufinufft2d1_exec (set 1)\n");
        return ier;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    totaltime += milliseconds;
    float exec_ms = milliseconds;
    printf("[time  ] cufinufft exec (set 1):\t %.3g s\n", milliseconds / 1000);

    cudaEventRecord(start);
    ier = cufinufft_setpts_impl<T>(M2, d_x2, d_y2, NULL, 0, NULL, NULL, NULL, dplan);
    if (ier != 0) {
        printf("err: cufinufft_setpts (set 2)\n");
        return ier;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    totaltime += milliseconds;
    printf("[time  ] cufinufft setNUpts (set 2):\t %.3g s\n", milliseconds / 1000);

    cudaEventRecord(start);
    ier = cufinufft_execute_impl<T>(d_c2, d_fk2, dplan);
    if (ier != 0) {
        printf("err: cufinufft2d1_exec (set 2)\n");
        return ier;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    totaltime += milliseconds;
    exec_ms += milliseconds;
    printf("[time  ] cufinufft exec (set 2):\t %.3g s\n", milliseconds / 1000);

    cudaEventRecord(start);
    ier = cufinufft_destroy_impl<T>(dplan);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    totaltime += milliseconds;
    printf("[time  ] cufinufft destroy:\t\t %.3g s\n", milliseconds / 1000);

    checkCudaErrors(cudaMemcpy(fk1, d_fk1, N1 * N2 * sizeof(cuda_complex<T>), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(fk2, d_fk2, N1 * N2 * sizeof(cuda_complex<T>), cudaMemcpyDeviceToHost));

    printf("[Method %d] (%d+%d) NU pts to %d U pts in %.3g s:      %.3g NU pts/s\n", opts.gpu_method, M1, M2, N1 * N2,
           totaltime / 1000, (M1 + M2) / totaltime * 1000);
    printf("\t\t\t\t\t(exec-only thoughput: %.3g NU pts/s)\n", (M1 + M2) / exec_ms * 1000);

    int nt1 = (int)(0.37 * N1), nt2 = (int)(0.26 * N2); // choose some mode index to check
    std::complex<T> Ft = std::complex<T>(0, 0), J = std::complex<T>(0, iflag);
    for (int j = 0; j < M1; ++j)
        Ft += c1[j] * exp(J * (nt1 * x1[j] + nt2 * y1[j])); // crude direct
    int it = N1 / 2 + nt1 + N1 * (N2 / 2 + nt2);            // index in complex F as 1d array
    //	printf("[gpu   ] one mode: abs err in F[%ld,%ld] is %.3g\n",(int)nt1,(int)nt2,abs(Ft-fk[it]));
    printf("[gpu   ] one mode: rel err in F[%ld,%ld] is %.3g (set 1)\n", (long)nt1, (long)nt2,
           abs(Ft - fk1[it]) / infnorm(N, fk1));
    Ft = std::complex<T>(0, 0);
    for (int j = 0; j < M2; ++j)
        Ft += c2[j] * exp(J * (nt1 * x2[j] + nt2 * y2[j])); // crude direct
    printf("[gpu   ] one mode: rel err in F[%ld,%ld] is %.3g (set 2)\n", (long)nt1, (long)nt2,
           abs(Ft - fk2[it]) / infnorm(N, fk2));

    cudaFreeHost(x1);
    cudaFreeHost(y1);
    cudaFreeHost(c1);
    cudaFreeHost(fk1);
    cudaFreeHost(x2);
    cudaFreeHost(y2);
    cudaFreeHost(c2);
    cudaFreeHost(fk2);
    cudaFree(d_x1);
    cudaFree(d_y1);
    cudaFree(d_c1);
    cudaFree(d_fk1);
    cudaFree(d_x2);
    cudaFree(d_y2);
    cudaFree(d_c2);
    cudaFree(d_fk2);

    // for cuda-memcheck to work
    cudaDeviceReset();

    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: cufinufft2d1nupts_test method\n"
                        "Arguments:\n"
                        "  method: One of\n"
                        "    1: nupts driven,\n"
                        "    2: sub-problem, or\n"
                        "  precision: f or d\n");
        return 1;
    }
    int method;
    sscanf(argv[1], "%d", &method);
    char prec = argv[2][0];

    if (prec == 'f')
        return run_test<float>(method);
    else if (prec == 'd')
        return run_test<double>(method);
    else
        fprintf(stderr, "Invalid precision supplied: %s\n", argv[2]);

    return 1;
}
