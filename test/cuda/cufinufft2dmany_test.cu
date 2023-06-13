#include <cmath>
#include <complex>
#include <helper_cuda.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>

#include <cufinufft/utils.h>
#include <cufinufft_eitherprec.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using cufinufft::utils::infnorm;

int main(int argc, char *argv[]) {
    if (argc != 10) {
        fprintf(stderr, "Usage: cufinufft2d1many_test method type N1 N2 ntransf maxbatchsize M tol checktol\n"
                        "Arguments:\n"
                        "  method: One of\n"
                        "    1: nupts driven,\n"
                        "    2: sub-problem, or\n"
                        "  type: Type of transform (1, 2)"
                        "  N1, N2: The size of the 2D array\n"
                        "  ntransf: Number of inputs\n"
                        "  maxbatchsize: Number of simultaneous transforms (or 0 for default)\n"
                        "  M: The number of non-uniform points\n"
                        "  tol: NUFFT tolerance\n"
                        "  checktol: relative error to pass test\n");
        return 1;
    }
    const int method = atoi(argv[1]);
    const int type = atoi(argv[2]);
    const int N1 = atof(argv[3]);
    const int N2 = atof(argv[4]);
    const int ntransf = atof(argv[5]);
    const int maxbatchsize = atoi(argv[6]);
    const int M = atoi(argv[7]);
    const CUFINUFFT_FLT tol = atof(argv[8]);
    const CUFINUFFT_FLT checktol = atof(argv[9]);

    const int N = N1 * N2;
    const int iflag = 1;
    int ier;

    std::cout << std::scientific << std::setprecision(3);

    printf("#modes = %d, #inputs = %d, #NUpts = %d\n", N, ntransf, M);

    thrust::host_vector<CUFINUFFT_FLT> x(M), y(M);
    thrust::host_vector<CUFINUFFT_CPX> c(M * ntransf), fk(ntransf * N1 * N2);

    thrust::device_vector<CUFINUFFT_FLT> d_x(M), d_y(M);
    thrust::device_vector<CUFINUFFT_CPX> d_c(M * ntransf), d_fk(ntransf * N1 * N2);

    std::default_random_engine eng(1);
    std::uniform_real_distribution<CUFINUFFT_FLT> dist11(-1, 1);
    auto randm11 = [&eng, &dist11]() { return dist11(eng); };

    // Making data
    for (int i = 0; i < M; i++) {
        x[i] = M_PI * randm11(); // x in [-pi,pi)
        y[i] = M_PI * randm11();
    }
    if (type == 1) {
        for (int i = 0; i < ntransf * M; i++) {
            c[i].real(randm11());
            c[i].imag(randm11());
        }
    } else if (type == 2) {
        for (int i = 0; i < ntransf * N1 * N2; i++) {
            fk[i].real(randm11());
            fk[i].imag(randm11());
        }
    } else {
        std::cerr << "Invalid type " << type << " supplied\n";
        return 1;
    }

    d_x = x;
    d_y = y;
    if (type == 1)
        d_c = c;
    else if (type == 2)
        d_fk = fk;

    cudaEvent_t start, stop;
    float milliseconds = 0;
    double totaltime = 0;
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

    // now to the test...
    CUFINUFFT_PLAN dplan;
    int dim = 2;

    // Here we setup our own opts, for gpu_method.
    cufinufft_opts opts;
    ier = CUFINUFFT_DEFAULT_OPTS(type, dim, &opts);
    if (ier != 0) {
        printf("err %d: CUFINUFFT_DEFAULT_OPTS\n", ier);
        return ier;
    }
    opts.gpu_method = method;

    int nmodes[3] = {N1, N2, 1};
    cudaEventRecord(start);
    ier = CUFINUFFT_MAKEPLAN(type, dim, nmodes, iflag, ntransf, tol, maxbatchsize, &dplan, &opts);
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
    ier = CUFINUFFT_SETPTS(M, d_x.data().get(), d_y.data().get(), NULL, 0, NULL, NULL, NULL, dplan);
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
    ier = CUFINUFFT_EXECUTE((CUCPX *)d_c.data().get(), (CUCPX *)d_fk.data().get(), dplan);
    if (ier != 0) {
        printf("err: cufinufft2d_exec\n");
        return ier;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float exec_ms = milliseconds;
    totaltime += milliseconds;
    printf("[time  ] cufinufft exec:\t\t %.3g s\n", milliseconds / 1000);

    cudaEventRecord(start);
    ier = CUFINUFFT_DESTROY(dplan);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    totaltime += milliseconds;
    printf("[time  ] cufinufft destroy:\t\t %.3g s\n", milliseconds / 1000);

    if (type == 1)
        fk = d_fk;
    else if (type == 2)
        c = d_c;

    CUFINUFFT_FLT rel_error = std::numeric_limits<CUFINUFFT_FLT>::max();
    if (type == 1) {
        int i = ntransf - 1;                                // // choose some data to check
        int nt1 = (int)(0.37 * N1), nt2 = (int)(0.26 * N2); // choose some mode index to check
        CUFINUFFT_CPX Ft = CUFINUFFT_CPX(0, 0), J = IMA * (CUFINUFFT_FLT)iflag;
        for (int j = 0; j < M; ++j)
            Ft += c[j + i * M] * exp(J * (nt1 * x[j] + nt2 * y[j])); // crude direct
        int it = N1 / 2 + nt1 + N1 * (N2 / 2 + nt2);                 // index in complex F as 1d array
        rel_error = abs(Ft - fk[it + i * N]) / infnorm(N, fk.data() + i * N);
        printf("[gpu   ] %dth data one mode: rel err in F[%d,%d] is %.3g\n", i, nt1, nt2, rel_error);
    } else if (type == 2) {
        const int t = ntransf - 1;
        CUFINUFFT_CPX *fkstart = fk.data() + t * N1 * N2;
        const CUFINUFFT_CPX *cstart = c.data() + t * M;
        const int jt = M / 2; // check arbitrary choice of one targ pt
        const CUFINUFFT_CPX J = IMA * (CUFINUFFT_FLT)iflag;
        CUFINUFFT_CPX ct = CUFINUFFT_CPX(0, 0);
        int m = 0;
        for (int m2 = -(N2 / 2); m2 <= (N2 - 1) / 2; ++m2) // loop in correct order over F
            for (int m1 = -(N1 / 2); m1 <= (N1 - 1) / 2; ++m1)
                ct += fkstart[m++] * exp(J * (m1 * x[jt] + m2 * y[jt])); // crude direct

        rel_error = abs(cstart[jt] - ct) / infnorm(M, c.data());
        printf("[gpu   ] %dth data one targ: rel err in c[%d] is %.3g\n", t, jt, rel_error);
    }

    printf("[totaltime] %.3g us, speed %.3g NUpts/s\n", totaltime * 1000, M * ntransf / totaltime * 1000);
    printf("\t\t\t\t\t(exec-only thoughput: %.3g NU pts/s)\n", M * ntransf / exec_ms * 1000);
    return std::isnan(rel_error) || rel_error > checktol;
}
