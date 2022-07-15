#include <cmath>
#include <complex>
#include <helper_cuda.h>
#include <iomanip>
#include <iostream>
#include <random>

#include <cufinufft/common.h>
#include <cufinufft/spreadinterp.h>
#include <cufinufft/utils.h>

int main(int argc, char *argv[]) {
    int nf1, nf2, nf3;
    CUFINUFFT_FLT sigma = 2.0;
    int N1, N2, N3, M;
    if (argc < 5) {
        fprintf(stderr, "Usage: interp3d method nupts_distr nf1 nf2 nf3 [M [tol [sort]]]\n"
                        "Arguments:\n"
                        "  method: One of\n"
                        "    1: nupts driven, or\n"
                        "    2: sub-problem.\n"
                        "  nupts_distr: The distribution of the points; one of\n"
                        "    0: uniform, or\n"
                        "    1: concentrated in a small region.\n"
                        "  nf1, nf2, nf3: The size of the 3D array.\n"
                        "  M: The number of non-uniform points (default nf1 * nf2 * nf3 / 8).\n"
                        "  tol: NUFFT tolerance (default 1e-6).\n"
                        "  sort: One of\n"
                        "     0: do not sort the points, or\n"
                        "     1: sort the points (default).\n");
        return 1;
    }
    double w;
    int method;
    sscanf(argv[1], "%d", &method);
    int nupts_distribute;
    sscanf(argv[2], "%d", &nupts_distribute);
    sscanf(argv[3], "%lf", &w);
    nf1 = (int)w; // so can read 1e6 right!
    sscanf(argv[4], "%lf", &w);
    nf2 = (int)w; // so can read 1e6 right!
    sscanf(argv[5], "%lf", &w);
    nf3 = (int)w; // so can read 1e6 right!

    N1 = (int)nf1 / sigma;
    N2 = (int)nf2 / sigma;
    N3 = (int)nf3 / sigma;
    M = N1 * N2 * N3; // let density always be 1
    if (argc > 6) {
        sscanf(argv[6], "%lf", &w);
        M = (int)w; // so can read 1e6 right!
        if (M == 0)
            M = N1 * N2 * N3;
    }

    CUFINUFFT_FLT tol = 1e-6;
    if (argc > 7) {
        sscanf(argv[7], "%lf", &w);
        tol = (CUFINUFFT_FLT)w; // so can read 1e6 right!
    }

    int sort = 1;
    if (argc > 8) {
        sscanf(argv[8], "%d", &sort);
    }
    int ier;

    int ns = std::ceil(-log10(tol / 10.0));

    std::cout << std::scientific << std::setprecision(3);

    CUFINUFFT_FLT *x, *y, *z;
    CUFINUFFT_CPX *c, *fw;
    cudaMallocHost(&x, M * sizeof(CUFINUFFT_FLT));
    cudaMallocHost(&y, M * sizeof(CUFINUFFT_FLT));
    cudaMallocHost(&z, M * sizeof(CUFINUFFT_FLT));
    cudaMallocHost(&c, M * sizeof(CUFINUFFT_CPX));
    cudaMallocHost(&fw, nf1 * nf2 * nf3 * sizeof(CUFINUFFT_CPX));

    CUFINUFFT_FLT *d_x, *d_y, *d_z;
    CUCPX *d_c, *d_fw;
    checkCudaErrors(cudaMalloc(&d_x, M * sizeof(CUFINUFFT_FLT)));
    checkCudaErrors(cudaMalloc(&d_y, M * sizeof(CUFINUFFT_FLT)));
    checkCudaErrors(cudaMalloc(&d_z, M * sizeof(CUFINUFFT_FLT)));
    checkCudaErrors(cudaMalloc(&d_c, M * sizeof(CUCPX)));
    checkCudaErrors(cudaMalloc(&d_fw, nf1 * nf2 * nf3 * sizeof(CUCPX)));

    int dim = 3;
    CUFINUFFT_PLAN dplan = new CUFINUFFT_PLAN_S;
    // Zero out your struct, (sets all pointers to NULL, crucial)
    memset(dplan, 0, sizeof(*dplan));
    ier = CUFINUFFT_DEFAULT_OPTS(2, dim, &(dplan->opts));
    dplan->opts.gpu_method = method;
    dplan->opts.gpu_maxsubprobsize = 1024;
    dplan->opts.gpu_kerevalmeth = 0; // not in cmd-line args
    dplan->opts.gpu_sort = sort;
    dplan->opts.gpu_spreadinterponly = 1;

    // binsize needs to be set here, since SETUP_BINSIZE() is not called in spread,
    // interp only wrappers.
    if (dplan->opts.gpu_method == 1) {
        dplan->opts.gpu_binsizex = 16;
        dplan->opts.gpu_binsizey = 16;
        dplan->opts.gpu_binsizez = 2;
    }
    if (dplan->opts.gpu_method == 2) {
        dplan->opts.gpu_binsizex = 16;
        dplan->opts.gpu_binsizey = 16;
        dplan->opts.gpu_binsizez = 2;
    }
    ier = setup_spreader_for_nufft(dplan->spopts, tol, dplan->opts);

    std::default_random_engine eng(1);
    std::uniform_real_distribution<CUFINUFFT_FLT> dist01(0, 1);
    std::uniform_real_distribution<CUFINUFFT_FLT> dist11(-1, 1);
    auto rand01 = [&eng, &dist01]() { return dist01(eng); };
    auto randm11 = [&eng, &dist11]() { return dist11(eng); };

    switch (nupts_distribute) {
    // Making data
    case 0: // uniform
    {
        for (int i = 0; i < M; i++) {
            x[i] = M_PI * randm11(); // x in [-pi,pi)
            y[i] = M_PI * randm11();
            z[i] = M_PI * randm11();
            // std::cout << x[i] << "," << y[i] << "," << z[i] << std::endl;
        }
    } break;
    case 1: // concentrate on a small region
    {
        for (int i = 0; i < M; i++) {
            x[i] = M_PI * rand01() / (nf1 * 2 / 32); // x in [-pi,pi)
            y[i] = M_PI * rand01() / (nf2 * 2 / 32);
            z[i] = M_PI * rand01() / (nf3 * 2 / 32);
        }
    } break;
    default:
        std::cerr << "error: nupts distr should be 0,1" << std::endl;
        return 1;
    }
    for (int i = 0; i < nf1 * nf2 * nf3; i++) {
        fw[i].real(1.0);
        fw[i].imag(0.0);
    }

    checkCudaErrors(cudaMemcpy(d_x, x, M * sizeof(CUFINUFFT_FLT), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y, y, M * sizeof(CUFINUFFT_FLT), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_z, y, M * sizeof(CUFINUFFT_FLT), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fw, fw, nf1 * nf2 * nf3 * sizeof(CUCPX), cudaMemcpyHostToDevice));

    CNTime timer;
    timer.restart();
    ier = CUFINUFFT_INTERP3D(nf1, nf2, nf3, d_fw, M, d_x, d_y, d_z, d_c, dplan);
    if (ier != 0) {
        std::cout << "error: cnufftinterp3d" << std::endl;
        return 0;
    }
    CUFINUFFT_FLT t = timer.elapsedsec();
    printf("[Method %d] %ld U pts to #%d NU pts in %.3g s (\t%.3g NU pts/s)\n", dplan->opts.gpu_method, nf1 * nf2 * nf3,
           M, t, M / t);
    checkCudaErrors(cudaMemcpy(c, d_c, M * sizeof(CUCPX), cudaMemcpyDeviceToHost));
#ifdef RESULT
    std::cout << "[result-input]" << std::endl;
    for (int j = 0; j < 10; j++) {
        printf(" (%2.3g,%2.3g)", c[j].real(), c[j].imag());
        std::cout << std::endl;
    }
    std::cout << std::endl;
#endif

    cudaFreeHost(x);
    cudaFreeHost(y);
    cudaFreeHost(z);
    cudaFreeHost(c);
    cudaFreeHost(fw);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_c);
    cudaFree(d_fw);
    return 0;
}
