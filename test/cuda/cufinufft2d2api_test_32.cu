/* This test should excercise the API
   close to how a user might use the code

   Note this single precision version changes
   doubles ~~> float    and
   cufinufft_* ~~> cufinufftf_* function names.

*/

#include <cmath>
#include <complex>
#include <helper_cuda.h>
#include <iomanip>
#include <iostream>
#include <random>

#include <cufinufft.h>
#include <cufinufft/utils.h>
#include <cufinufft/profile.h>
using cufinufft::utils::infnorm;


int main(int argc, char *argv[]) {
    using complex_float = std::complex<float>;

    int N1 = 256;
    int N2 = 256;
    int M = N1 * N2;

    float tol = 1e-6;

    int iflag = 1;

    std::cout << std::scientific << std::setprecision(3);
    int ier;

    // malloc host arrays
    float *x, *y;
    complex_float *c, *fk;
    checkCudaErrors(cudaMallocHost(&x, M * sizeof(float)));
    checkCudaErrors(cudaMallocHost(&y, M * sizeof(float)));
    checkCudaErrors(cudaMallocHost(&c, M * sizeof(complex_float)));
    checkCudaErrors(cudaMallocHost(&fk, N1 * N2 * sizeof(complex_float)));

    // malloc device arrays
    float *d_x, *d_y;
    cuFloatComplex *d_c, *d_fk;
    checkCudaErrors(cudaMalloc(&d_x, M * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_y, M * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_c, M * sizeof(cuFloatComplex)));
    checkCudaErrors(cudaMalloc(&d_fk, N1 * N2 * sizeof(cuFloatComplex)));

    std::default_random_engine eng(1);
    std::uniform_real_distribution<CUFINUFFT_FLT> dist11(-1, 1);
    auto randm11 = [&eng, &dist11]() { return dist11(eng); };
    
    // Making data
    for (int i = 0; i < M; i++) {
        x[i] = M_PI * randm11(); // x in [-pi,pi)
        y[i] = M_PI * randm11();
    }
    for (int i = 0; i < N1 * N2; i++) {
        fk[i].real(1.0);
        fk[i].imag(1.0);
    }

    // Copy data to device memory, real users might just populate in memory.
    checkCudaErrors(cudaMemcpy(d_x, x, M * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y, y, M * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fk, fk, N1 * N2 * sizeof(complex_float), cudaMemcpyHostToDevice));

    // construct plan
    cufinufftf_plan dplan;
    int dim = 2;
    int type = 2;

    int nmodes[3];
    int ntransf = 1;
    int maxbatchsize = 1;
    nmodes[0] = N1;
    nmodes[1] = N2;
    nmodes[2] = 1;

    ier = cufinufftf_makeplan(type, dim, nmodes, iflag, ntransf, tol, maxbatchsize, &dplan, NULL);
    if (ier != 0) {
        printf("err: cufinufft2d_plan\n");
        return ier;
    }

    // Set Non uniform points
    ier = cufinufftf_setpts(M, d_x, d_y, NULL, 0, NULL, NULL, NULL, dplan);
    if (ier != 0) {
        printf("err: cufinufft_setpts\n");
        return ier;
    }

    // Execute the plan on the data
    ier = cufinufftf_execute(d_c, d_fk, dplan);
    if (ier != 0) {
        printf("err: cufinufft2d2_exec\n");
        return ier;
    }

    // Destroy the plan when done processing
    ier = cufinufftf_destroy(dplan);
    if (ier != 0) {
        printf("err: cufinufft_destroyc\n");
        return ier;
    }

    // Copy test data back to host and compare
    checkCudaErrors(cudaMemcpy(c, d_c, M * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    int jt = M / 2; // check arbitrary choice of one targ pt
    complex_float J = IMA * (float)iflag;
    complex_float ct = complex_float(0, 0);
    int m = 0;
    for (int m2 = -(N2 / 2); m2 <= (N2 - 1) / 2; ++m2) // loop in correct order over F
        for (int m1 = -(N1 / 2); m1 <= (N1 - 1) / 2; ++m1)
            ct += fk[m++] * exp(J * (m1 * x[jt] + m2 * y[jt])); // crude direct
    printf("[gpu   ] one targ: rel err in c[%ld] is %.3g\n", (int64_t)jt, abs(c[jt] - ct) / infnorm(M, c));

    // Cleanup
    checkCudaErrors(cudaFreeHost(x));
    checkCudaErrors(cudaFreeHost(y));
    checkCudaErrors(cudaFreeHost(c));
    checkCudaErrors(cudaFreeHost(fk));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_c));
    checkCudaErrors(cudaFree(d_fk));

    return 0;
}