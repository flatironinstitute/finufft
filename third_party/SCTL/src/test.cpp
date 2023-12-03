#include "sctl.hpp"

void ProfileMemgr() {
  long N = 5e5;
  {  // Without memory manager
    sctl::Profile::Tic("No-Memgr");

    sctl::Profile::Tic("Alloc");
    auto A = new double[N];
    sctl::Profile::Toc();

    sctl::Profile::Tic("Array-Write");
#pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) A[i] = 0;
    sctl::Profile::Toc();

    sctl::Profile::Tic("Free");
    delete[] A;
    sctl::Profile::Toc();

    sctl::Profile::Toc();
  }
  {  // With memory manager
    sctl::Profile::Tic("With-Memgr");

    sctl::Profile::Tic("Alloc");
    auto A = sctl::aligned_new<double>(N);
    sctl::Profile::Toc();

    sctl::Profile::Tic("Array-Write");
#pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) A[i] = 0;
    sctl::Profile::Toc();

    sctl::Profile::Tic("Free");
    sctl::aligned_delete(A);
    sctl::Profile::Toc();

    sctl::Profile::Toc();
  }
}

void TestMatrix() {
  sctl::Profile::Tic("TestMatrix");
  sctl::Matrix<double> M1(1000, 1000);
  sctl::Matrix<double> M2(1000, 1000);

  sctl::Profile::Tic("Init");
  for (long i = 0; i < M1.Dim(0) * M1.Dim(1); i++) M1[0][i] = i;
  for (long i = 0; i < M2.Dim(0) * M2.Dim(1); i++) M2[0][i] = i * i;
  sctl::Profile::Toc();

  sctl::Profile::Tic("GEMM");
  sctl::Matrix<double> M3 = M1 * M2;
  sctl::Profile::Toc();

  sctl::Profile::Toc();
}

int main(int argc, char** argv) {
  sctl::SphericalHarmonics<double>::test_stokes();
  return 0;

  sctl::Comm::MPI_Init(&argc, &argv);

  // Dry run (profiling disabled)
  ProfileMemgr();

  // With profiling enabled
  sctl::Profile::Enable(true);
  ProfileMemgr();

  TestMatrix();

  // Print profiling results
  sctl::Profile::print();

  {  // Test out-of-bound writes
    sctl::Iterator<char> A = sctl::aligned_new<char>(10);
    A[9];
    A[10];  // Should print stack trace here (in debug mode).
    // sctl::aligned_delete(A); // Show memory leak warning when commented
  }

  sctl::Comm::MPI_Finalize();
  return 0;
}
