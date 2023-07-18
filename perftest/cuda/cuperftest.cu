#include <getopt.h>

#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>

#include <cufinufft.h>
#include <cufinufft/impl.h>

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct timespec get_wtime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts;
}

double get_wtime_diff(const struct timespec &ts, const struct timespec &tf) {
    return (tf.tv_sec - ts.tv_sec) + (tf.tv_nsec - ts.tv_nsec) * 1E-9;
}

std::string get_or(const std::unordered_map<std::string, std::string> &m, const std::string &key,
                   const std::string &default_value) {
    auto it = m.find(key);
    if (it == m.end()) {
        return default_value;
    }
    return it->second;
}

struct test_options_t {
    char prec;
    int type;
    int n_runs;
    int N[3];
    int M;
    int kerevalmethod;
    int method;
    int sort;
    double tol;

    test_options_t(int argc, char *argv[]) {
        std::unordered_map<std::string, std::string> options_map;

        while (true) {
            int option_index = 0;

            // clang-format off
            static struct option long_options[] {
                {"prec", required_argument, 0, 0},
                {"type", required_argument, 0, 0},
                {"n_runs", required_argument, 0, 0},
                {"N1", required_argument, 0, 0},
                {"N2", required_argument, 0, 0},
                {"N3", required_argument, 0, 0},
                {"M", required_argument, 0, 0},
                {"tol", required_argument, 0, 0},
                {"method", required_argument, 0, 0},
                {"kerevalmethod", required_argument, 0, 0},
                {"sort", required_argument, 0, 0},
                {0, 0, 0, 0},
            };
            // clang-format on

            int c = getopt_long(argc, argv, "", long_options, &option_index);
            if (c == -1)
                break;

            switch (c) {
            case 0:
                options_map[long_options[option_index].name] = optarg;
                break;

            default:
                break;
            }
        }

        prec = get_or(options_map, "prec", "f")[0];
        type = std::stoi(get_or(options_map, "type", "1"));
        n_runs = std::stoi(get_or(options_map, "n_runs", "10"));
        N[0] = std::stof(get_or(options_map, "N1", "1E6"));
        N[1] = std::stof(get_or(options_map, "N2", "1"));
        N[2] = std::stof(get_or(options_map, "N3", "1"));
        M = std::stof(get_or(options_map, "M", "2E6"));
        method = std::stoi(get_or(options_map, "method", "1"));
        kerevalmethod = std::stoi(get_or(options_map, "kerevalmethod", "0"));
        sort = std::stoi(get_or(options_map, "sort", "1"));
        tol = std::stof(get_or(options_map, "tol", "1E-5"));
    }

    friend std::ostream &operator<<(std::ostream &outs, const test_options_t &opts) {
        return outs << "prec = " << opts.prec << "\n"
                    << "type = " << opts.type << "\n"
                    << "n_runs = " << opts.n_runs << "\n"
                    << "N1 = " << opts.N[0] << "\n"
                    << "N2 = " << opts.N[1] << "\n"
                    << "N3 = " << opts.N[2] << "\n"
                    << "M = " << opts.M << "\n"
                    << "method = " << opts.method << "\n"
                    << "kerevalmethod = " << opts.kerevalmethod << "\n"
                    << "sort = " << opts.sort << "\n"
                    << "tol = " << opts.tol << "\n";
    }
};

template <class F, class... Args>
inline double timeit(F f, Args... args) {
    auto st = get_wtime();
    f(args...);
    cudaDeviceSynchronize();
    auto ft = get_wtime();
    return get_wtime_diff(st, ft);
}

void gpu_warmup() {
    int nf1 = 1;
    cufftHandle fftplan;
    cufftPlan1d(&fftplan, nf1, CUFFT_Z2Z, 1);
}

template <typename T>
void run_test(test_options_t &test_opts) {
    std::cout << test_opts;
    const int M = test_opts.M;
    const int N = test_opts.N[0] * test_opts.N[1] * test_opts.N[2];
    const int type = test_opts.type;
    constexpr int iflag = 1;

    thrust::host_vector<T> x(M), y(M), z(M);
    thrust::host_vector<thrust::complex<T>> c(M), fk(N);

    thrust::device_vector<T> d_x(M), d_y(M), d_z(M);
    thrust::device_vector<thrust::complex<T>> d_c(M), d_fk(N);

    std::default_random_engine eng(1);
    std::uniform_real_distribution<T> dist11(-1, 1);
    auto randm11 = [&eng, &dist11]() { return dist11(eng); };

    // Making data
    for (int i = 0; i < M; i++) {
        x[i] = M_PI * randm11(); // x in [-pi,pi)
        y[i] = M_PI * randm11();
        z[i] = M_PI * randm11();
    }
    d_x = x, d_y = y, d_z = z;

    if (type == 1) {
        for (int i = 0; i < M; i++) {
            c[i].real(randm11());
            c[i].imag(randm11());
        }
        d_c = c;
    } else if (type == 2) {
        for (int i = 0; i < N; i++) {
            fk[i].real(randm11());
            fk[i].imag(randm11());
        }
        d_fk = fk;
    } else {
        std::cerr << "Invalid type " << type << " supplied\n";
        return;
    }

    gpu_warmup();

    cufinufft_opts opts;
    int dim = 0;
    for (int i = 0; i < 3; ++i)
        dim = test_opts.N[i] > 1 ? i + 1 : dim;

    cufinufft_default_opts(test_opts.type, dim, &opts);
    opts.gpu_method = test_opts.method;
    opts.gpu_sort = test_opts.sort;
    opts.gpu_kerevalmeth = test_opts.kerevalmethod;

    cufinufft_plan_t<T> *dplan;
    double makeplan_time{0}, setpts_time{0}, execute_time{0};
    makeplan_time =
        timeit(cufinufft_makeplan_impl<T>, test_opts.type, dim, test_opts.N, iflag, 1, test_opts.tol, &dplan, &opts);

    T *d_x_p = dim >= 1 ? d_x.data().get() : nullptr;
    T *d_y_p = dim >= 2 ? d_y.data().get() : nullptr;
    T *d_z_p = dim == 3 ? d_z.data().get() : nullptr;
    cuda_complex<T> *d_c_p = (cuda_complex<T> *)d_c.data().get();
    cuda_complex<T> *d_fk_p = (cuda_complex<T> *)d_fk.data().get();
    for (int i = 0; i < test_opts.n_runs; ++i) {
        setpts_time += timeit(cufinufft_setpts_impl<T>, M, d_x_p, d_y_p, d_z_p, 0, nullptr, nullptr, nullptr, dplan);
        execute_time += timeit(cufinufft_execute_impl<T>, d_c_p, d_fk_p, dplan);
    }

    setpts_time /= test_opts.n_runs;
    execute_time /= test_opts.n_runs;

    std::cout << std::endl;
    std::cout << "makeplan: " << makeplan_time << std::endl;
    std::cout << "setpts  : " << setpts_time << std::endl;
    std::cout << "execute : " << execute_time << std::endl;
    std::cout << "total   : " << makeplan_time + setpts_time + execute_time << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        test_options_t default_opts(0, nullptr);
        // clang-format off
        std::cout << "Valid options:\n"
                     "    --prec <char>\n"
                     "           float or double precision. i.e. 'f' or 'd'\n"
                     "           default: " << default_opts.prec << "\n" << 
                     "    --type <int>\n"
                     "           type of transform. 1 or 2\n"
                     "           default: " << default_opts.type << "\n" << 
                     "    --n_runs <int>\n"
                     "           number of runs to average performance over\n"
                     "           default: " << default_opts.n_runs << "\n" <<
                     "    --N1 <int>\n"
                     "           number of modes in first dimension. Scientific notation accepted (i.e. 1E6)\n"
                     "           default: " << default_opts.N[0] << "\n" <<
                     "    --N2 <int>\n"
                     "           number of modes in second dimension. Scientific notation accepted (i.e. 1E6)\n"
                     "           default: " << default_opts.N[1] << "\n" <<
                     "    --N3 <int>\n"
                     "           number of modes in third dimension. Scientific notation accepted (i.e. 1E6)\n"
                     "           default: " << default_opts.N[2] << "\n" <<
                     "    --M <int>\n"
                     "           number of non-uniform points. Scientific notation accepted (i.e. 1E6)\n"
                     "           default: " << default_opts.M << "\n" <<
                     "    --tol <float>\n"
                     "           NUFFT tolerance. Scientific notation accepted (i.e. 1.2E-7)\n"
                     "           default: " << default_opts.tol << "\n" <<
                     "    --method <int>\n"
                     "           NUFFT method\n"
                     "               1: nupts driven\n"
                     "               2: sub-problem\n"
                     "               4: block-gather\n"
                     "           Note that not all methods are compatible with all dim/type combinations\n"
                     "           default: " << default_opts.method << "\n" <<
                     "    --kerevalmeth <int>\n"
                     "           kernel evaluation method\n"
                     "               0: Exponential of square root\n"
                     "               1: Horner evaluation\n"
                     "           default: " << default_opts.kerevalmethod << "\n" <<
                     "    --sort: <int>\n"
                     "           sort strategy\n"
                     "               0: do not sort the points\n"
                     "               1: sort the points\n"
                     "           default: " << default_opts.sort << "\n";
        // clang-format on
        return 0;
    }
    test_options_t opts(argc, argv);

    if (opts.prec == 'f')
        run_test<float>(opts);
    else if (opts.prec == 'd')
        run_test<double>(opts);

    return 0;
}
