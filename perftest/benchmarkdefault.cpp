#include <cstdint>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <complex>
#include <random>
#include <regex>
#include <chrono>

#include <finufft.h>
#ifndef FINUFFT_USE_DUCC0
#include <fftw3.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#define NTHREADS omp_get_max_threads()
#else
#define NTHREADS 1
#endif

static const double PI = 3.141592653589793238462643383279502884;


struct timing_parser_t {
    #define FLOAT_REGEX "\\d(?:\\.\\d+)?(?:e[\\+\\-]\\d+)?"
    const std::vector<std::pair<std::string, std::regex>> message_patterns = {
        {"count_threads",            std::regex("\\[(FINUFFT_PLAN_T)] detected \\d+ threads in (" FLOAT_REGEX ") sec.[^]*")}, 
        {"precompute_horner_coeffs", std::regex("\\[(precompute_horner_coeffs)] ns=\\d+:\\t(" FLOAT_REGEX ") s[^]*")},
        {"kernel_fser",              std::regex("\\[(init_grid_kerFT_FFT)] kernel fser \\(ns=\\d+\\):\\t\\t(" FLOAT_REGEX ") s[^]*")},
        {"fft_plan",                 std::regex("\\[(init_grid_kerFT_FFT)] FFT plan \\(mode \\d+, nthr=\\d+\\):\\t(" FLOAT_REGEX ") s[^]*")},
        {"sort",                     std::regex("\\[(.+)] sort \\(didSort=\\d+\\):\\t\\t(" FLOAT_REGEX ") s[^]*")},
        {"phase_deconv",             std::regex("\\[(setpts t3)] phase & deconv factors:\\t(" FLOAT_REGEX ") s[^]*")},
        {"inner_t2",                 std::regex("\\[(setpts t3)] inner t2 plan & setpts: \\t(" FLOAT_REGEX ") s[^]*")},
        {"spread",                   std::regex("\\[(.+)\\][^]*tot spread:\\t\\t(" FLOAT_REGEX ")[^]*")},
        {"deconvolve",               std::regex("\\[(.+)\\][^]*tot deconvolve:\\t\\t(" FLOAT_REGEX ")[^]*")},
        {"fft",                      std::regex("\\[(.+)\\][^]*tot FFT:\\t\\t(" FLOAT_REGEX ")[^]*")},
        {"interpolate",              std::regex("\\[(.+)\\][^]*tot interp:\\t\\t(" FLOAT_REGEX ")[^]*")},
        {"prephase",                 std::regex("\\[(.+)\\][^]*tot prephase:\\t\\t(" FLOAT_REGEX ")[^]*")},
        {"inner_NUFFT",              std::regex("\\[(.+)\\][^]*tot inner NUFFT:\\t\\t(" FLOAT_REGEX ")[^]*")},
        {"postphase",                std::regex("\\[(.+)\\][^]*tot postphase:\\t\\t(" FLOAT_REGEX ")[^]*")}
    };

    int type;
    int idx;
    timing_parser_t(int idx, int type) : idx(idx), type(type) {}

    struct timing_t {
        int parameters_idx;
        int type;
        std::string stage;
        std::string name;
        std::string location;
        float duration;
    
        friend std::ostream &operator<<(std::ostream &outs, const timing_t &timing) {
            return outs << timing.parameters_idx << "," << timing.type << "," << timing.stage << "," << timing.name << "," << timing.location << "," << timing.duration << "\n";
        }
    };

    std::vector<timing_t> timings = {};
    void parse_messages(std::string stage, std::string debug_message) {
        for(const auto& [name, re]: message_patterns) {
            std::smatch match;
            if(std::regex_match(debug_message, match, re)) {
                timings.push_back((timing_t){idx, type, stage, name, match.str(1), std::stof(match.str(2))});
            }
        }
    }
};

bool is_float(std::string value) {
    std::istringstream vs;
    float res;
    vs >> res;
    if(vs.fail()) return false;
    return true;
}

struct parameters_t {
    int idx;
    char prec;
    std::int64_t N[3];
    int ntransf;
    int thread;
    std::int64_t M;
    float tol;
    parameters_t(std::string line) {
        std::stringstream field_stream(line);
        std::string field;
        try {
            std::getline(field_stream, field, ','), idx     = std::stoi(field);
            std::getline(field_stream, field, ','), prec    = field[0];
            std::getline(field_stream, field, ','), N[0]    = static_cast<int64_t>(std::stof(field));
            std::getline(field_stream, field, ','), N[1]    = static_cast<int64_t>(std::stof(field));
            std::getline(field_stream, field, ','), N[2]    = static_cast<int64_t>(std::stof(field));
            std::getline(field_stream, field, ','), ntransf = is_float(field) ? static_cast<int>(std::stof(field)) : NTHREADS;
            std::getline(field_stream, field, ','), thread  = is_float(field) ? static_cast<int>(std::stof(field)) : NTHREADS;
            std::getline(field_stream, field, ','), M       = static_cast<int64_t>(std::stof(field));
            std::getline(field_stream, field, ','), tol     = std::stof(field);
        } catch (const std::exception &e) {
            std::cerr << "Failed to parse a line: '" << line << "'\n";
            throw;
        }
    }
    int ndims() const {
        int ndims = 0;
        for(int i=0;i<3;i++) 
            ndims += N[i]>1;
        return ndims;
    }

    friend std::ostream &operator<<(std::ostream &outs, const parameters_t &params) {
        return outs << "" << params.idx
         << ", prec=" << params.prec
         << ", N=[" << params.N[0] << "," << params.N[1] << "," << params.N[2] << "]"
         << ", ndims=" << params.ndims()  
         << ", ntransf=" << params.ntransf 
         << ", thread=" << params.thread
         << ", M=" << params.M
         << ", tol=" <<  params.tol;
    }
};

template<class F, class... Args> inline std::vector<std::string> capture_stdout(F f, Args... args) {
    std::vector<std::string> result;
    int pipefd[2];
    if(pipe(pipefd) == -1) return result;
    int stdout_file = dup(STDOUT_FILENO);
    fflush(stdout);
    dup2(pipefd[1], STDOUT_FILENO);
    close(pipefd[1]);

    f(args...);

    fflush(stdout);
    dup2(stdout_file, STDOUT_FILENO);
    close(stdout_file);

    char buf[1024];
    ssize_t nbytes;
    std::stringstream ss;
    while ((nbytes = read(pipefd[0], buf, sizeof(buf))) > 0) {
        for(int i=0;i<nbytes;i++) {
            if(buf[i] == '[' && (ss.seekg(0, std::ios::end), ss.tellg() > 0)) {
                result.push_back(std::move(ss.str()));
                ss.str(std::string());
            }
            ss << buf[i];
        }
    }
    if(ss.seekg(0, std::ios::end), ss.tellg() > 0)
        result.push_back(std::move(ss.str()));
    return result;
}

template<typename T> void run_test(parameters_t &par, std::ofstream &stagef, std::ofstream &sectf) {
    auto N = par.N[0]*par.N[1]*par.N[2];
    std::vector<T> x(par.M * par.ntransf), y(par.M * par.ntransf), z(par.M * par.ntransf);
    std::vector<T> s(N * par.ntransf), t(N * par.ntransf), u(N * par.ntransf);
    std::vector<std::complex<T>> c(par.M * par.ntransf), fk(N * par.ntransf);

    std::default_random_engine eng{42};
    std::uniform_real_distribution<T> dist11(-1, 1);
    auto randm11 = [&eng, &dist11]() {
      return dist11(eng);
    };

    for (int i = 0; i < par.M * par.ntransf; i++) {
      c[i].real(randm11());
      c[i].imag(randm11());
    }

    for (int i = 0; i < N * par.ntransf; i++) {
      fk[i].real(randm11());
      fk[i].imag(randm11());

      s[i] = PI * randm11();
      t[i] = PI * randm11();
      u[i] = PI * randm11();
    }
    finufft_opts opts;
    finufft_default_opts(&opts);
    int iflag = 1; 
    int type = 3;
    opts.debug=2;
    for(int type=1;type<=1;type++) {
        timing_parser_t parser(par.idx, type);
        std::vector<std::string> debug_output;
        std::chrono::time_point<std::chrono::steady_clock> start, end;
        T *x_p = par.ndims() >= 1 ? x.data() : nullptr;
        T *y_p = par.ndims() >= 2 ? y.data() : nullptr;
        T *z_p = par.ndims() == 3 ? z.data() : nullptr;
        T *s_p = type == 3 && par.ndims() >= 1 ? s.data() : nullptr;
        T *t_p = type == 3 && par.ndims() >= 2 ? t.data() : nullptr;
        T *u_p = type == 3 && par.ndims() == 3 ? u.data() : nullptr;

        if constexpr (std::is_same_v<T, double>) {
            finufft_plan_s *plan{nullptr};
            start = std::chrono::steady_clock::now();
            debug_output = capture_stdout(finufft_makeplan, type, par.ndims(), par.N, iflag, par.ntransf, par.tol, &plan, &opts);
            end = std::chrono::steady_clock::now();
            stagef << par.idx << "," << type << "," << "makeplan" << "," << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "\n";
            for(auto &m: debug_output) parser.parse_messages("makeplan", m);

            start = std::chrono::steady_clock::now();
            debug_output = capture_stdout(finufft_setpts, plan, par.M, x_p, y_p, z_p, N, s_p, t_p, u_p);
            end = std::chrono::steady_clock::now();
            stagef << par.idx << "," << type << "," << "setpts" << "," << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "\n";
            for(auto &m: debug_output) parser.parse_messages("setpts", m);

            start = std::chrono::steady_clock::now();
            debug_output = capture_stdout(finufft_execute, plan, c.data(), fk.data());
            end = std::chrono::steady_clock::now();
            stagef << par.idx << "," << type << "," << "execute" << "," << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "\n";
            for(auto &m: debug_output) parser.parse_messages("execute", m);
        } else {
            finufftf_plan_s *plan{nullptr};
            start = std::chrono::steady_clock::now();
            debug_output = capture_stdout(finufftf_makeplan, type, par.ndims(), par.N, iflag, par.ntransf, par.tol, &plan, &opts);
            end = std::chrono::steady_clock::now();
            stagef << par.idx << "," << type << "," << "makeplan" << "," << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "\n";
            for(auto &m: debug_output) parser.parse_messages("makeplan", m);

            start = std::chrono::steady_clock::now();
            debug_output = capture_stdout(finufftf_setpts, plan, par.M, x_p, y_p, z_p, N, s_p, t_p, u_p);
            end = std::chrono::steady_clock::now();
            stagef << par.idx << "," << type << "," << "setpts" << "," << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "\n";
            for(auto &m: debug_output) parser.parse_messages("setpts", m);

            start = std::chrono::steady_clock::now();
            debug_output = capture_stdout(finufftf_execute, plan, c.data(), fk.data());
            end = std::chrono::steady_clock::now();
            stagef << par.idx << "," << type << "," << "execute" << "," << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "\n";
            for(auto &m: debug_output) parser.parse_messages("execute", m);
        }

        for(auto &t : parser.timings) {
            sectf << t;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc == 1 || (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))) {
        std::cout << "Usage: benchmarkdefault input-parameters.csv [stage-timings.csv] [section-timings.csv]" << "\n";
    }
    std::ifstream input(argv[1]);
    std::ofstream stagef(argc > 2 ? argv[2] : "stages.csv");
    stagef << "par_idx,type,stage,ms\n";
    std::ofstream sectf(argc > 3 ? argv[3] : "sections.csv");
    sectf << "par_idx,type,stage,name,location,ms\n";
    std::string row;
    // Skip header
    std::getline(input, row);
    std::vector<parameters_t> variants;

    while(std::getline(input, row)) {
        variants.push_back(parameters_t(row));
    }

    for(auto &v: variants) {
        std::cout << v << std::endl;
        if(v.prec == 'd') {
            run_test<double>(v, stagef, sectf);
        } else {
            run_test<float>(v, stagef, sectf);
        }
    }
    return 0;
}


