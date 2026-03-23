#include <array>
#include <vector>
#include <typeindex>
#include <ostream>

namespace finufft::kernel {
double get_sigma(double tol, int type, int dim);
double map_to_domain(double x, double lower, double upper);
struct SigmaEstimator {
public:
    static constexpr int NCOEFFS = 4;
    SigmaEstimator(int type, int dim, const std::vector<double> &coeffs, double lower_tol, double upper_tol, std::type_index prec);
    bool match(int transform_type, int transform_dim, std::type_index &transform_precision);
    double best_sigma(double tol);
    friend std::ostream &operator<<(std::ostream &os, const SigmaEstimator &self);
private:
    int type;
    int dim;
    std::array<double, NCOEFFS> coefficients; 
    double lower_tol;
    double upper_tol;
    std::type_index precision;
};
extern const SigmaEstimator trained[];
}

