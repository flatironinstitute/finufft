#include <finufft_common/common.h>
const std::vector<finufft::heuristics::SigmaEstimator> finufft::heuristics::trained = {
finufft::kernel::SigmaEstimator(1,1,{1.04249,0.0932473,0.287557,-0.473353},0.000203721,0.00503031,typeid(float)),
finufft::kernel::SigmaEstimator(2,1,{1.04357,0.0936058,0.291001,-0.473793},0.000203721,0.00448601,typeid(float)),
finufft::kernel::SigmaEstimator(3,1,{1.47045,-0.349957,0.202519,0.0160669},1.17759e-06,4.65378e-06,typeid(float))
};