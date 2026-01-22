#ifndef MATH_PSWF_H
#define MATH_PSWF_H

namespace finufft::common {
/*
normalized zeroth-order pswf
*/
double pswf(double c, double x);

/*
copy from ducc0 not_yet_integrated folder
*/
double pswf_ducc(double c, double x);

} // namespace finufft::common
#endif // MATH_PSWF_H
