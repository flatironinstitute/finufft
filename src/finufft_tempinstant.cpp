#include <finufft_tempinstant.h>
#include <common_tempinstant.h>
#include <utils_tempinstant.h>

#undef T
#define T float
#include "finufft.cpp"

#undef T
#define T double
#include "finufft.cpp"
