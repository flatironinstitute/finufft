#include <spreadinterp_tempinstant.h>
#include <utils_tempinstant.h>

#undef T
#define T float
#include "spreadinterp.cpp"

#undef T
#define T double
#include "spreadinterp.cpp"
