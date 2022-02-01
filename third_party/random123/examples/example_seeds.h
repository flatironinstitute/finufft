#ifndef EXAMPLE_SEEDS_H__
#define EXAMPLE_SEEDS_H__ 1

/*
 * This entire file is overkill to allow seeds to be set in Random123
 * example and test programs via a R123EXAMPLE_ENVCONF_SEED environment
 * variable, mainly to illustrate and test how one might use a user-set
 * seed to produce different random streams for different runs.
 * None of this code is needed for the correct functioning or
 * use of the Random123 library.
 */

#include <stdlib.h> // for strtoul
#include <limits.h> // for ULONG_MAX
#include <errno.h>  // for errno
#include <string.h> // for strerror
#include <stdio.h>  // for stderr

/*
 * The following arbitrary values for sample seeds (used to
 * initialize keys and counters in the examples) have no
 * particular meaning.  They could equally easily all be 0.
 */
#define EXAMPLE_SEED1_U32   0x11111111U
#define EXAMPLE_SEED2_U32   0x22222222U
#define EXAMPLE_SEED3_U32   0x33333333U
#define EXAMPLE_SEED4_U32   0x44444444U

#define EXAMPLE_SEED5_U32   0xdeadbeefU
#define EXAMPLE_SEED6_U32   0xbeadcafeU
#define EXAMPLE_SEED7_U32   0x12345678U
#define EXAMPLE_SEED8_U32   0x90abcdefU
#define EXAMPLE_SEED9_U32   0xdecafbadU

#if R123_USE_64BIT
#define EXAMPLE_SEED1_U64   R123_64BIT(0xdeadbeef12345678)
#define EXAMPLE_SEED2_U64   R123_64BIT(0xdecafbadbeadfeed)
#endif

static inline unsigned long example_seed_u64(uint64_t defaultseed) {
    const char *e = "R123EXAMPLE_ENVCONF_SEED";
    const char *cp = getenv(e);
    unsigned long u;
    char *ep;
    errno = 0;
    if (cp) {
	u = strtoul(cp, &ep, 0);
	if (u == ULONG_MAX && errno != 0) {
	    fprintf(stderr, "strtoul failed to convert environment variable %s=\"%s\" to unsigned long: %s\n",
		    e, cp, strerror(errno));
	    exit(1);
	} else if (*ep != '\0') {
	    fprintf(stderr, "strtoul failed to fully convert environment variable %s=\"%s\" to unsigned long, got 0x%lu\n",
		    e, cp, u);
	    exit(1);
	}
    } else {
	u = defaultseed;
    }
    return u;
}

static inline uint32_t example_seed_u32(uint32_t defaultseed) {
    uint64_t u64 = example_seed_u64(defaultseed);
    if (u64 > 0xFFFFFFFFUL /* UINT32_MAX, which clang29 does not have, sigh */) {
	fprintf(stderr, "Warning: truncating seed 0x%lu to uint32_t\n", (unsigned long)u64);
    }
    return (uint32_t)u64;
}


#endif /* EXAMPLE_SEEDS_H__ */
