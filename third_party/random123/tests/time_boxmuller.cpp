// Test for boxmuller.h on CPU
#include <Random123/philox.h>
#include <Random123/threefry.h>
#include <Random123/boxmuller.hpp>
#include "util.h"   // for timer()

#if __GNUC__>=7
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#endif

typedef r123::Philox4x32 CBRNGF;
#if R123_USE_64BIT
typedef r123::Threefry2x64 CBRNGD;
#endif

const char *progname = "time_boxmuller";

// Each call to boxmuller() returns a pair of values in the .x and .y
// members, which we add up into sum just to avoid being optimized away.
template <typename CBRNG, typename F, typename F2>
F timedloop(typename CBRNG::ukey_type k, size_t Ntry){
    F sum = 0.f;
    typename CBRNG::ctr_type ctr = {{}};
    const size_t csize = sizeof(ctr)/sizeof(ctr[0]);
    CBRNG rng;

    for(size_t i=0; i<Ntry; i+=csize){
        ctr.incr();
        typename CBRNG::ctr_type u = rng(ctr, k);
	F2 f2;
	switch(csize) {
	case 8: f2 = r123::boxmuller(u[6], u[7]); sum += f2.x + f2.y;
                f2 = r123::boxmuller(u[4], u[5]); sum += f2.x + f2.y;
	case 4: f2 = r123::boxmuller(u[2], u[3]); sum += f2.x + f2.y;
	case 2: f2 = r123::boxmuller(u[0], u[1]); sum += f2.x + f2.y;
                break;
	default:
	        R123_ASSERT(0);
	}
    }
    return sum;
}

template <typename CBRNG, typename F2>
void dumploop(FILE *fp, typename CBRNG::ukey_type k, size_t Ntry){
    typename CBRNG::ctr_type ctr = {{}};
    const size_t csize = sizeof(ctr)/sizeof(ctr[0]);
    CBRNG rng;

    for(size_t i=0; i<Ntry; i+=csize){
        ctr.incr();
        typename CBRNG::ctr_type u = rng(ctr, k);
	F2 f2;
	switch(csize) {
	case 8: f2 = r123::boxmuller(u[6], u[7]); fprintf(fp, "%g\n%g\n", f2.x, f2.y);
                f2 = r123::boxmuller(u[4], u[5]); fprintf(fp, "%g\n%g\n", f2.x, f2.y);
	case 4: f2 = r123::boxmuller(u[2], u[3]); fprintf(fp, "%g\n%g\n", f2.x, f2.y);
	case 2: f2 = r123::boxmuller(u[0], u[1]); fprintf(fp, "%g\n%g\n", f2.x, f2.y); break;
	default:
	    R123_ASSERT(0);
	}
    }
}

#define NREPEAT 20

template <typename CBRNG, typename F, typename F2>
void timedcall(const char *tname, typename CBRNG::ukey_type k, size_t Ntry, char *out_fname) {
    double cur_time, dt;
    F sums[NREPEAT];
    int i;
    FILE *fp;
    char *fname;
    if (out_fname) {
	fname = (char *) malloc(strlen(out_fname) + strlen(tname) + 2);
	CHECKNOTZERO(fname);
	sprintf(fname, "%s-%s", out_fname, tname);
	fp = fopen(fname, "w");
	CHECKNOTZERO(fp);
    } else {
	fname = NULL;
	fp = NULL;
    }
    (void) timer(&cur_time);
    /*
     * we call timedloop NREPEAT times so that it is easy to keep
     * Ntry the same for boxmuller.cu and boxmuller.cpp, so sum[0]
     * can be checked.
     */
    for (i = 0; i < NREPEAT; i++) {
	k.v[sizeof(k)/sizeof(k.v[0])-1] = i;
	if (fp)
	    dumploop<CBRNG, F2>(fp, k, Ntry);
	else
	    sums[i] = timedloop<CBRNG, F, F2>(k, Ntry);
    }
    dt = timer(&cur_time);
    if (fp) {
	printf("%s %lu written to %s in %g sec: %gM/sec\n", tname, (unsigned long)(Ntry*NREPEAT), fname, dt, Ntry*NREPEAT*1.e-6/dt);
	fclose(fp);
	free(fname);
    } else {
	printf("%s %lu in %g sec: %gM/sec, sum = %g\n", tname, (unsigned long)(Ntry*NREPEAT), dt, Ntry*NREPEAT*1.e-6/dt, sums[0]);
	for (i = 1; i < NREPEAT; i++) {
	    printf(" %g", sums[i]);
	}
	printf("\n");
    }
}

const size_t DEF_N = 200000;

int main(int argc, char **argv){
    CBRNGF::ukey_type keyf = {{}};
#if R123_USE_64BIT
    CBRNGD::ukey_type keyd = {{}};
#endif
    size_t Ntry = DEF_N;
    char *dumpfname;
    
    dumpfname = getenv("BOXMULLER_DUMPFILE");
    if(argc>1) {
	if (argv[1][0] == '-') {
	    fprintf(stderr, "Usage: %s [iterations_per_thread [key0 [key1]]]\n", argv[0]);
	    exit(1);
	}
        Ntry = atol(argv[1]);
    }
    for (int i = 0; i < (int)(sizeof(keyf)/sizeof(keyf[0])-1) && 2+i < argc; i++) {
	keyf.v[i] = atol(argv[2+i]);
    }
    timedcall<CBRNGF,float,r123::float2>("float", keyf, Ntry, dumpfname);

#if R123_USE_64BIT
    for (int i = 0; i < (int)(sizeof(keyd)/sizeof(keyd[0])-1) && 2+i < argc; i++) {
	keyd.v[i] = atol(argv[2+i]);
    }
    timedcall<CBRNGD,double,r123::double2>("double", keyd, Ntry, dumpfname);
#endif
    return 0;
}
    
