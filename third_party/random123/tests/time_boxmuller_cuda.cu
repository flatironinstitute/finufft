// Test for boxmuller.h on CUDA
#include <Random123/philox.h>
#include <Random123/threefry.h>
#include "util.h"   // for timer()
#include "util_cuda.h"	// for cuda_init, CHECKCALL
#include <Random123/boxmuller.hpp>

typedef r123::Philox4x32 CBRNGF;
typedef r123::Threefry2x64 CBRNGD;
int debug = 0;
const char *progname = "time_boxmuller_cuda";

// Sometimes warnings are A LOT more trouble than they're worth.
// if we just write u[6], we get warnings
// so we write u[(csize>n)?6:0].
#define UGLY(n) (csize>n)?n:0

// The timedloop kernel sums N randoms per thread for timing and
// records that sum in out[tid] (mainly to ensure that
// the random generation process does not get optimized away)
template <typename CBRNG, typename F, typename F2>
__global__ void timedloop(F *out, typename CBRNG::ukey_type k, size_t N){
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t klast = sizeof(k)/sizeof(k[0]) - 1;
    R123_ASSERT(k[klast] == 0); // uses last element of key to
    k[klast] = tid;		// ensure unique key per thread
    F sum = 0.f;
    typename CBRNG::ctr_type ctr = {};
    const size_t csize = sizeof(ctr)/sizeof(ctr[0]);
    CBRNG rng;

    for(size_t i=0; i<N; i+=csize){
        ctr.incr();
        typename CBRNG::ctr_type u = rng(ctr, k);
	F2 f2;
	// Using a loop instead of the Duff device here costs 10%,
	// at least in CUDA4.2 circa Jan 2013 on a Tesla C2050!
	switch(csize) {
	case 8: f2 = r123::boxmuller(u[UGLY(6)], u[UGLY(7)]); sum += f2.x + f2.y;
		f2 = r123::boxmuller(u[UGLY(4)], u[UGLY(5)]); sum += f2.x + f2.y;
	case 4: f2 = r123::boxmuller(u[UGLY(2)], u[UGLY(3)]); sum += f2.x + f2.y;
	case 2: f2 = r123::boxmuller(u[0], u[1]); sum += f2.x + f2.y;
	        break;
	default:
	        R123_ASSERT(0);
	}
    }
    out[tid] = sum;
}

// The dumploop kernel records all the normal randoms individually in out,
// so it produces N randoms per thread.  Each thread records
// its randoms in tid, NTHREADS+tid, NTHREAD*2+tid, ..., NTHREADS*(N-1)+tid
// which hopefully results in nicely coalesced writes from each warp.
template <typename CBRNG, typename F, typename F2>
__global__ void dumploop(F *out, typename CBRNG::ukey_type k, size_t N){
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t klast = sizeof(k)/sizeof(k[0]) - 1;
    R123_ASSERT(k[klast] == 0); // uses last element of key to
    k[klast] = tid;		// ensure unique key per thread
    typename CBRNG::ctr_type ctr = {};
    const size_t csize = sizeof(ctr)/sizeof(ctr[0]);
    CBRNG rng;

    for(size_t i=0; i<N;){
        ctr.incr();
        typename CBRNG::ctr_type u = rng(ctr, k);
	F2 f2;
	// Using a loop instead of the Duff device here costs 10%,
	// at least in CUDA4.2 circa Jan 2013 on a Tesla C2050!
	switch(csize) {
	case 8: f2 = r123::boxmuller(u[UGLY(6)], u[UGLY(7)]);
		out[blockDim.x*gridDim.x*i + tid] = f2.x;
		i++;
		out[blockDim.x*gridDim.x*i + tid] = f2.y;
		i++;
		f2 = r123::boxmuller(u[UGLY(4)], u[UGLY(5)]);
		out[blockDim.x*gridDim.x*i + tid] = f2.x;
		i++;
		out[blockDim.x*gridDim.x*i + tid] = f2.y;
		i++;
	case 4: f2 = r123::boxmuller(u[UGLY(2)], u[UGLY(3)]);
#undef UGLY
		out[blockDim.x*gridDim.x*i + tid] = f2.x;
		i++;
		out[blockDim.x*gridDim.x*i + tid] = f2.y;
		i++;
	case 2: f2 = r123::boxmuller(u[0], u[1]);
		out[blockDim.x*gridDim.x*i + tid] = f2.x;
		i++;
		out[blockDim.x*gridDim.x*i + tid] = f2.y;
		i++;
		break;
	default:
		asm("trap;");
	}
    }
}

template <typename CBRNG, typename F, typename F2>
void timedcall(const char *tname, const char *out_fname, CUDAInfo *infop, typename CBRNG::ukey_type k, size_t N) {
    double cur_time, dt;
    const int nthreads = infop->blocks_per_grid*infop->threads_per_block;
    const size_t nrand = out_fname ? N * nthreads : nthreads;
    const size_t out_size = nrand*sizeof(F);
    F *d_out, *h_out = (F *) malloc(out_size);
    CHECKNOTZERO(h_out);
    CHECKCALL(cudaMalloc(&d_out, out_size));
    (void) timer(&cur_time);
    if (out_fname)
	dumploop<CBRNG,F,F2> <<<infop->blocks_per_grid, infop->threads_per_block>>> (d_out, k, N);
    else
	timedloop<CBRNG,F,F2> <<<infop->blocks_per_grid, infop->threads_per_block>>> (d_out, k, N);
    CHECKCALL(cudaDeviceSynchronize());
    CHECKCALL(cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost));
    dt = timer(&cur_time);
    printf("%s %zd in %g sec: %gM/sec\n", tname, N*nthreads, dt, N*nthreads*1.e-6/dt);
    if (out_fname) {
	char *fname = (char *) malloc(strlen(out_fname) + strlen(tname) + 2);
	CHECKNOTZERO(fname);
	sprintf(fname, "%s-%s", out_fname, tname);
	FILE *fp = fopen(fname, "w");
	CHECKNOTZERO(fp);
	for (size_t i = 0; i < nrand; i++){
	    fprintf(fp, "%g\n", h_out[i]);
	}
	fclose(fp);
	free(fname);
    } else {
	int nwoops = 0;
	printf("%s h_out[0] = %g\n", tname, h_out[0]);
	for (size_t i = 0; i < nrand; i++){
	    if(h_out[i] == 0.f){
		if(nwoops++<10)
		    printf("Woops %s h_out[%zd] = %g\n", tname, i, h_out[i]);

	    }
	}
	if(nwoops>10){
	    printf("Woops %s %d times\n", tname, nwoops);
	}
    }
    CHECKCALL(cudaFree(d_out));
    free(h_out);
}

const size_t DEF_N = 200000;

int main(int argc, char **argv){
    CBRNGF::ukey_type keyf = {};
    CBRNGD::ukey_type keyd = {};
    size_t Ntry = DEF_N;
    char *cp = getenv("R123_DEBUG");
    if (cp)
	debug = atoi(cp);
    if ((cp = getenv("BOXMULLER_DUMPFILE")) != NULL) {
	Ntry = 8;
    } else {
	Ntry = DEF_N;
    }
    if(argc>1) {
	if (argv[1][0] == '-') {
	    fprintf(stderr, "Usage: %s [iterations_per_thread [key0 [key1]]]\n", argv[0]);
	    exit(1);
	}
        Ntry = atol(argv[1]);
    }
    // XXX cannot use keyf.size in host code, only in device code
    for (int i = 0; i < (int)(sizeof(keyf)/sizeof(keyf[0])-1) && 2+i < argc; i++) {
	keyf.v[i] = atol(argv[2+i]);
    }
    for (int i = 0; i < (int)(sizeof(keyd)/sizeof(keyd[0])-1) && 2+i < argc; i++) {
	keyd.v[i] = atol(argv[2+i]);
    }
    CUDAInfo *infop = cuda_init(NULL);
    timedcall<CBRNGF,float,r123::float2>("float", cp, infop, keyf, Ntry);
    timedcall<CBRNGD,double,r123::double2>("double",cp, infop, keyd, Ntry);
    cuda_done(infop);
    return 0;
}
    
