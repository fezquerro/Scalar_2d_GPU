
//Random number generators, selected with the RNG flag in kernel.cu
//MRG: MRG32k3a from cuRAND, one state per site stored in global memory and initialized by setup_kernel
//PHILOX: Philox4_32_10 from cuRAND, counter based: the random numbers of a site are computed from (seed, site, step),
//        so nothing is stored in memory. Each update kernel launch must receive a different step (step 0 is used for the initial lattice)

__constant__ unsigned long long c_seed; //seed used by PHILOX

#if RNG==PHILOX
//two uniform doubles in (0,1] for the site id at the given step
#define RNG_DECL(id, step) curandStatePhilox4_32_10_t rng_st; curand_init(c_seed, id, 4ULL * (step), &rng_st); double2 rng_u = curand_uniform2_double(&rng_st);
#define RNG_U1(id) rng_u.x
#define RNG_U2(id) rng_u.y
#else
#define RNG_DECL(id, step)
#define RNG_U1(id) curand_uniform_double(&state[id])
#define RNG_U2(id) curand_uniform_double(&state[id])
#endif
