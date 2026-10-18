#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <math.h>

//define the flag for each random number generator
#define MRG 0 //MRG32k3a, one state per site stored in memory
#define PHILOX 1 //Philox4_32_10, counter based without stored state (faster)
#define RNG MRG //select the random number generator

//define the flag for each update algorithm
#define METROPOLIS 0 //Metropolis with proposals of size epsilonl
#define HEATBATH 1 //heat-bath, new value drawn from the exact gaussian distribution of the site (decorrelates faster, use a smaller N_cor)
#define ALGO HEATBATH //select the update algorithm

#include "functions\rng.h" //Random number generators
#include "functions\metropolis.h" //Metropolis accept test
#include "functions\site_update.h" //Update of one site with the selected algorithm
#include "functions\lattice_setup.h" // Functions for initializing the lattice
#include "functions\mc_updates.h" //MC updates with periodic boundary conditions (PBC)
#include "functions\mc_updates_anti.h" //MC updates with anti-periodic boundary conditions (ABC)
#include "functions\mc_updates_dir.h" //MC updates with Dirichlet boundary conditions (DBC)
#include "functions\mc_updates_neuman.h" //MC updates with Neumann boundary conditions (NBC)
#include "functions\mc_updates_zar.h" //MC updates with Zaremba boundary conditions (ZBC)
#include "functions\energy.h" //Compute the energy as the observable
#include "functions\statistics.h" //Compute the average over the lattice
#include "functions\output.h" //Save the results
#include "functions\analytic.h" //Compute analytically the results

//define the flag for each boundary condition
#define PBC 0 //periodic boundary conditions
#define ABC 1 //anti-periodic boundary conditions
#define DBC 2 //Dirichlet boundary conditions
#define NBC 3 //Neumann boundary conditions
#define ZBC 4 //Zaremba boundary conditions

#define BC PBC //select the boundary condition
#define N 10000 // number of measures
#define N_term 2000 //steps for thermalization
#define N_block 5 //blocks that the
#define N_cor 2 // steps between measures 
#define save 1 // 0 for saving the simulation results

#define dy 8 //size of y direction
#define dx 16 //size of x direction
#define dt 16 //size of t direction
#define ml 0.1// mass of the scalar field

#define epsilonl 1.1 //delta for the metropolis algotithm
#define W 80 // termf of correlation computed 
#define pmeasure 4
#define SEED 0 // fixed seed for reproducible runs, 0 to use time(NULL)
int main()
{
	char nsave[2] = "1";//counter for saving different runs with the same parameters
	for (int co = 0; co < 1; co++) {
		struct lattice l = { 2 * co + dy,dx,dt, (2 * co + dy) * dx * dt,ml,epsilonl };//dimy,dimx,dimt,V,m,delta
		struct lattice l2 = { l.y / 2,l.x,l.t,l.V / 2,l.m,l.epsilon }; // half lattice
		printf("N %d Lt %d Lx %d Ly %d\n", N, l.t, l.x, l.y);// print parameters of the lattice
		if (l.t % 2 || l.x % 2 || l.y % 2) { // the checkerboard decomposition needs even sizes in every direction
			printf("Lt, Lx and Ly must be even\n");
			return 1;
		}
		int nblocks, nthreads;
		nthreads = 256; // largest block size <= 256 that divides the half lattice, so no site is skipped
		while (l2.V % nthreads != 0)
			nthreads--;
		nblocks = 2 * (l2.V / nthreads); // blocks for the full lattice, the MC kernels use nblocks / 2
		printf("nblocks %d nthreads %d\n", nblocks, nthreads);
		char bcond;
		if (BC == 0) {printf("PERIODIC BOUNDARY CONDITIONS\n"); bcond = 'p';}
		if (BC == 1) { printf("ANTI-PERIODIC BOUNDARY CONDITIONS\n"); bcond = 'a'; }
		if (BC == 2) { printf("DIRICHLET BOUNDARY CONDITIONS\n"); bcond = 'd'; }
		if (BC == 3) { printf("NEUMANN BOUNDARY CONDITIONS\n"); bcond = 'n'; }
		if (BC == 4) { printf("ZAREMBA BOUNDARY CONDITIONS\n"); bcond = 'z'; }
		if (ALGO == METROPOLIS) printf("METROPOLIS UPDATES epsilon %.2f\n", l.epsilon);
		if (ALGO == HEATBATH) printf("HEAT-BATH UPDATES\n");

		unsigned long long seed = (SEED == 0) ? (unsigned long long)time(NULL) : SEED;
		printf("Seed %llu\n", seed);
		curandStateMRG32k3a_t* devStates = NULL; //only used by MRG
		unsigned long long rng_step = 1; //counter of update kernels for PHILOX, step 0 is the initial lattice
#if RNG==PHILOX
		printf("RNG PHILOX\n");
		cudaMemcpyToSymbol(c_seed, &seed, sizeof(seed));
#else
		printf("RNG MRG\n");
		cudaMalloc((void**)&devStates, l2.V * sizeof(curandStateMRG32k3a_t));
		setup_kernel << <nblocks / 2, nthreads >> > (devStates, seed);//setup the random number generator
#endif


		double* phi = (double*)malloc(l.V * sizeof(double)), * phid; cudaMalloc(&phid, l.V * sizeof(double));// define the variable to save the lattice in CPU and GPU

#if BC==DBC || BC==ZBC
		rand_init2_dir << <nblocks / 2, nthreads >> > (phid, devStates, l2);//create the lattice with random entries in (-1,1) and 0 in y=0 for DBC 

#else
		rand_init2 << <nblocks / 2, nthreads >> > (phid, devStates, l2);//create the lattice with random entries in (-1,1)

#endif

		int* accept, * accepth = (int*)malloc(l.V * sizeof(int));
		cudaMalloc(&accept, l.V * sizeof(int)); cudaMemset(accept, 0, l.V * sizeof(int));

		double* end, * en = (double*)malloc(N * sizeof(double)); cudaMalloc(&end, N * sizeof(double)); cudaMemset(end, 0, N * sizeof(double));//energy of each measurement in GPU and CPU
		int i, j;
		int t_ini, t_fin, t_n, block_size = N / N_block;
		for (j = 0; j < N_term; j++) {//thermalize with the choosen BC
#if BC==PBC
			MC_update_even << < nblocks / 2, nthreads >> > (phid, devStates, accept, l2, rng_step++);
			MC_update_odd << < nblocks / 2, nthreads >> > (phid, devStates, accept, l2, rng_step++);

#elif BC==ABC
			MC_update_even_anti << < nblocks / 2, nthreads >> > (phid, devStates, accept, l2, rng_step++);
			MC_update_odd_anti << < nblocks / 2, nthreads >> > (phid, devStates, accept, l2, rng_step++);
#elif BC==DBC
			MC_update_even_dir << < nblocks / 2, nthreads >> > (phid, devStates, accept, l2, rng_step++);
			MC_update_odd_dir << < nblocks / 2, nthreads >> > (phid, devStates, accept, l2, rng_step++);
#elif BC==NBC
			MC_update_even_neumann << < nblocks / 2, nthreads >> > (phid, devStates, accept, l2, rng_step++);
			MC_update_odd_neumann << < nblocks / 2, nthreads >> > (phid, devStates, accept, l2, rng_step++);
#elif BC==ZBC
			MC_update_even_zar << < nblocks / 2, nthreads >> > (phid, devStates, accept, l2, rng_step++);
			MC_update_odd_zar << < nblocks / 2, nthreads >> > (phid, devStates, accept, l2, rng_step++);
#endif

		}
		cudaMemset(accept, 0, l.V * sizeof(int));
		t_ini = time(NULL); t_n = time(NULL);
		for (i = 1; i < N + 1; i++) {
			for (j = 0; j < N_cor; j++) {//MC steps between measures
#if BC==PBC
				MC_update_even << < nblocks / 2, nthreads >> > (phid, devStates, accept, l2, rng_step++);
				MC_update_odd << < nblocks / 2, nthreads >> > (phid, devStates, accept, l2, rng_step++);
#elif BC==ABC
				MC_update_even_anti << < nblocks / 2, nthreads >> > (phid, devStates, accept, l2, rng_step++);
				MC_update_odd_anti << < nblocks / 2, nthreads >> > (phid, devStates, accept, l2, rng_step++);
#elif BC==DBC
				MC_update_even_dir << < nblocks / 2, nthreads >> > (phid, devStates, accept, l2, rng_step++);
				MC_update_odd_dir << < nblocks / 2, nthreads >> > (phid, devStates, accept, l2, rng_step++);
#elif BC==NBC
				MC_update_even_neumann << < nblocks / 2, nthreads >> > (phid, devStates, accept, l2, rng_step++);
				MC_update_odd_neumann << < nblocks / 2, nthreads >> > (phid, devStates, accept, l2, rng_step++);
#elif BC==ZBC
				MC_update_even_zar << < nblocks / 2, nthreads >> > (phid, devStates, accept, l2, rng_step++);
				MC_update_odd_zar << < nblocks / 2, nthreads >> > (phid, devStates, accept, l2, rng_step++);
#endif
			}
#if BC==PBC || BC==DBC //energy of the configuration, averaged over the lattice in the same kernel
			energy << <nblocks, nthreads, nthreads * sizeof(double) >> > (phid, end + i - 1, l2, 1.0 / l.V);
#elif BC==ABC
			energy_anti << <nblocks, nthreads, nthreads * sizeof(double) >> > (phid, end + i - 1, l2, 1.0 / l.V);
#elif BC==NBC || BC==ZBC
			energy_neu << <nblocks, nthreads, nthreads * sizeof(double) >> > (phid, end + i - 1, l2, 1.0 / l.V);
#endif
			if (i % block_size == 0) {
				printf("%lf Time %d Left(min)%lf\n", (double)i / N, (int)(time(NULL) - t_n), (double)(time(NULL) - t_ini) * (N - i) / i / 60);
				t_n = time(NULL);
			}

		}
		t_fin = time(NULL);
		printf("Time of the simulation %d\n", t_fin - t_ini);
		cudaError_t err = cudaGetLastError();
		if (err == cudaSuccess) err = cudaDeviceSynchronize();
		if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
		//cudaMemcpy(phi, phid, l.V * sizeof(double), cudaMemcpyDeviceToHost); printlattice(phi);
		

		cudaMemcpy(en, end, N * sizeof(double), cudaMemcpyDeviceToHost);
		double saccept = mean_lattice(accept, accepth, N * N_cor, l, false);
#if BC==DBC || BC==ZBC
		saccept *= (double)l.y / (l.y - 1); // the sites at y=0 are fixed and never updated
#endif
		printf("Accept metropolis ratio  %.10lf\n", saccept);

		double ener[2]; //mean_variance(eblock_average, N_block, ener,true);
		printf("Energy value "); mean_variance(en, N, ener, true);
		double ener_analytical = energy_analytical(l.t, l.x, l.y, l.m, bcond); printf("Energy analytical %lf\n", ener_analytical);
		printf("Sigma deviation %lf\n", (ener[0] - ener_analytical) / sqrt(ener[1] / (N - 1)));

		double autoc[W]; autocorrelation(en, N, W, ener[0], ener[1], autoc);

#if save==0
		save_energy(ener[0], ener[1], N_term, N, N_cor, saccept, l,bcond,nsave);
		save_autocorrelation(autoc, W, N, N_cor, N_term, l,bcond,nsave);
#endif

		free(phi); free(accepth); cudaFree(phid); cudaFree(accept); cudaFree(devStates);
		free(en); cudaFree(end);
	}
}
