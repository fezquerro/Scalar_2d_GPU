
//Energy of the configuration (first component of the energy-momentum tensor), summed over the lattice in the same kernel: one kernel per measurement
//Each thread computes the energy of its site, written with the forward differences summed by parts, the block adds the values
//in shared memory and thread 0 adds the block sum divided by the volume to the measurement with atomicAdd.
//It works for any lattice size and block size (launch with nthreads * sizeof(double) bytes of shared memory)

__device__ __forceinline__ void block_sum_add(double v, double* res, double norm) {//add the sum of v over the block, times norm, to *res
	extern __shared__ double sh[];
	int tid = threadIdx.x;
	sh[tid] = v;
	__syncthreads();
	int p = 1;
	while (p < blockDim.x) p <<= 1;//smallest power of two >= block size
	for (int s = p / 2; s > 0; s >>= 1) {//tree reduction
		if (tid < s && tid + s < blockDim.x)
			sh[tid] += sh[tid + s];
		__syncthreads();
	}
	if (tid == 0)
		atomicAdd(res, sh[0] * norm);
}

__global__ void energy(double* phi, double* result, struct lattice l, double norm) {//periodic and Dirichlet boundary conditions
	int posf = (blockDim.x * blockIdx.x + threadIdx.x);// define the position in the lattice
	int parity = posf / l.V; //#the first sublattice is "even" and the second "odd" sites
	int pos = posf % l.V;//position in the sublattice
	int pos_y = pos % l.y;
	int pos_x = (pos % (l.x * l.y)) / l.y;
	int pos_t = pos / (l.x * l.y);
	int par_aux = (pos_t + pos_x + parity) % 2;
	int yp, xp, tp;

	yp = par_aux; //the positive derivative in y
	if (pos_y == (l.y - 1) && par_aux == 1)
		yp = -l.y + 1;
	xp = l.y;//the positive derivative in x
	if (pos_x == (l.x - 1))
		xp = -l.y * (l.x - 1);
	tp = l.y * l.x;//the positive derivative in t
	if (pos_t == (l.t - 1))
		tp = -l.y * l.x * (l.t - 1);
	double aux2 = 0;
	aux2 += phi[posf + yp + l.V * (1 - 2 * parity)];//Term with y
	aux2 += phi[posf + xp + l.V * (1 - 2 * parity)];//Term with x
	aux2 += -phi[posf + tp + l.V * (1 - 2 * parity)];//Term with t
	double e = 0.5 * (2 + l.m * l.m) * phi[posf] * phi[posf] - aux2 * phi[posf];

	block_sum_add(e, result, norm);
}

__global__ void energy_anti(double* phi, double* result, struct lattice l, double norm) {//anti-periodic boundary conditions
	int posf = (blockDim.x * blockIdx.x + threadIdx.x);// define the position in the lattice
	int parity = posf / l.V; //#the first sublattice is "even" and the second "odd" sites
	int pos = posf % l.V;//position in the sublattice
	int pos_y = pos % l.y;
	int pos_x = (pos % (l.x * l.y)) / l.y;
	int pos_t = pos / (l.x * l.y);
	int par_aux = (pos_t + pos_x + parity) % 2;
	int yp, xp, tp;

	yp = par_aux; //the positive derivative in y
	if (pos_y == (l.y - 1) && par_aux == 1)
		yp = -l.y + 1;
	xp = l.y;//the positive derivative in x
	if (pos_x == (l.x - 1))
		xp = -l.y * (l.x - 1);
	tp = l.y * l.x;//the positive derivative in t
	if (pos_t == (l.t - 1))
		tp = -l.y * l.x * (l.t - 1);
	double aux2 = 0;
	if (!(pos_y == l.y - 1 && par_aux == 1))
		aux2 += phi[posf + yp + l.V * (1 - 2 * parity)];//Term with y
	else //boundary points to the right (y=l.y) have the opposite coupling
		aux2 += -phi[posf + yp + l.V * (1 - 2 * parity)];//Term with y
	aux2 += phi[posf + xp + l.V * (1 - 2 * parity)];//Term with x
	aux2 += -phi[posf + tp + l.V * (1 - 2 * parity)];//Term with t
	double e = 0.5 * (2 + l.m * l.m) * phi[posf] * phi[posf] - aux2 * phi[posf];

	block_sum_add(e, result, norm);
}

__global__ void energy_neu(double* phi, double* result, struct lattice l, double norm) {//Neumann and Zaremba boundary conditions
	int posf = (blockDim.x * blockIdx.x + threadIdx.x);// define the position in the lattice
	int parity = posf / l.V; //#the first sublattice is "even" and the second "odd" sites
	int pos = posf % l.V;//position in the sublattice
	int pos_y = pos % l.y;
	int pos_x = (pos % (l.x * l.y)) / l.y;
	int pos_t = pos / (l.x * l.y);
	int par_aux = (pos_t + pos_x + parity) % 2;
	int yp, xp, tp;

	yp = par_aux; //the positive derivative in y
	if (pos_y == (l.y - 1) && par_aux == 1)
		yp = -l.y + 1;
	xp = l.y;//the positive derivative in x
	if (pos_x == (l.x - 1))
		xp = -l.y * (l.x - 1);
	tp = l.y * l.x;//the positive derivative in t
	if (pos_t == (l.t - 1))
		tp = -l.y * l.x * (l.t - 1);
	double aux2 = 0, coef = 2;
	if (!(pos_y == l.y - 1 && par_aux == 1))//all sites except the right boundary (y=l.y) have the term with y
		aux2 += phi[posf + yp + l.V * (1 - 2 * parity)];//Term with y
	if ((pos_y == 0 && par_aux == 0) || (pos_y == l.y - 1 && par_aux == 1))//boundary points have the modified action
		coef = 1;
	aux2 += phi[posf + xp + l.V * (1 - 2 * parity)];//Term with x
	aux2 += -phi[posf + tp + l.V * (1 - 2 * parity)];//Term with t
	double e = 0.5 * (coef + l.m * l.m) * phi[posf] * phi[posf] - aux2 * phi[posf];

	block_sum_add(e, result, norm);
}
