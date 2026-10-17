
struct lattice {
	int y;
	int x;
	int t;
	int V;
	float m;
	float epsilon;

};
__global__ void setup_kernel(curandStateMRG32k3a_t* state, unsigned long long seed)//initialize the curands state for RNG
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	/* Each thread gets same seed, a different sequence
	   number, no offset */
	curand_init(seed, id, 0, &state[id]);
}


__global__ void rand_init2(double* phi, curandStateMRG32k3a_t* state, struct lattice l) {//initialize lattice with ranadom entries
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	phi[i] = 2 * curand_uniform_double(&state[i]) - 1;
	phi[i + l.V] = 2 * curand_uniform_double(&state[i]) - 1;
	//phi[l] = 1;
	//printf(" %d %lf %d %lf\n",l,phi[l], l + l.x * l.y * l.z * l.t,phi[l+ l.x * l.y * l.z * l.t]);


}

__global__ void rand_init2_dir(double* phi, curandStateMRG32k3a_t* state, struct lattice l) {//initialize lattice with random entries and DBC
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int pos_y = id % l.y;
	int pos_x = (id % (l.x * l.y)) /  l.y;
	int pos_t = id / (l.x * l.y);
	int par_aux = (pos_t + pos_x) % 2;
	if (pos_y == 0 && par_aux == 0)
		phi[id] = 0;
	else
		phi[id] = 2 * curand_uniform_double(&state[id]) - 1;
	if (pos_y == 0 && par_aux == 1)
		phi[id + l.V] = 0;
	else
		phi[id + l.V] = 2 * curand_uniform_double(&state[id]) - 1;
	//phi[l] = 1;
	//printf(" %d %lf %d %lf\n",l,phi[l], l + l.x * l.y * l.z * l.t,phi[l+ l.x * l.y * l.z * l.t]);


}
void printlattice(double* phi, struct lattice l)//print the lattice values
{
	for (int t = 0; t < l.t; t++)
	{
		for (int x = 0; x < l.x; x++)
		{
			for (int y = 0; y < l.y; y++)
			printf("%lf ", phi[y +  l.y * x + l.x * l.y * t]);
			
			printf("\n");
		}
		printf("\n\n");
	}

}

__global__ void sumd(double* phi, struct lattice l)
{
	double sum = 0;
	for (int t = 0; t < l.t; t++)
	{
		for (int x = 0; x < l.x; x++)
		{
			for (int y = 0; y < l.y; y++)
					sum += phi[ y + l.y * x + l.x * l.y  * t];


		}

	}
	printf("Result sum 2 %.16lf\n", sum / l.V);
}