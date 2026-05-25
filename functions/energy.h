

__global__ void energy(double* phi, double* result, struct lattice l) {
	int posf = (blockDim.x * blockIdx.x + threadIdx.x);// define the position in the lattice
	int parity= posf / l.V; //#the first sublattice is "even" and the second "odd" sites
	int pos = posf % l.V;//position in the sublattice
	int pos_y = pos %  l.y;
	int pos_x = (pos % (l.x * l.y )) /  l.y;
	int pos_t = pos / (l.x * l.y);
	int par_aux = (pos_t + pos_x  + parity) % 2;
	int  yp, xp, tp;

	yp = par_aux; //the positive derivative in z
	if (pos_y == (l.y - 1) && par_aux == 1)
		yp = -l.y + 1;
	xp =  l.y;//the positive derivative in x
	if (pos_x == (l.x - 1))
		xp = -l.y * (l.x - 1);
	tp =  l.y * l.x;//the positive derivative in t
	if (pos_t == (l.t - 1))
		tp = - l.y * l.x * (l.t - 1);
	double aux2 = 0;
	

	aux2 += phi[posf + yp + l.V * (1 - 2 * parity)];//Term with y

	aux2 += phi[posf + xp + l.V * (1 - 2 * parity)];//Term with x

	aux2 += -phi[posf + tp + l.V * (1 - 2 * parity)];//Term with t

	result[posf] += 0.5*(2 + l.m * l.m) * phi[posf] * phi[posf] - aux2 * phi[posf];
}
__global__ void energy_anti(double* phi, double* result, struct lattice l) {
	int posf = (blockDim.x * blockIdx.x + threadIdx.x);// define the position in the lattice
	int parity = posf / l.V; //#the first sublattice is "even" and the second "odd" sites
	int pos = posf % l.V;//position in the sublattice
	int pos_y = pos % l.y;
	int pos_x = (pos % (l.x * l.y)) / l.y;
	int pos_t = pos / (l.x * l.y);
	int par_aux = (pos_t + pos_x + parity) % 2;
	int  yp, xp, tp;

	yp = par_aux; //the positive derivative in z
	if (pos_y == (l.y - 1) && par_aux == 1)
		yp = -l.y + 1;
	xp = l.y;//the positive derivative in x
	if (pos_x == (l.x - 1))
		xp = -l.y * (l.x - 1);
	tp = l.y * l.x;//the positive derivative in t
	if (pos_t == (l.t - 1))
		tp = -l.y * l.x * (l.t - 1);
	double aux2 = 0, sum = 0;
	if ( !(pos_y == l.y - 1 && par_aux == 1)) {
		aux2 += phi[posf + yp + l.V * (1 - 2 * parity)];//Term with y

		aux2 += phi[posf + xp + l.V * (1 - 2 * parity)];//Term with x

		aux2 += -phi[posf + tp + l.V * (1 - 2 * parity)];//Term with t

		result[posf] += 0.5 * (2 + l.m * l.m) * phi[posf] * phi[posf] - aux2 * phi[posf];
		//printf("Nope %d %d %d %d\n", pos_t, pos_x, pos_y, par_aux);
	}
	else { //boundary points to the right (y=l.y) have the opposite coupling
		
			aux2 += -phi[posf + yp + l.V * (1 - 2 * parity)];//Term with y

			aux2 += phi[posf + xp + l.V * (1 - 2 * parity)];//Term with x

			aux2 += -phi[posf + tp + l.V * (1 - 2 * parity)];//Term with t

			result[posf] += 0.5 * (2 + l.m * l.m) * phi[posf] * phi[posf] - aux2 * phi[posf];
			//printf("uno %d %d %d %d\n", pos_t, pos_x, pos_y, par_aux);
		

	}



}
__global__ void energy_neu(double* phi, double* result, struct lattice l) {
	int posf = (blockDim.x * blockIdx.x + threadIdx.x);// define the position in the lattice
	int parity = posf / l.V; //#the first sublattice is "even" and the second "odd" sites
	int pos = posf % l.V;//position in the sublattice
	int pos_y = pos % l.y;
	int pos_x = (pos % (l.x * l.y)) / l.y;
	int pos_t = pos / (l.x * l.y);
	int par_aux = (pos_t + pos_x + parity) % 2;
	int  yp, xp, tp;

	yp = par_aux; //the positive derivative in z
	if (pos_y == (l.y - 1) && par_aux == 1)
		yp = -l.y + 1;
	xp = l.y;//the positive derivative in x
	if (pos_x == (l.x - 1))
		xp = -l.y * (l.x - 1);
	tp = l.y * l.x;//the positive derivative in t
	if (pos_t == (l.t - 1))
		tp = -l.y * l.x * (l.t - 1);
	double aux2 = 0;

	if (!(pos_y == 0 && par_aux == 0) && !(pos_y == l.y - 1 && par_aux == 1)) { //bulk sites
		aux2 += phi[posf + yp + l.V * (1 - 2 * parity)];//Term with y

		aux2 += phi[posf + xp + l.V * (1 - 2 * parity)];//Term with x

		aux2 += -phi[posf + tp + l.V * (1 - 2 * parity)];//Term with t

		result[posf] += 0.5 * (2 + l.m * l.m) * phi[posf] * phi[posf] - aux2 * phi[posf];
	}
	else{
		if (pos_y == l.y - 1 && par_aux == 1) {//boundary points to the right (y=l.y) have the modified action

			aux2 += phi[posf + xp + l.V * (1 - 2 * parity)];//Term with x

			aux2 += -phi[posf + tp + l.V * (1 - 2 * parity)];//Term with t

			result[posf] += 0.5 * (1 + l.m * l.m) * phi[posf] * phi[posf] - aux2 * phi[posf];
		}
		else {//boundary points to the left (y=0) have the modified action
			aux2 += phi[posf + yp + l.V * (1 - 2 * parity)];//Term with y

			aux2 += phi[posf + xp + l.V * (1 - 2 * parity)];//Term with x

			aux2 += -phi[posf + tp + l.V * (1 - 2 * parity)];//Term with t

			result[posf] += 0.5 * (1 + l.m * l.m) * phi[posf] * phi[posf] - aux2 * phi[posf];

		}
	}

	
	
}
__global__ void energy_zar(double* phi, double* result, struct lattice l) {
	int posf = (blockDim.x * blockIdx.x + threadIdx.x);// define the position in the lattice
	int parity = posf / l.V; //#the first sublattice is "even" and the second "odd" sites
	int pos = posf % l.V;//position in the sublattice
	int pos_y = pos % l.y;
	int pos_x = (pos % (l.x * l.y)) / l.y;
	int pos_t = pos / (l.x * l.y);
	int par_aux = (pos_t + pos_x + parity) % 2;
	int  yp, xp, tp;

	yp = par_aux; //the positive derivative in z
	if (pos_y == (l.y - 1) && par_aux == 1)
		yp = -l.y + 1;
	xp = l.y;//the positive derivative in x
	if (pos_x == (l.x - 1))
		xp = -l.y * (l.x - 1);
	tp = l.y * l.x;//the positive derivative in t
	if (pos_t == (l.t - 1))
		tp = -l.y * l.x * (l.t - 1);
	double aux2 = 0;

	if (!(pos_y == 0 && par_aux == 0) && !(pos_y == l.y - 1 && par_aux == 1)) {//bulk sites
		aux2 += phi[posf + yp + l.V * (1 - 2 * parity)];//Term with y

		aux2 += phi[posf + xp + l.V * (1 - 2 * parity)];//Term with x

		aux2 += -phi[posf + tp + l.V * (1 - 2 * parity)];//Term with t

		result[posf] += 0.5 * (2 + l.m * l.m) * phi[posf] * phi[posf] - aux2 * phi[posf];
	}
	else {
		if (pos_y == l.y - 1 && par_aux == 1) { //boundary points to the right (y=l.y) have the modified action

			aux2 += phi[posf + xp + l.V * (1 - 2 * parity)];//Term with x

			aux2 += -phi[posf + tp + l.V * (1 - 2 * parity)];//Term with t

			result[posf] += 0.5 * (1 + l.m * l.m) * phi[posf] * phi[posf] - aux2 * phi[posf];
		}
		
	}



}

