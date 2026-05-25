
__global__ void MC_update_even_anti(double* phi, curandStateMRG32k3a_t* state, int* accept, struct lattice l) {//update even sites of the lattice
	int id = blockDim.x * blockIdx.x + threadIdx.x;// define the position in the lattice
	int pos_y = id % l.y;
	int pos_x = (id % (l.x * l.y )) / (l.y);
	int pos_t = id / (l.x * l.y);
	int par_aux = (pos_t + pos_x) % 2;
	int pos = id + l.V;
	double phi_new = phi[id] + (2 * curand_uniform_double(&state[id]) - 1) * l.epsilon;//generate the new value
	//double phi_new = phi[id] + 1;
	double aux2 = 0; int yp, ym, xp, xm, tp, tm;

	yp = par_aux; //the positive derivative in y
	if (pos_y == (l.y - 1) && par_aux == 1)
		yp = -l.y + 1;
	ym = par_aux - 1; //the negative derivative in y
	if (pos_y == 0 && par_aux == 0)
		ym = l.y - 1;
	xp = l.y;//the positive derivative in x
	if (pos_x == (l.x - 1))
		xp = - l.y * (l.x - 1);
	xm = -l.y;//the negative derivative in x
	if (pos_x == 0)
		xm =  l.y * (l.x - 1);
	tp =  l.y * l.x;//the positive derivative in t
	if (pos_t == (l.t - 1))
		tp = - l.y * l.x * (l.t - 1);
	tm = - l.y * l.x;//the negative derivative in t
	if (pos_t == 0)
		tm =  l.y * l.x * (l.t - 1);

	if (!(pos_y == 0 && par_aux == 0) && !(pos_y == l.y - 1 && par_aux == 1)) {
		aux2 += phi[pos + yp] + phi[pos + ym];//Term with y
		aux2 += phi[pos + xp] + phi[pos + xm];//Term with x
		aux2 += phi[pos + tp] + phi[pos + tm];//Term with t
	}
	else {
		if (pos_y == l.y - 1 && par_aux == 1) {//boundary points at the right have the opposite coupling to the first point in y direction
			aux2 += -phi[pos + yp] + phi[pos + ym];//Term with y
			aux2 += phi[pos + xp] + phi[pos + xm];//Term with x
			aux2 += phi[pos + tp] + phi[pos + tm];//Term with t
			//printf("Even f %d %d %d %d %d %d %d %d %d\n", pos_y, pos_x, pos_t, yp, ym, xp, xm, tp, tm);
		}
		else { //boundary points at the left have the opposite coupling to the last point in y direction
			aux2 += phi[pos + yp] - phi[pos + ym];//Term with y
			aux2 += phi[pos + xp] + phi[pos + xm];//Term with x
			aux2 += phi[pos + tp] + phi[pos + tm];//Term with t
			//printf("Even s %d %d %d %d %d %d %d %d %d\n", pos_y, pos_x, pos_t, yp, ym, xp, xm, tp, tm);
		}
	}
//	printf("Even %d %d %d %d %d %d %d %d %d\n", pos_y, pos_x, pos_t, yp, ym, xp, xm, tp, tm);
	double s = 0.5 * (l.m * l.m + 6) * (phi_new * phi_new - phi[id] * phi[id]) - aux2 * (phi_new - phi[id]);//obtain the delta of the action
	//The action is a^2/2*((2d+m^2a^2)*phi^2-\sum (phi(x+a)+phi(x-a))*phi) but we use the adimensional phi=a/phi, M=am
	if (s < 0) {//if the delta of the action is negative it accepts the change
		phi[id] = phi_new;
		accept[id]++;
	}

	else {
		if (expf(-s) > curand_uniform_double(&state[id])) {//metropolis algorithm for accepting the change
			phi[id] = phi_new;
			accept[id]++;
		}
	}
}

__global__ void MC_update_odd_anti(double* phi, curandStateMRG32k3a_t* state, int* accept, struct lattice l) {//update odd sites of the lattice

	int id = blockDim.x * blockIdx.x + threadIdx.x;// define the position in the lattice
	int pos_y = id % l.y;
	int pos_x = (id % (l.x * l.y)) /  l.y;
	int pos_t = id / (l.x * l.y);
	int par_aux = (pos_t + pos_x) % 2;
	int pos = id + l.V;
	double phi_new = phi[pos] + (2 * curand_uniform_double(&state[id]) - 1) * l.epsilon;
	double aux2 = 0; int  yp, ym, xp, xm, tp, tm;

	yp = 1 - par_aux; //the positive derivative in z
	if (pos_y == (l.y - 1) && par_aux == 0)
		yp = -l.y + 1;
	ym = -par_aux; //the negative derivative in z
	if (pos_y == 0 && par_aux == 1)
		ym = l.y - 1;
	
	xp = l.y;////the positive derivative in x
	if (pos_x == (l.x - 1))
		xp = - l.y * (l.x - 1);
	xm = -l.y;///the negative derivative in x
	if (pos_x == 0)
		xm =  l.y * (l.x - 1);
	
	tp =  l.y * l.x;//the positive derivative in t
	if (pos_t == (l.t - 1))
		tp = - l.y * l.x * (l.t - 1);
	tm = - l.y * l.x;//the negative derivative in t
	if (pos_t == 0)
		tm =  l.y * l.x * (l.t - 1);

	if (!(pos_y == 0 && par_aux == 1) && !(pos_y == l.y - 1 && par_aux == 0)) {
		aux2 += phi[id + yp] + phi[id + ym];//Term with y
		aux2 += phi[id + xp] + phi[id + xm];//Term with x
		aux2 += phi[id + tp] + phi[id + tm];//Term with t
	}
	else {
		if (pos_y == l.y - 1 && par_aux == 0) {//boundary points at the right (y=l.y) have the opposite coupling to the first point in y direction
			aux2 += -phi[id + yp] + phi[id + ym];//Term with y
			aux2 += phi[id + xp] + phi[id + xm];//Term with x
			aux2 += phi[id + tp] + phi[id + tm];//Term with t
			//printf("Odd f %d %d %d %d %d %d %d %d %d\n", pos_y, pos_x, pos_t, yp, ym, xp, xm, tp, tm);
		}
		else {//boundary points at the left (y=0) have the opposite coupling to the last point in y direction
			aux2 += phi[id + yp] - phi[id + ym];//Term with y
			aux2 += phi[id + xp] + phi[id + xm];//Term with x
			aux2 += phi[id + tp] + phi[id + tm];//Term with t
			//printf("Odd s %d %d %d %d %d %d %d %d %d\n", pos_y, pos_x, pos_t, yp, ym, xp, xm, tp, tm);
		}
	}
	//printf("Odd %d %d %d %d %d %d %d %d %d\n", pos_y, pos_x, pos_t, yp, ym, xp, xm, tp, tm);
	double s = 0.5 * (l.m * l.m + 6) * (phi_new * phi_new - phi[pos] * phi[pos]) - aux2 * (phi_new - phi[pos]);//obtain the delta of the action
	//The action is a^2/2*((2d+m^2a^2)*phi^2-\sum (phi(x+a)+phi(x-a))*phi) but we use the adimensional phi=a/phi, M=am
	if (s < 0) {//if the delta of the action is negative it accepts the change
		phi[pos] = phi_new;
		accept[pos]++;
	}

	else if (exp(-s) > curand_uniform_double(&state[id])) {//metropolis algorithm for accepting the change
		phi[pos] = phi_new;
		accept[pos]++;
	}
}


