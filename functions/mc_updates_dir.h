

__global__ void MC_update_even_dir(double* phi, curandStateMRG32k3a_t* state, int* accept, const struct lattice l, unsigned long long step) {//update even sites of the lattice
	int id = blockDim.x * blockIdx.x + threadIdx.x;// define the position in the lattice
	RNG_DECL(id, step) //random numbers of this site for this update
	int pos_y = id % l.y;
	int pos_x = (id % (l.x * l.y)) / l.y;
	int pos_t = id / (l.x * l.y);
	int par_aux = (pos_t + pos_x) % 2;
	int pos = id + l.V;
	double phi_new = phi[id] + (2 * RNG_U1(id) - 1) * l.epsilon;//generate the new value
	double aux2 = 0; int yp, ym, xp, xm, tp, tm;

	yp = par_aux; //the positive derivative in z
	if (pos_y == (l.y - 1) && par_aux == 1)
		yp = -l.y + 1;
	ym = par_aux - 1; //the negative derivative in z
	xp =  l.y;//the positive derivative in x
	if (pos_x == (l.x - 1))
		xp = - l.y * (l.x - 1);
	xm = - l.y;//the negative derivative in x
	if (pos_x == 0)
		xm =  l.y * (l.x - 1);
	tp =  l.y * l.x;//the positive derivative in t
	if (pos_t == (l.t - 1))
		tp = - l.y * l.x * (l.t - 1);
	tm = -l.y * l.x;//the negative derivative in t
	if (pos_t == 0)
		tm =  l.y * l.x * (l.t - 1);
	if (!(pos_y == 0 && par_aux == 0)) {// do not update the boundary points (y=0) since they are fixed to zero
		aux2 += phi[pos + yp] + phi[pos + ym];//Term with y
		aux2 += phi[pos + xp] + phi[pos + xm];//Term with x
		aux2 += phi[pos + tp] + phi[pos + tm];//Term with t
		double s = 0.5 * (l.m * l.m + 6) * (phi_new * phi_new - phi[id] * phi[id]) - aux2 * (phi_new - phi[id]);//obtain the delta of the action

		if (s < 0) {//if the delta of the action is negative it accepts the change
			phi[id] = phi_new;
			accept[id]++;
		}

		else {
			if (metropolis_accept(s, RNG_U2(id))) {//metropolis algorithm for accepting the change,if not using double for the random number it affects
				phi[id] = phi_new;
				accept[id]++;
			}
		}
	}
}

__global__ void MC_update_odd_dir(double* phi, curandStateMRG32k3a_t* state, int* accept, const struct lattice l, unsigned long long step) {//update odd sites of the lattice

	int id = blockDim.x * blockIdx.x + threadIdx.x;// define the position in the lattice
	RNG_DECL(id, step) //random numbers of this site for this update
	int pos_y = id %  l.y;
	int pos_x = (id % (l.x * l.y )) /  l.y;
	int pos_t = id / (l.x * l.y);
	int par_aux = (pos_t + pos_x) % 2;
	int pos = id + l.V;
	double phi_new = phi[pos] + (2 * RNG_U1(id) - 1) * l.epsilon;
	double aux2 = 0; int  yp, ym, xp, xm, tp, tm;

	yp = 1 - par_aux; //the positive derivative in z
	if (pos_y == (l.y - 1) && par_aux == 0)
		yp = -l.y + 1;
	ym = -par_aux; //the negative derivative in z
	//the positive derivative in x
	xp =  l.y;//jump for the boundary
	if (pos_x == (l.x - 1))
		xp = - l.y * (l.x - 1);
	xm = - l.y;///the negative derivative in x
	if (pos_x == 0)
		xm =  l.y * (l.x - 1);

	tp =  l.y * l.x;//the positive derivative in t
	if (pos_t == (l.t - 1))
		tp = - l.y * l.x * (l.t - 1);
	tm = - l.y * l.x;//the negative derivative in t
	if (pos_t == 0)
		tm = l.y * l.x * (l.t - 1);
	if (!(pos_y == 0 && par_aux == 1)) { // do not update the boundary points (y=0) since they are fixed to zero
		aux2 += phi[id + yp] + phi[id + ym];//Term with y
		aux2 += phi[id + xp] + phi[id + xm];//Term with x
		aux2 += phi[id + tp] + phi[id + tm];//Term with t

		double s = 0.5 * (l.m * l.m + 6) * (phi_new * phi_new - phi[pos] * phi[pos]) - aux2 * (phi_new - phi[pos]);//obtain the delta of the action

		if (s < 0) {//if the delta of the action is negative it accepts the change
			phi[pos] = phi_new;
			accept[pos]++;
		}

		else if (metropolis_accept(s, RNG_U2(id))) {//metropolis algorithm for accepting the change, if not using double for the random number it affects
			phi[pos] = phi_new;
			accept[pos]++;
		}
	}
}

