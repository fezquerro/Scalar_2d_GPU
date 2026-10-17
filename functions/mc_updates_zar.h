
__global__ void MC_update_even_zar(double* phi, curandStateMRG32k3a_t* state, int* accept, struct lattice l, unsigned long long step) {//update even sites of the lattice
	int id = blockDim.x * blockIdx.x + threadIdx.x;// define the position in the lattice
	RNG_DECL(id, step) //random numbers of this site for this update
	int pos_y = id % l.y;
	int pos_x = (id % (l.x * l.y )) / (l.y);
	int pos_t = id / (l.x * l.y);
	int par_aux = (pos_t + pos_x) % 2;
	int pos = id + l.V;
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
	double kappa = 0;
	if (!(pos_y == 0 && par_aux == 0) && !(pos_y == l.y - 1 && par_aux == 1)) {//bulk sites
		aux2 += phi[pos + yp] + phi[pos + ym];//Term with y
		aux2 += phi[pos + xp] + phi[pos + xm];//Term with x
		aux2 += phi[pos + tp] + phi[pos + tm];//Term with t
		kappa = l.m * l.m + 6;//coefficient of phi^2 in the local action
	}
	else {
		if (pos_y == l.y - 1 && par_aux == 1) {//At the right boundary points (y=l.y) apply the modified action of NBC
			aux2 +=  phi[pos + ym];//Term with y
			aux2 += phi[pos + xp] + phi[pos + xm];//Term with x
			aux2 += phi[pos + tp] + phi[pos + tm];//Term with t
			 kappa = l.m * l.m + 5;//coefficient of phi^2 in the local action
			//printf("Even f %d %d %d %d %d %d %d %d %d\n", pos_y, pos_x, pos_t, yp, ym, xp, xm, tp, tm);
		}
	}
//	printf("Even %d %d %d %d %d %d %d %d %d\n", pos_y, pos_x, pos_t, yp, ym, xp, xm, tp, tm);
	

	if (!(pos_y == 0 && par_aux == 0)) {//Do not update the boundary points at the left (y=0)
		site_update(&phi[id], &accept[id], aux2, kappa, l.epsilon, RNG_U1(id), RNG_U2(id));//update the site with the selected algorithm
	}
}

__global__ void MC_update_odd_zar(double* phi, curandStateMRG32k3a_t* state, int* accept, struct lattice l, unsigned long long step) {//update odd sites of the lattice

	int id = blockDim.x * blockIdx.x + threadIdx.x;// define the position in the lattice
	RNG_DECL(id, step) //random numbers of this site for this update
	int pos_y = id % l.y;
	int pos_x = (id % (l.x * l.y)) /  l.y;
	int pos_t = id / (l.x * l.y);
	int par_aux = (pos_t + pos_x) % 2;
	int pos = id + l.V;
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
	double kappa = 0;
	if (!(pos_y == 0 && par_aux == 1) && !(pos_y == l.y - 1 && par_aux == 0)) {//bulk sites
		aux2 += phi[id + yp] + phi[id + ym];//Term with y
		aux2 += phi[id + xp] + phi[id + xm];//Term with x
		aux2 += phi[id + tp] + phi[id + tm];//Term with t
		kappa = l.m * l.m + 6;//coefficient of phi^2 in the local action
	}
	else {
		if (pos_y == l.y - 1 && par_aux == 0) {//At the right boundary points (y=l.y) apply the modified action of NBC
			aux2 += phi[id + ym];//Term with y
			aux2 += phi[id + xp] + phi[id + xm];//Term with x
			aux2 += phi[id + tp] + phi[id + tm];//Term with t
			kappa = l.m * l.m + 5;//coefficient of phi^2 in the local action
			//printf("Odd f %d %d %d %d %d %d %d %d %d\n", pos_y, pos_x, pos_t, yp, ym, xp, xm, tp, tm);
		}
	}
	//printf("Odd %d %d %d %d %d %d %d %d %d\n", pos_y, pos_x, pos_t, yp, ym, xp, xm, tp, tm);
	
	//The action is a^2/2*((2d+m^2a^2)*phi^2-\sum (phi(x+a)+phi(x-a))*phi) but we use the adimensional phi=a/phi, M=am
	if (!(pos_y == 0 && par_aux == 1)) {//Do not update the boundary points at the left (y=0)

		site_update(&phi[pos], &accept[pos], aux2, kappa, l.epsilon, RNG_U1(id), RNG_U2(id));//update the site with the selected algorithm
	}
}


