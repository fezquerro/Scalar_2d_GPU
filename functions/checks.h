#pragma once
__global__ void check_p(double* res1, double* res2, struct lattice l) {
	for (int i = 0; i < l.x * l.y * l.z * l.t; i++) {
		printf(" %d %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", i, res1[i * 9], res1[i * 9 + 1], res1[i * 9 + 2], res1[i * 9 + 3], res1[i * 9 + 4],
			res1[i * 9 + 5], res1[i * 9 + 6], res1[i * 9 + 7], res1[i * 9 + 8]);
		printf(" %d %lf %lf %lf %lf %lf %lf %lf %lf %lf\n\n\n", i, res2[i * 9], res2[i * 9 + 1], res2[i * 9 + 2], res2[i * 9 + 3], res2[i * 9 + 4],
			res2[i * 9 + 5], res2[i * 9 + 6], res2[i * 9 + 7], res2[i * 9 + 8]);
	}
}
__global__ void check_p2(double* res1, double* res2, struct lattice l) {
	for (int i = 0; i < l.V; i++) {
		printf(" %d %lf %lf %lf %lf %lf \n", i, res1[i * 5], res1[i * 5 + 1], res1[i * 5 + 2], res1[i * 5 + 3], res1[i * 5 + 4]);

		printf(" %d %lf %lf %lf %lf %lf \n\n\n", i, res2[i * 5], res2[i * 5 + 1], res2[i * 5 + 2], res2[i * 5 + 3], res2[i * 5 + 4]);
	}
}
__global__ void copy_check(double* source, double* save, double* pattern) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	save[id] = source[(int)pattern[id]];

}
__global__ void init1(double* phi, struct lattice l) {
	for (int i = 0; i < l.V; i++) {
		phi[i] = i;

	}
	//phi[l] = 1;
	//printf(" %d %lf %d %lf\n",l,phi[l], l + l.x * l.y * l.z * l.t,phi[l+ l.x * l.y * l.z * l.t]);


}
__global__ void init2(double* phi, struct lattice l) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int pos = id % l.V, pos_z, pos_y, pos_x, pos_t, par_aux;
	int parity;
	for (int i = 0; i < l.x * l.y * l.z * l.t; i++) {
		pos_z = i % l.z;
		pos_y = (i % (l.z * l.y)) / (l.z);
		pos_x = (i % (l.x * l.y * l.z)) / (l.z * l.y);
		pos_t = i / (l.x * l.y * l.z);
		par_aux = (pos_t + pos_x + pos_y) % 2;
		if (par_aux == 0) {
			phi[i] = 2 * i;
			phi[i + l.x * l.y * l.z * l.t] = 2 * i + 1;
		}
		else
		{
			phi[i] = 2 * i + 1;
			phi[i + l.x * l.y * l.z * l.t] = 2 * i;
		}
	}
	//phi[l] = 1;
	//printf(" %d %lf %d %lf\n",l,phi[l], l + l.x * l.y * l.z * l.t,phi[l+ l.x * l.y * l.z * l.t]);


}
/*__global__ void MC_update(double* phi, curandStateXORWOW_t* state, int parity, int* accept) {//update one site
	int pos = blockDim.x * blockIdx.x + threadIdx.x;// define the position in the lattice
	int pos_z = pos % l.z;
	int pos_y = (pos % (l.z * l.y)) / (l.z);
	int pos_x = (pos % (l.x * l.y * l.z)) / (l.z * l.y);
	int pos_t = pos / (l.x * l.y * l.z);
	//printf("%d %d %d %d %d %d \n", pos,pos_x,pos_y,pos_z,pos_z, (pos_t % 2 + pos_x % 2 + pos_y % 2 + pos_z % 2) % 2);
	if ((pos_t % 2 + pos_x % 2 + pos_y % 2 + pos_z % 2) % 2 == parity) {
		double phi_new = phi[pos] + (2 * curand_uniform(&state[pos]) - 1) * epsilon;//generate the new value
		//double phi_new = phi[pos] + 1;
		double aux,aux2 = 0; int zp, zm, yp, ym, xp, xm, tp, tm;
		if (pos_z < (l.z - 1))//the positive derivative in z
			zp = 1;
		else
			zp = -l.z + 1;
		if (pos_z > 0)//the negative derivative in z
			zm = -1;
		else
			zm = l.z - 1;
		if (pos_y < (l.y - 1))//the positive derivative in y
			yp = l.z;//jump for the boundary
		else
			yp = -l.z * (l.y - 1);
		if (pos_y > 0)//the negative derivative in y
			ym = -l.z;//jump for the boundary
		else
			ym = l.z * (l.y - 1);
		if (pos_x < (l.x - 1))//the positive derivative in x
			xp = l.z * l.y;//jump for the boundary
		else
			xp = -l.z * l.y * (l.x - 1);
		if (pos_x > 0)//the negative derivative in x
			xm = -l.z * l.y;//jump for the boundary
		else
			xm = l.z * l.y * (l.x - 1);
		if (pos_t < (l.t - 1))//the positive derivative in t
			tp = l.z * l.y * l.x;//jump for the boundary
		else
			tp = -l.z * l.y * l.x * (l.t - 1);
		if (pos_t > 0)//the negative derivative in t
			tm = -l.z * l.y * l.x;//jump for the boundary
		else
			tm = l.z * l.y * l.x * (l.t - 1);
		aux2 += phi[pos + zp] + phi[pos + zm];//Term with z
		aux2 += phi[pos + yp] + phi[pos + ym];//Term with y
		aux2 += phi[pos + xp] + phi[pos + xm];//Term with x
		aux2 += phi[pos + tp] + phi[pos + tm];//Term with t
		aux = phi[pos];
		double s = 0.5*M_ef * (phi_new * phi_new - aux * aux) - aux2 * (phi_new - aux);//obtain the delta of the action
		//The action is a^2/2*((2d+m^2a^2)*phi^2-\sum (phi(x+a)+phi(x-a))*phi) but we use the adimensional phi=a/phi, M=am
		//printf(" %d %lf %lf %lf %d %d %d \n", pos,s,phi[pos],phi_new, pos_t,tm,yp);

		if (s < 0) {//if the delta of the action is negative it accepts the change
			phi[pos] = phi_new;
			accept[pos]++;
		}
		else {
			if (exp(-s) > curand_uniform(&state[pos])) {//metropolis algorithm for accepting the change
				phi[pos] = phi_new;
				accept[pos]++;
			}
		}

}*/

/*__global__ void energy(double* phi, double* result, int V) {
	int pos = (blockDim.x * blockIdx.x + threadIdx.x);// define the position in the lattice
	int pos_z = pos % (l.z);
	int pos_y = (pos % (l.z * l.y)) / (l.z);
	int pos_x = (pos % (l.x * l.y * l.z)) / (l.z * l.y);
	int pos_t = pos / (l.x * l.y * l.z);
	int zp, yp, xp, tp;
	if (pos_z < (l.z - 1))//positions that don't need of jump in z
		zp = 1;
	else //positions that need of jump in z
		zp = -l.z + 1;
	if (pos_y < (l.y - 1))//the positive derivative in y
		yp = l.z;//jump for the boundary
	else
		yp = -l.z * (l.y - 1);
	if (pos_x < (l.x - 1))//the positive derivative in x
		xp = l.z * l.y;//jump for the boundary
	else
		xp = -l.z * l.y * (l.x - 1);
	if (pos_t < (l.t - 1))//the positive derivative in t
		tp = l.z * l.y * l.x;//jump for the boundary
	else
		tp = -l.z * l.y * l.x * (l.t - 1);
	double aux,aux2=0,aux3;
	aux2 += phi[pos + zp];//Term with z
	aux2 += phi[pos + yp] ;//Term with y
	aux2 += phi[pos + xp] ;//Term with x
	aux2 += -phi[pos + tp] ;//Term with t
	aux3 = phi[pos];
	aux = -(4+M_ef2)*aux3* aux3+2*aux2*aux3;
	result[pos] += aux;//sum over the lattice
	/* result[pos * 5] = aux;
	 result[pos * 5 + 1] = phi[pos+zp];
	 result[pos * 5 + 2] = phi[pos + yp];
	 result[pos * 5 + 3] = phi[pos + xp];
	 result[pos * 5 + 4] = phi[pos + tp];
	//result2[pos+V*(j-1)] += aux * aux;
//	square[pos] = phi[pos] * phi[pos];
	//printf(" %d %d %d %d %d %d %f\n",j,pos,tp,xp, yp, zp,aux);
}*/