
//formula for computing the energy for the lattice scalar field with the different boundayr conditions in 2+1 dimensions,
//see Chapter 3 in https://arxiv.org/abs/2605.18220 for the derivation of these formulas

#define pi 3.14159265358979324

double energy_analytical(const int Lt, const int Lx, const int Ly, double m, char bound) {
	 double m2=m*m/4,a,s=0,res=NAN;
	 double* st = (double*)malloc(Lt * sizeof(double)); double* sx = (double*)malloc(Lx * sizeof(double)); double* sy = (double*)malloc(Ly * sizeof(double));

	for (int i = 0; i < Lt; i++) {//compute the eigenvalues of the temporal side
		a = sin(pi * i / Lt);
		st[i] = a * a;
	}
	for (int i = 0; i < Lx; i++) {//compute eigenvalues of x side
		a = sin(pi * i / Lx);
		sx[i] = a * a;
	}
	if (bound == 'p'){// Periodic boundary conditions

		for (int i = 0; i < Ly; i++) {//compute eigenvalues y side
			a = sin(pi * i / Ly);
			sy[i] = a * a;
		}
		for (int t = 0; t < Lt; t++) {// compute the energy
			for (int x = 0; x < Lx; x++) {
				for (int y = 0; y < Ly; y++) {
					if (m!=0 || (y+t+x)!=0)
					s += st[t] /(st[t] + sx[x] + sy[y] + m2);
				}
			}
		}
		if (m!=0)
		    res = 0.5 - s / Lt / Lx / Ly;
		else
			res = 0.5*(1-1.0/ Lt / Lx / Ly) - s / Lt / Lx / Ly;
	}
	if (bound == 'd') { // Dirichlet boundary conditions
		for (int i = 1; i < Ly; i++) { //compute eigenvalues y side
			a = sin(pi * i / Ly/2);
			sy[i-1] = a * a;
		}
		for (int t = 0; t < Lt; t++) {// compute the energy
			for (int x = 0; x < Lx; x++) {
				for (int y = 0; y < Ly-1; y++) {
						s += st[t] / (st[t] + sx[x] + sy[y] + m2);
				}
			}
		}
		res = 0.5 * (1 - 1.0 / Ly) - s / Lt / Lx / Ly;

	}
	if (bound == 'a') { // Anti-Periodic boundary conditions
		for (int i = 0; i < Ly; i++) {//compute eigenvalues y side
			a = sin(pi * (i+0.5) / Ly);
			sy[i] = a * a;
		}
		for (int t = 0; t < Lt; t++) {// compute the energy
			for (int x = 0; x < Lx; x++) {
				for (int y = 0; y < Ly; y++) {
						s += st[t] / (st[t] + sx[x] + sy[y] + m2);
				}
			}
		}
			res = 0.5 - s / Lt / Lx / Ly;
	}
	if (bound == 'n') { // Neumann boundary conditions
		for (int i = 0; i < Ly; i++) {//compute eigenvalues y side
			a = sin(pi * i / Ly / 2);
			sy[i] = a * a;
		}
		for (int t = 0; t < Lt; t++) {// compute the energy
			for (int x = 0; x < Lx; x++) {
				for (int y = 0; y < Ly; y++) {
					if (m != 0 || (y + t + x) != 0)
					s += st[t] / (st[t] + sx[x] + sy[y] + m2);
				}
			}
		}
		if (m != 0)
			res = 0.5 - s / Lt / Lx / Ly;
		else
			res = 0.5 * (1 - 1.0 / Lt / Lx / Ly) - s / Lt / Lx / Ly;

	}
	if (bound == 'z') { // Zaremba boundary conditions
		for (int i = 0; i < Ly-1; i++) {//compute eigenvalues y side
			a = sin(pi * (i+0.5) / (2*Ly-1));
			sy[i] = a * a;
		}
		for (int t = 0; t < Lt; t++) {// compute the energy
			for (int x = 0; x < Lx; x++) {
				for (int y = 0; y < Ly-1; y++) {
					s += st[t] / (st[t] + sx[x] + sy[y] + m2);
				}
			}
		}
		res = 0.5 * (1 - 1.0 / Ly) - s / Lt / Lx / Ly;

	}
	if (isnan(res)) printf("Unknown boundary condition '%c'\n", bound);
	free(st); free(sx); free(sy);
	return res;
}
