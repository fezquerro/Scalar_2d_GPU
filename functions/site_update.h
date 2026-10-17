
//Update of a single site, selected with the ALGO flag in kernel.cu
//The local action of the site is S(phi) = 0.5 * kappa * phi^2 - aux2 * phi, where aux2 is the sum of its neighbours
//and kappa = m^2 + 6 in the bulk (m^2 + 5 for sites that lose a neighbour with Neumann boundary conditions)
//METROPOLIS: propose phi + epsilon*(2*u1-1) and accept it with probability min(1, exp(-dS))
//HEATBATH: draw the new value directly from the conditional distribution of the site, a gaussian
//          with mean aux2/kappa and variance 1/kappa (Box-Muller with u1, u2), so it is always accepted
__device__ __forceinline__ void site_update(double* phi_site, int* accept_site, double aux2, double kappa, double epsilon, double u1, double u2) {
#if ALGO==HEATBATH
	double gauss = sqrt(-2 * log(u1)) * cospi(2 * u2); //standard normal number, u1 in (0,1] so the log is finite
	*phi_site = aux2 / kappa + gauss / sqrt(kappa);
	(*accept_site)++;
#else
	double phi_old = *phi_site;
	double phi_new = phi_old + (2 * u1 - 1) * epsilon;//generate the new value
	double s = 0.5 * kappa * (phi_new * phi_new - phi_old * phi_old) - aux2 * (phi_new - phi_old);//obtain the change of the action due to the new value
	if (s < 0 || metropolis_accept(s, u2)) {//accept if the action decreases, otherwise with probability exp(-s)
		*phi_site = phi_new;
		(*accept_site)++;
	}
#endif
}
