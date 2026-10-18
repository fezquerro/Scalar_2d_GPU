
//Metropolis accept test: returns exp(-s) > u, with the same decision as computing exp in double precision
//The double exp is slow on GPUs with reduced FP64 throughput, so exp(-s) is first estimated in single precision.
//expf has a relative error below 3e-7 and rounding s to float adds below |s|*6e-8 (5e-6 for s < 88, where expf underflows),
//so the exact value is always inside ef*(1 -+ 1e-4): outside that band the float result decides,
//inside it (about 1e-5 of the updates) exp is computed in double. For s > 88 ef = 0 and the change is rejected, which is also exact
//because exp(-88) ~ 6e-39 is smaller than any uniform number produced by the generators
__device__ __forceinline__ bool metropolis_accept(double s, double u) {
	const double tol = 1e-4; //relative band around the single precision estimate
	double ef = expf(-(float)s);
	if (u < ef * (1 - tol)) //certainly accepted
		return true;
	if (u > ef * (1 + tol)) //certainly rejected
		return false;
	return exp(-s) > u; //rare case: decide with the exact value
}
