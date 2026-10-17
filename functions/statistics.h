
double mean_lattice(double* measured,double *measure,int n, struct lattice l, bool print) {
	cudaMemcpy(measure, measured, l.V * sizeof(double), cudaMemcpyDeviceToHost);
	double mean = 0,variance=0;
	for (int i = 0; i < l.V; i++) {
		mean += measure[i];
		variance += measure[i] * measure[i];
	}
		mean = mean / l.V / n;
		variance = sqrt((variance / l.V/n - mean * mean) /(l.V*n - 1));
	if (print) printf("%.10lf +- %.10lf\n",mean,variance);
	//cudaMemset(measured, 0, l.V * sizeof(double));
	return  mean;
}
__global__ void  suml(double* measure, double* result, int n) {
	int id = (blockDim.x * blockIdx.x + threadIdx.x);
	for (int i = 0; i < n; i++) {
		result[id] += measure[id * n + i];
		measure[id * n + i] = 0;
	}

}
__global__ void  sumlf(double* measure, double* result, int n, int V) {
	int id = (blockDim.x * blockIdx.x + threadIdx.x);
	for (int i = 0; i < n; i++) {
		*result += measure[id * n + i];
		measure[id * n + i] = 0;
	}
	*result /= V;
	//printf("Result sum 1 %.16lf\n", *result);
}




void mean_lattice_par96(double* measure, double* result, double* res, int V, int y) {//parallel reduction for lattice 96 x 96 x y
	int nthreads = V / y * 2;
	suml << < nthreads / 1024, 1024 >> > (measure, result, y / 2);
	nthreads /= 18;
	suml << <1, nthreads >> > (result, measure, 18);
	nthreads /= 8;
	suml << < 1, nthreads >> > (measure, result, 8);
	nthreads /= 8;
	suml << <1, nthreads >> > (result, measure, 8);
	nthreads /= 16;
	sumlf << < 1, 1 >> > (measure, res, 16, V);

}

void mean_lattice_par64(double* measure, double* result, double* res, int V, int y) {//parallel reduction for lattice 64 x 64 x y
	int nthreads = V / y * 2;
	suml << < nthreads / 1024, 1024 >> > (measure, result, y / 2);
	nthreads /= 8;
	suml << <1, nthreads >> > (result, measure, 8);
	nthreads /= 8;
	suml << < 1, nthreads >> > (measure, result, 8);
	nthreads /= 8;
	suml << <1, nthreads >> > (result, measure, 8);
	nthreads /= 16;
	sumlf << < 1, 1 >> > (measure, res, 16, V);

}

void mean_lattice_par32(double* measure, double* result, double* res, int V, int y) {
	int nthreads = V / y * 2;
	suml << < nthreads / 1024, 1024 >> > (measure, result, y / 2);
	nthreads /= 8;
	suml << <1, nthreads >> > (result, measure, 8);
	nthreads /= 8;
	suml << <1, nthreads >> > (measure, result, 8);
	nthreads /= 8;
	suml << <1, nthreads >> > (result, measure, 8);
	nthreads /= 4;
	sumlf << < 1, 1 >> > (measure, res, 4, V);

}


void mean_lattice_small(double* measure, double* result, int V) {

	sumlf << < 1, 1 >> > (measure, result, V, V);
}



double mean_lattice(int* measured, int* measure, int n, struct lattice l, bool print) {
	cudaMemcpy(measure, measured, l.V * sizeof(int), cudaMemcpyDeviceToHost);
	double mean = 0, variance = 0;
	for (int i = 0; i < l.V; i++) {
		mean += measure[i];
		variance += (double)measure[i] * measure[i];
	}
	mean = mean / l.V / n;
	variance = sqrt((variance / l.V / n - mean * mean) / l.V / (n-1));
	if (print) printf("%.10lf\n", mean);
	cudaMemset(measured, 0, l.V * sizeof(int));
	return  mean;
}

void mean_variance(double* measure, int N, double* results, bool print) {
	double s = 0, s2 = 0;
	for (int i = 0; i < N; i++) {
		s += measure[i];
		s2 += measure[i] * measure[i];
	}
	results[0] = s / N;
	results[1] = (s2 / N - results[0] * results[0]);
	if (print) printf("%.10lf +- %.10lf\n", results[0], sqrt(results[1]/(N-1)));
}

void autocorrelation(double* measure, int N, int W, double mean,double variance,double* result) {
	double s = 0;
	for (int i=1; i < W+1; i++) {
		for (int j = 0; j < (N - i); j++) 
			s += (measure[j] - mean) * (measure[j + i] - mean);
		
		result[i-1] = s / (N - i) / variance;
		if (i<11) printf("%d %lf\n", i, result[i-1]);
		s = 0;
	}
}


