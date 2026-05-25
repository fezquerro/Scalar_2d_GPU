void save_autocorrelation(double* result, int W, int N, int N_cor, int N_term, struct lattice l, char bcond, char nsave[2]) {// save autocorrelation of the observable
	char filename[256]; FILE* f;
	snprintf(filename, sizeof(filename), "Autocorrelation_%d_%d_%d_M_%.2lf_%c_%s.txt", l.t, l.x, l.y, l.m,bcond,nsave);
	
	if ((f = fopen(filename, "w")) == NULL) {
		printf("Error creating the file");
		exit(1);
	}
	fprintf(f, "N_termalizacion %d N %d N_correlation %d  D_t %d D_x %d D_ y %d masa %.2lf deltaphi %.2lf\n",N_term,N,N_cor,l.t,l.x,l.y,l.m,l.epsilon);
	fprintf(f, "N Autocorrelation\n");
	for(int i=0;i<W;i++)
		fprintf(f, "%d %.10lf\n",i+1,result[i]);
	fclose(f);

}
void save_energy(double mean,double variance,int N_term, int N,int N_cor,double accept, struct lattice l, char bcond, char nsave[2]) {// save the energy 
	char filename[256]; FILE* f;

	snprintf(filename, sizeof(filename), "Energy_%d_%d_%d_M_%.2lf_%c_%s.txt", l.t, l.x, l.y, l.m,bcond,nsave);
	if ((f = fopen(filename, "w")) == NULL) {
		printf("Error creating the file");
		exit(1);
	}
	fprintf(f, "N_termalizacion %d N %d N_correlation %d  D_t %d D_x %d D_ y %d deltaphi %.2lf accept %.2lf\n", N_term, N, N_cor, l.t, l.x, l.y, l.epsilon, accept);
	fprintf(f, "M Energy Error\n");
	fprintf(f, "%f  %.10lf %.10lf\n", l.m, mean, sqrt(variance/(N-1)));
	fclose(f);

}