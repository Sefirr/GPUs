
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
	int nthreads, tid;
	int i, INTERVALS; 
	double n_1, x, pi = 0.0; 
	INTERVALS=128000;

	n_1 = 1.0 / (double)INTERVALS; 
	/* Parallel loop with reduction for calculating PI */  
	#pragma offload target (mic)
	#pragma omp parallel for reduction(+:pi)
	for (i = 0; i < INTERVALS; i++)
	{
//printf("Iam %i of %i\n", omp_get_thread_num(), omp_get_num_threads());
		x = n_1 * ((double)i  - 0.5);
		pi += 4.0 / (1.0 + x * x);
	}
	pi *= n_1; 
	printf ("Pi = %.12lf\n", pi); 
}
