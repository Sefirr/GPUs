#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

//CUDA
#include <cuda.h>

double wtime(void)
{
        static struct timeval   tv0;
        double time_;

        gettimeofday(&tv0,(struct timezone*)0);
        time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
        return( time_/1000000);
}

int check_matrix(float *h, float *d, int N)
{
	int i,j;

	for (i=0; i<N; i++){
//		printf("Line %i: ", i);
		for (j=0; j<N; j++)
				if(fabsf (h[i*N+j]-d[i*N+j]) > 1e-5){
					printf("device!host %f!=%f in (%i,%i)\n", 
						d[i*N+j], h[i*N+j], i, j);
					return(0);
				}
	}
	return(1);
}


void addMatrix(float *a, float *b, float *c, int N)
{
	int i, j, idx;
	for (i=0; i<N; i++)
		for(j=0; j<N;j++){
			idx = i*N+j;
			a[idx]=b[idx]+c[idx];
		}
} 


__global__ void addMatrixGPU(float *a, float *b, float *c, int N )
{
	int idx;
	int i= threadIdx.x + blockIdx.x*blockDim.x;
	if(i < N) {
		for(int j= 0; j < N;j++) {
			idx = i*N+j;
			a[idx]=b[idx]+c[idx];
		}

	}
	
}

int main(int argc, char *argv[])
{
	float *a, *b, *c, *a_host;
	float *a_GPU, *b_GPU, *c_GPU;

	int i, j, N;

	double t0, t1;


	if(argc>1) {
		N = atoi(argv[1]); printf("N=%i\n", N);
	} else {
		printf("Error!!!! \n ./exec number\n");
	return (0);
	}

	// Mallocs CPU
	a  = (float *)malloc(sizeof(float)*N*N);
	b  = (float *)malloc(sizeof(float)*N*N);
	c  = (float *)malloc(sizeof(float)*N*N);
	for (i=0; i<N*N; i++){ b[i] = i-1; c[i] = i;}

	/*****************/
	/* Add Matrix CPU*/
	/*****************/
	t0 = wtime();
	addMatrix(a, b, c, N);
	t1 = wtime(); printf("Time CPU=%f\n", t1-t0);

	/* Mallocs GPU */
	cudaMalloc((void**)&a_GPU,sizeof(float)*N*N);
	cudaMalloc((void**)&b_GPU,sizeof(float)*N*N);
	cudaMalloc((void**)&c_GPU,sizeof(float)*N*N);

	/* CPU->GPU */
	cudaMemcpy(b_GPU,b,sizeof(float)*N*N,cudaMemcpyHostToDevice);
	cudaMemcpy(c_GPU,c,sizeof(float)*N*N,cudaMemcpyHostToDevice);

	/*****************/
	/* Add Matrix GPU*/
	/*****************/
	int nThreads_previo = 128;
	/*printf("nThreads=%i\n", nThreads_previo);*/
	dim3 nThreads(nThreads_previo);
	dim3 nBlocks((N/128)+1);
	t0 = wtime();
	addMatrixGPU<<<nBlocks,nThreads>>>(a_GPU, b_GPU, c_GPU, N);
	cudaThreadSynchronize();
	t1 = wtime(); printf("Time GPU=%f\n", t1-t0);

	/* GPU->CPU */
	a_host  = (float *)malloc(sizeof(float)*N*N);
	cudaMemcpy(a_host,a_GPU,sizeof(float)*N*N,cudaMemcpyDeviceToHost);

	/************/
	/* Results  */
	/************/
	check_matrix(a,a_host,N);

	/* Free CPU */
	free(a);
	free(b);
	free(c);
	free(a_host);

	/* Free GPU */
	cudaFree(a_GPU);
	cudaFree(b_GPU);
	cudaFree(c_GPU);


	return(1);
}

