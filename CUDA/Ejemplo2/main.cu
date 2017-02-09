#include <stdio.h>
#include <malloc.h>
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


void init_matrix(float *M, int hM, int wM, float k)
{
	int i,j;

	for (i=0; i<hM; i++)
		for (j=0; j<wM; j++)
			if (i==j)
				M[i*wM+j] = k*k*1.0f;
			else
				M[i*wM+j] = -k*1.0f;
}

int check_matrix(float *h, float *d, int hM, int wM)
{
	int i,j;

	for (i=0; i<hM; i++){
//		printf("Line %i: ", i);
		for (j=0; j<wM; j++)
				if(fabsf (h[i*wM+j]-d[i*wM+j]) > 1e-5){
					printf("device!host %f!=%f in (%i,%i)\n", 
						d[i*wM+j], h[i*wM+j], i, j);
					return(0);
				}
	}
	return(1);
}


void print_matrix(float *M, int hM, int wM)
{
	int i,j;

	for (i=0; i<hM; i++){
//		printf("Line %i: ", i);
		for (j=0; j<wM; j++)
			printf("%4.1f ", M[i*wM+j]);
		printf("\n");
	}
}


void Mul(float *A, float *B, int hA, int wA, int wB, float *C)
{
	int i,j,k;

	for (i=0; i<hA; i++)
		for (j=0; j<wB; j++){
			C[i*wB+j] = 0.0;
			for (k=0; k<wA; k++)
				C[i*wB+j] += A[i*wA+k]*B[k*wA+j];
		}
}

//MUL EN CUDA

__global__ void Mul_GPU(float *A, float *B, int hA, int wA, int wB, float *C)
{	
	int k;
	int i= threadIdx.y + blockIdx.y*blockDim.y;
	int j= threadIdx.x + blockIdx.x*blockDim.x;
	if((i < hA) && (j < wB) ) {
		C[i*wB+j] = 0.0;
		for (k=0; k<wA; k++)
			C[i*wB+j] += A[i*wA+k]*B[k*wA+j];
	}

}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	// Matrix variables
	float *A, *B, *C_host, *C_device;
	float *a_GPU, *b_GPU, *c_GPU;
	
	int hA, wA, hB, wB;
	
	double t0, t1;
	

	setbuf(stdout, NULL);

	if (argc!=4){
		printf("./exec hA hB/WA wB\n");
		exit(-1);
	}

	hA = atoi(argv[1]);
	hB = wA = atoi(argv[2]);
	wB = atoi(argv[3]);

	// Init A and B, malloc C
	int size_A = wA * hA;
	A = (float*)malloc(size_A*sizeof(float));
	init_matrix(A, hA, wA,2.0);

	int size_B = wB * hB;
	B = (float*)malloc(size_B*sizeof(float));
	init_matrix(B, hB, wB,1.0);

	int size_C = wB * hA;
	C_host = (float*)malloc(size_C*sizeof(float));
	
	/*****************/
	/* Mult Matrix CPU*/
	/*****************/
	
	t0 = wtime();
	Mul(A, B, hA, wA, wB, C);	
	t1 = wtime(); printf("Time CPU=%f\n", t1-t0);

/* Mallocs GPU */
	cudaMalloc((void**)&a_GPU,sizeof(float)*wA*hA);
	cudaMalloc((void**)&b_GPU,sizeof(float)*wB*hB);
	cudaMalloc((void**)&c_GPU,sizeof(float)*hA*wB);

/* CPU->GPU */
	cudaMemcpy(a_GPU,A,sizeof(float)*wA*hA,cudaMemcpyHostToDevice);
	cudaMemcpy(b_GPU,B,sizeof(float)*wB*hB,cudaMemcpyHostToDevice);
	
	int nThreads_previo = 16;	
	int myblocks, myblocks2;
	
	if (hA%16==0)
		myblocks=hA/16;
	else 
		myblocks = hA/16+1;

	if (wB%16==0)
		myblocks2=wB/16;
	else 
		myblocks2 = wB/16+1;

	dim3 nThreads(nThreads_previo, nThreads_previo);
	dim3 nBlocks(myblocks, myblocks2);

	//printf("myblocks=%i , myblocks2=%i", myblocks, myblocks2);

	t0 = wtime();
	Mul_GPU<<<nBlocks,nThreads>>>(a_GPU, b_GPU, hA, wA, wB, c_GPU);	
	cudaThreadSynchronize();
	t1 = wtime(); printf("Time GPU=%f\n", t1-t0);
	
	/* GPU->CPU */
	C_device = (float*)malloc(size_C*sizeof(float));
	cudaMemcpy(C_device,c_GPU,sizeof(float)*hA*wB,cudaMemcpyDeviceToHost);

	/************/
	/* Results  */
	/************/
	check_matrix(C_host, C_device, hA, wB);
	
	/* Free CPU */
	free(A);
	free(B);
	free(C_host);
	free(C_device);

	/* Free GPU */
	cudaFree(a_GPU);
	cudaFree(b_GPU);
	cudaFree(c_GPU);

	return (1);

}