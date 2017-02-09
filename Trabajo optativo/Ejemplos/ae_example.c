#include <stdio.h>
// To compile: 			>> icc  -vec-report2  -openmp ae_example.c -o ae.mic
// To compile for CPU: 	>> icc -vec-report2   -openmp ae_example.c -o ae.cpu -no-offload
// To execute: >> export OFFLOAD_REPORT=2
//             >> ./axpy_offload


int main(){
  
  double *a, *b, *c;
  int i;
  int n=100;
  // allocated memory on the heap aligned to 64 byte boundary
  a = (double*)malloc(n*sizeof(double));
  b = (double*)malloc(n*sizeof(double));
  c = (double*)malloc(n*sizeof(double));

  // initialize matrices
  for( i = 0; i < n; i++ )
     b[i]=(double)(i);
  for( i = 0; i < n; i++ )
     c[i]=(double)(i);


  //offload code
#pragma offload target(mic) in(b,c:length(n)) out(a:length(n)) 
  {
    //parallelize via OpenMP on MIC
    #pragma omp parallel for
    for( i = 0; i < n; i++ ) {
      a[i] = sin(b[i]) + cos(b[i]);
    }
  }

}

