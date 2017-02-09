#include <stdio.h>
#include <omp.h>

double do_some_integratin(long long nsize)
{

  long long i, start_int, end_int;
  int       iam, np;
  double    mypi, n_1, x;
  double    pi = 0.;
  
  /* Calculate the interval size */
  n_1 = (double)1.0/(double)nsize;
   
  /* Calculate the interval section */
  start_int = 1;
  end_int = 1 + nsize;
  
  /* Begin the OpenMP section */
#pragma omp parallel private(iam,x,i,np)
  {

#pragma omp barrier
    iam = omp_get_thread_num() ;
    np=omp_get_num_threads() ;
  
    /* Output the # of tasks and threads */
    printf("Thread:%5d of thread:%5d\n",iam,np); 
  
#pragma omp barrier
  
    /* integrate over the appropriate interval over np threads */
#pragma omp for schedule(static),reduction(+:pi)
    for(i=start_int;i<=end_int;i++)
    {
      x = n_1 * ((double)i - 0.5);
      pi+= (4./(1. + x*x));
    }
  } 
  
  mypi = n_1 * pi;
  return mypi;
}     
     
int main( argc, argv )
int argc;
char **argv;
{
  
  long long  nsize=501000000000;
  double     pi=0;
  const char *str;

  if (argc < 2) {
        fprintf(stderr, "Usage: %s <number_of_intervals> \n", argv[0]);
        exit(EXIT_FAILURE);
    }

   str = argv[1];
   nsize=atoll(str);
  
  printf("\n");
  
  /*  Compute pi = integral(4/(1+x2) on the interval (0,1) */
  pi = do_some_integratin(nsize);
  
  /*  Collect all the partial sums */
  
  printf ("The value of pi is: %11.8f\n",pi);
  
  return 0;
}
