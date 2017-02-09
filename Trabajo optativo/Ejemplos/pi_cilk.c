#include <stdio.h>

#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>

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
  
  
    cilk_for(i=start_int;i<=end_int;i++)
    {
      x = n_1 * ((double)i - 0.5);
      pi+= (4./(1. + x*x));
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
