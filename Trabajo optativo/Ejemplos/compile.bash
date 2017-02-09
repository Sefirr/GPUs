#/bin/bash
icc -openmp -O2 pi_omp.c -o pi_omp.cpu
icc -openmp -mmic -O2 pi_omp.c -o pi_omp.mic
icc -openmp -O2 pi_offload.c -o pi_offload.mic

icc -O2 pi_cilk.c -o pi_cilk.cpu 
icc -O2 pi_cilk_offload.c -o pi_cilk.mic
