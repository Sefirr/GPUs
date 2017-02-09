// transpose_kernel.cl
// Kernel source file for calculating the transpose of a matrix

__kernel
void matrixTranspose(__global float * output,
                     __global float * input,
                     const    uint    width)

{
	int idx = get_global_id(0);
	int idy = get_global_id(1);
	if (idx<width && idy < width)
		output[idy*width+idx] = input[idx*width+idy]; 
}

