#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "CL/cl.h"

//pick up device type from compiler command line or from 
//the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

double get_time(){
	static struct timeval 	tv0;
	double time_, time;

	gettimeofday(&tv0,(struct timezone*)0);
	time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
	time = time_/1000000;
	return(time);
}

float pi(int n) {

	float area= 0.0;
	float pi, x;
	int i;
	for(i=1; i<n; i++) {
		x = (i+0.5)/n;
		area += 4.0/(1.0 + x*x)/n;
	}

	return area;
}

int main(int argc, char **argv)
{
	//int i,
	int n;
	cl_mem dpi;
	float *pi_GPU_parcial, pi_GPU, pi_CPU;
	double t0, t1;
	//double x, area, pi;

///////
	// OpenCL host variables
	cl_uint num_devs_returned;
	cl_context_properties properties[3];
	cl_device_id device_id;
	cl_int err;
	cl_platform_id platform_id;
	cl_uint num_platforms_returned;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel kernel;
	size_t global[1];

	// variables used to read kernel source file
	FILE *fp;
	long filelen;
	long readlen;
	char *kernel_src;  // char string to hold kernel source

//////////

	if (argc>1){
		n = atoi(argv[1]);
	} else {
		printf("./exec n\n");
		return(-1);
	}

	pi_GPU_parcial = (float*)malloc(n*sizeof(float));


///////////////
	// read the kernel
	fp = fopen("pi.cl","r");
	fseek(fp,0L, SEEK_END);
	filelen = ftell(fp);
	rewind(fp);

	kernel_src = malloc(sizeof(char)*(filelen+1));
	readlen = fread(kernel_src,1,filelen,fp);
	if(readlen!= filelen)
	{
		printf("error reading file\n");
		exit(1);
	}
	
	// ensure the string is NULL terminated
	kernel_src[filelen+1]='\0';

	// Set up platform and GPU device

	cl_uint numPlatforms;

	// Find number of platforms
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (err != CL_SUCCESS || numPlatforms <= 0)
	{
		printf("Error: Failed to find a platform!\n%s\n",err_code(err));
		return EXIT_FAILURE;
	}

	// Get all platforms
	cl_platform_id Platform[numPlatforms];
	err = clGetPlatformIDs(numPlatforms, Platform, NULL);
	if (err != CL_SUCCESS || numPlatforms <= 0)
	{
		printf("Error: Failed to get the platform!\n%s\n",err_code(err));
		return EXIT_FAILURE;
	}

	// Secure a GPU
	int i;
	for (i = 0; i < numPlatforms; i++)
	{
		err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
		if (err == CL_SUCCESS)
		{
			break;
		}
	}

	if (device_id == NULL)
	{
		printf("Error: Failed to create a device group!\n%s\n",err_code(err));
		return EXIT_FAILURE;
	}

	err = output_device_info(device_id);

	// Create a compute context 
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n%s\n", err_code(err));
		return EXIT_FAILURE;
	}

	// Create a command queue
	cl_command_queue command = clCreateCommandQueue(context, device_id, 0, &err);
	if (!command)
	{
		printf("Error: Failed to create a command commands!\n%s\n", err_code(err));
		return EXIT_FAILURE;
	}

	// create program object from source. 
	// kernel_src contains source read from file earlier
	program = clCreateProgramWithSource(context, 1 ,(const char **)
                                          &kernel_src, NULL, &err);
	if (err != CL_SUCCESS)
	{	
		printf("Unable to create program object. Error Code=%d\n",err);
		exit(1);
	}       
	
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
        	printf("Build failed. Error Code=%d\n", err);

		size_t len;
		char buffer[2048];
		// get the build log
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                                  sizeof(buffer), buffer, &len);
		printf("--- Build Log -- \n %s\n",buffer);
		exit(1);
	}

	kernel = clCreateKernel(program, "calculatePi", &err);
	if (err != CL_SUCCESS)
	{	
		printf("Unable to create kernel object. Error Code=%d\n",err);
		exit(1);
	}

	// create buffer objects to input and output args of kernel function
	//darray1D       =   clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*n*n, array1D, NULL);
 	//darray1D  	= clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float)*n*n, NULL, NULL);
	dpi = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n*sizeof(float),  NULL, NULL);
	
	//darray1D_trans  = clCreateBuffer(context, CL_MEM_WRITE_ONLY , sizeof(float)*n*n, NULL, NULL);
	//err = clEnqueueWriteBuffer(command, darray1D, CL_TRUE, 0, sizeof(float)*n*n, array1D, 0, NULL, NULL);

	// set the kernel arguments
	if ( clSetKernelArg(kernel, 0, sizeof(cl_mem), &dpi) ||
           //clSetKernelArg(kernel, 2, sizeof(cl_uint), &n) ||
             clSetKernelArg(kernel, 1, sizeof(cl_uint), &n) != CL_SUCCESS)
	{
		printf("Unable to set kernel arguments. Error Code=%d\n",err);
		exit(1);
	}

	// set the global work dimension size
	global[0]= n;

	// Enqueue the kernel object with 
	// Dimension size = 2, 
	// global worksize = global, 
	// local worksize = NULL - let OpenCL runtime determine
	// No event wait list
	double t0d = getMicroSeconds();
	err = clEnqueueNDRangeKernel(command, kernel, 1, NULL, 
                                   global, NULL, 0, NULL, NULL);
	double t1d = getMicroSeconds();

	if (err != CL_SUCCESS)
	{	
		printf("Unable to enqueue kernel command. Error Code=%d\n",err);
		exit(1);
	}

	err = clEnqueueReadBuffer(command, dpi, CL_TRUE, 0, n*sizeof(float), pi_GPU_parcial, 0, NULL, NULL );  
	if (err != CL_SUCCESS)
	{	
		printf("Error enqueuing read buffer command. Error Code=%s\n",err_code(err));
		exit(1);
	}
	//RED
	pi_GPU = 0.0;
	for (i=1;i<n;i++){
		pi_GPU+=pi_GPU_parcial[i];
	}


	


//////

	t0 = get_time();
	pi_CPU = pi(n);
	t1 = get_time();
	printf("\n %f=%f!!\n", pi_GPU, pi_CPU);
	//if (pi_GPU != pi_CPU)
	//	printf("\n Pi Host-Device differs %f!=%f!!\n", pi_GPU, pi_CPU);
	//else
	//	printf("PI=%3.10f time=%lf (s)\n", pi, t1-t0);
	

	// clean up
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	free(kernel_src);
	free(pi_GPU_parcial);


	return 0;
}
