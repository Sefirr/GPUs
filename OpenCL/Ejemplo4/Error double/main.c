#include <stdio.h>
#include <stdlib.h>
//#include <time.h>
#include <math.h>

#include <malloc.h>
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

unsigned char *readBMP(char *file_name, char header[54], int *w, int *h)
{
	//Se abre el fichero en modo binario para lectura
	FILE *f=fopen(file_name, "rb");
	if (!f){
		perror(file_name); exit(1);
	}

	// Cabecera archivo imagen
	//***********************************
	//Devuelve cantidad de bytes leidos
	int n=fread(header, 1, 54, f);

	//Si no lee 54 bytes es que la imagen de entrada es demasiado pequenya
	if (n!=54)
		fprintf(stderr, "Entrada muy pequenia (%d bytes)\n", n), exit(1);

	//Si los dos primeros bytes no corresponden con los caracteres BM no es un fichero BMP
	if (header[0]!='B'|| header[1]!='M')
		fprintf(stderr, "No BMP\n"), exit(1);

	//El tamanyo de la imagen es el valor de la posicion 2 de la cabecera menos 54 bytes que ocupa esa cabecera
	int imagesize=*(int*)(header+2)-54;
	printf("Tamanio archivo = %d\n", imagesize);

	//Si la imagen tiene tamanyo negativo o es superior a 48MB la imagen se rechaza
	if (imagesize<=0|| imagesize > 0x3000000)
		fprintf(stderr, "Imagen muy grande: %d bytes\n", imagesize), exit(1);

	//Si la cabecera no tiene el tamanyo de 54 o el numero de bits por pixel es distinto de 24 la imagen se rechaza
	if (*(int*)(header+10)!=54|| *(short*)(header+28)!=24)
		fprintf(stderr, "No color 24-bit\n"), exit(1);
	
	//Cuando la posicion 30 del header no es 0, es que existe compresion por lo que la imagen no es valida
	if (*(int*)(header+30)!=0)
		fprintf(stderr, "Compresion no suportada\n"), exit(1);
	
	//Se recupera la altura y anchura de la cabecera
	int width=*(int*)(header+18);
	int height=*(int*)(header+22);
	//**************************************


	// Lectura de la imagen
	//*************************************
	unsigned char *image = (unsigned char*)malloc(imagesize+256+width*6); //Se reservan "imagesize+256+width*6" bytes y se devuelve un puntero a estos datos

	unsigned char *tmp;
	image+=128+width*3;
	if ((n=fread(image, 1, imagesize+1, f))!=imagesize)
		fprintf(stderr, "File size incorrect: %d bytes read insted of %d\n", n, imagesize), exit(1);

	fclose(f);
	printf("Image read correctly (width=%i height=%i, imagesize=%i).\n", width, height, imagesize);

	/* Output variables */
	*w = width;
	*h = height;

	return(image);
}

void writeBMP(float *imageFLOAT, char *file_name, char header[54], int width, int height)
{

	FILE *f;
	int i, n;

	int imagesize=*(int*)(header+2)-54;

	unsigned char *image = (unsigned char*)malloc(3*sizeof(unsigned char)*width*height);

	for (i=0;i<width*height;i++){
		image[3*i]   = imageFLOAT[i]; //R 
		image[3*i+1] = imageFLOAT[i]; //G
		image[3*i+2] = imageFLOAT[i]; //B
	}
	

	f=fopen(file_name, "wb");		//Se abre el fichero en modo binario de escritura
	if (!f){
		perror(file_name); 
		exit(1);
	}

	n=fwrite(header, 1, 54, f);		//Primeramente se escribe la cabecera de la imagen
	n+=fwrite(image, 1, imagesize, f);	//Y despues se escribe el resto de la imagen
	if (n!=54+imagesize)			//Si se han escrito diferente cantidad de bytes que la suma de la cabecera y el tamanyo de la imagen. Ha habido error
		fprintf(stderr, "Escritos %d de %d bytes\n", n, imagesize+54);
	fclose(f);

	free(image);

}


float *RGB2BW(unsigned char *imageUCHAR, int width, int height)
{
	int i, j;
	float *imageBW = (float *)malloc(sizeof(float)*width*height);

	unsigned char R, B, G;

	for (i=0; i<height; i++)
		for (j=0; j<width; j++)
		{
			R = imageUCHAR[3*(i*width+j)];
			G = imageUCHAR[3*(i*width+j)+1];
			B = imageUCHAR[3*(i*width+j)+2];

			imageBW[i*width+j] = 0.2989 * R + 0.5870 * G + 0.1140 * B;
		}

	return(imageBW);
}

#define MAX_WINDOW_SIZE 5*5

void mergeSort(float arr[],int low,int mid,int high){

    int i,m,k,l;
    float temp[MAX_WINDOW_SIZE];

    l=low;
    i=low;
    m=mid+1;

    while((l<=mid)&&(m<=high)){

         if(arr[l]<=arr[m]){
             temp[i]=arr[l];
             l++;
         }
         else{
             temp[i]=arr[m];
             m++;
         }
         i++;
    }

    if(l>mid){
         for(k=m;k<=high;k++){
             temp[i]=arr[k];
             i++;
         }
    }
    else{
         for(k=l;k<=mid;k++){
             temp[i]=arr[k];
             i++;
         }
    }
   
    for(k=low;k<=high;k++){
         arr[k]=temp[k];
    }
}


void partition(float arr[],int low,int high){

    int mid;

    if(low<high){
         mid=(low+high)/2;
         partition(arr,low,mid);
         partition(arr,mid+1,high);
         mergeSort(arr,low,mid,high);
    }
}



void remove_noise(float *im, float *image_out, 
	float thredshold, int window_size,
	int height, int width)
{
	int i, j, ii, jj;

	float window[MAX_WINDOW_SIZE];
	float median;
	int ws2 = (window_size-1)>>1; 

	for(i=ws2; i<height-ws2; i++)
		for(j=ws2; j<width-ws2; j++)
		{
			for (ii =-ws2; ii<=ws2; ii++)
				for (jj =-ws2; jj<=ws2; jj++)
					window[(ii+ws2)*window_size + jj+ws2] = im[(i+ii)*width + j+jj];

			// SORT
			partition(window,0,window_size*window_size-1);
			median = window[(window_size*window_size-1)>>1+1];

			if (fabsf((median-im[i*width+j])/median) <=thredshold)
				image_out[i*width + j] = im[i*width+j];
			else
				image_out[i*width + j] = window[(window_size*window_size-1)>>1+1];

				
		}
}

int remove_noise_opencl(float *imageBW, float *imageOUT, 
	float thredshold, int window_size,
	int height, int width)
{
			cl_mem im_buffer;
			cl_mem imageout_buffer;
			cl_mem thredshold1;

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
			size_t global[2];

			// variables used to read kernel source file
			FILE *fp;
			long filelen;
			long readlen;
			char *kernel_src;  // char string to hold kernel source
/****/
	/*
				To fill
				cannyGPU
			*/
			/*****/
			// read the kernel
			fp = fopen("pimienta.cl","r");
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

			kernel = clCreateKernel(program, "remove_noise", &err);
			if (err != CL_SUCCESS)
			{	
				printf("Unable to create kernel object. Error Code=%d\n",err);
				exit(1);
			}

			// create buffer objects to input and output args of kernel function
			float *jamon;
			jamon[0] = 0.1;
			im_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, width*height*sizeof(float),  imageBW, NULL);
			imageout_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width*height*sizeof(float),  NULL, NULL);
			thredshold1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float),jamon  , NULL);

			// set the kernel arguments
			if ( clSetKernelArg(kernel, 0, sizeof(cl_mem), &im_buffer) ||
			     clSetKernelArg(kernel, 1, sizeof(cl_uint), &imageout_buffer)  || 
			     clSetKernelArg(kernel, 2, sizeof(cl_uint), &thredshold1)  ||
			     clSetKernelArg(kernel, 3, sizeof(cl_uint), &window_size)  ||
			     clSetKernelArg(kernel, 4, sizeof(cl_uint), &height)  ||
			     clSetKernelArg(kernel, 5, sizeof(cl_uint), &width)  != CL_SUCCESS)
			{
				printf("Unable to set kernel arguments. Error Code=%d\n",err);
				exit(1);
			}

			// set the global work dimension size
			global[0]= width;
			global[1]= height;

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

			err = clEnqueueReadBuffer(command, imageout_buffer, CL_TRUE, 0, width*height*sizeof(float), imageOUT, 0, NULL, NULL );  
			if (err != CL_SUCCESS)
			{	
				printf("Error enqueuing read buffer command. Error Code=%s\n",err_code(err));
				exit(1);
			}
	
			// clean up
			clReleaseProgram(program);
			clReleaseKernel(kernel);
			clReleaseCommandQueue(command_queue);
			clReleaseContext(context);
			free(kernel_src);


			return 0;


		/*****/
	
}

void freeMemory(unsigned char *imageUCHAR, float *imageBW, float *imageOUT)
{
	//free(imageUCHAR);
	free(imageBW);
	free(imageOUT);

}	


int main(int argc, char **argv) {

	int width, height;
	int window_size;
	unsigned char *imageUCHAR;
	float *imageBW;

	char header[54];


	//Variables para calcular el tiempo
	double t0, t1;
	double cpu_time_used = 0.0;
/***/
	

	//Tener menos de 3 argumentos es incorrecto
	if (argc < 4) {
		fprintf(stderr, "Uso incorrecto de los parametros. exe  input.bmp output.bmp [cg]\n");
		exit(1);
	}


	// READ IMAGE & Convert image
	imageUCHAR = readBMP(argv[1], header, &width, &height);
	imageBW = RGB2BW(imageUCHAR, width, height);


	// Aux. memory
	float *imageOUT = (float *)malloc(sizeof(float)*width*height);




	////////////////
	// CANNY      //
	////////////////
	switch (argv[3][0]) {
		case 'c':
			t0 = get_time();
			remove_noise(imageBW, imageOUT,
				0.1, 3, height, width);
			t1 = get_time();
			printf("CPU Exection time %f ms.\n", t1-t0);
			break;
		case 'g':
			t0 = get_time();
			remove_noise_opencl(imageBW, imageOUT, 0.1, 3, height, width);
			break;
		default:
			printf("Not Implemented yet!!\n");


	}


	// WRITE IMAGE
	writeBMP(imageOUT, argv[2], header, width, height);

	freeMemory(imageUCHAR, imageBW, imageOUT);	
	return 0;
}

