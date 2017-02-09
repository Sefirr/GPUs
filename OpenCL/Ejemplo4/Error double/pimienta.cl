// pimienta.cl
// Kernel source file for remove_noise

#define MAX_WINDOW_SIZE 5*5


void swap ( float* a, float* b )
{
    float t = *a;
    *a = *b;
    *b = t;
}
 
/* This function is same in both iterative and recursive*/
int partition (float arr[], int l, int h)
{
    int x = arr[h];
    int i = (l - 1);
 
    for (int j = l; j <= h- 1; j++)
    {
        if (arr[j] <= x)
        {
            i++;
            swap (&arr[i], &arr[j]);
        }
    }
    swap (&arr[i + 1], &arr[h]);
    return (i + 1);
}
 
/* A[] --> Array to be sorted, l  --> Starting index, h  --> Ending index */
void quickSortIterative (float *arr, int l, int h)
{
    // Create an auxiliary stack
    float stack[MAX_WINDOW_SIZE];
 
    // initialize top of stack
    int top = -1;
 
    // push initial values of l and h to stack
    stack[ ++top ] = l;
    stack[ ++top ] = h;
 
    // Keep popping from stack while is not empty
    while ( top >= 0 )
    {
        // Pop h and l
        h = stack[ top-- ];
        l = stack[ top-- ];
 
        // Set pivot element at its correct position in sorted array
        int p = partition( arr, l, h );
 
        // If there are elements on left side of pivot, then push left
        // side to stack
        if ( p-1 > l )
        {
            stack[ ++top ] = l;
            stack[ ++top ] = p - 1;
        }
 
        // If there are elements on right side of pivot, then push right
        // side to stack
        if ( p+1 < h )
        {
            stack[ ++top ] = p + 1;
            stack[ ++top ] = h;
        }
    
    }
}



__kernel
void remove_noise(__global float *im, __global float *image_out, 
	const float thredshold, const int window_size,
	const int height, const int width)
{
	int i, j, ii, jj;

	float window[MAX_WINDOW_SIZE];
	float median;
	int ws2 = (window_size-1)>>1; 
	
	int idy = get_global_id(0);
	int idx = get_global_id(1);
	//for(i=ws2; i<height-ws2; i++)
	if((idx >= ws2) && (idx < width-ws2))
		//for(j=ws2; j<width-ws2; j++)
		if((idy >= ws2) && (idy < height-ws2))
		{
			for (ii =-ws2; ii<=ws2; ii++)
				for (jj =-ws2; jj<=ws2; jj++)
					window[(ii+ws2)*window_size + jj+ws2] = im[(i+ii)*width + j+jj];

			// SORT
			//partition_gpu(window,0,window_size*window_size-1);
			quickSortIterative (window, 0, window_size*window_size-1);
			median = window[(window_size*window_size-1)>>1+1];

			if (fabsf((median-im[i*width+j])/median) <=thredshold)
				image_out[i*width + j] = im[i*width+j];
			else
				image_out[i*width + j] = window[(window_size*window_size-1)>>1+1];

				
		}
}

