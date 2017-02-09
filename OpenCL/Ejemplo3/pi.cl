// pi.cl
// Kernel source file for calculating pi

__kernel
void calculatePi(__global float *pi_parcial,
                     const    uint    n)

{
	int idx = get_global_id(0);
	float x = 0.0;
	if (idx>0 && idx<n) {
		x = (idx+0.5)/n;
		// Reduccion
		pi_parcial[idx] = (4.0/(1.0 + x*x))/n;
	}
	
}

