#include "memutils.cuh"

double* allocateAndCopy(double* array, int length) {
	double* dev_array;
	cudaError_t result = cudaMalloc((void**)&dev_array, length * sizeof(double));
	if (result != cudaSuccess) {
        cerr << "Could not allocate the double array \n";
		return 0;
	}
	result = cudaMemcpy(dev_array, array, sizeof(double) * length,
		cudaMemcpyHostToDevice);
	if (result != cudaSuccess) {
		cerr << "Could not copy the double array to the device \n";
		return 0;
	}
	return dev_array;
}

double* allocateAndNull(int length) {
    double* dev_array;
    cudaError_t result = cudaMalloc((void**)&dev_array, length * sizeof(double));
    if (result != cudaSuccess) {
        cerr << "Could not allocate the double array \n";
        return 0;
    }
    result = cudaMemset(dev_array, 0, sizeof(double) * length);
    if (result != cudaSuccess) {
        cerr << "Could not set the double array to zero \n";
        return 0;
    }
    return dev_array;
}