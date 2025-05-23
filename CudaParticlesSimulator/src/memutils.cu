#include "memutils.cuh"

T* allocateAndCopy<T>(T* array, int length) {
	T* dev_array;
	cudaError_t result = cudaMalloc((void**)&dev_array, length * sizeof(T));
	if (result != cudaSuccess) {
        cerr << "Could not allocate the " << typeid(T).name() << " array \n";
		return 0;
	}
	result = cudaMemcpy(dev_array, array, sizeof(T) * length,
		cudaMemcpyHostToDevice);
	if (result != cudaSuccess) {
		cerr << "Could not copy the " << typeid(T).name() << " array to the device \n";
		return 0;
	}
	return dev_array;
}

T* allocateAndNull<T>(int length) {
    T* dev_array;
    cudaError_t result = cudaMalloc((void**)&dev_array, length * sizeof(T));
    if (result != cudaSuccess) {
        cerr << "Could not allocate the " << typeid(T).name() << " array \n";
        return 0;
    }
    result = cudaMemset(dev_array, 0, sizeof(T) * length);
    if (result != cudaSuccess) {
        cerr << "Could not set the " << typeid(T).name() << " array to zero \n";
        return 0;
    }
    return dev_array;
}