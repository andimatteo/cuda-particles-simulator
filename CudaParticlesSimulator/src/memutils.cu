#include "memutils.cuh"

template <typename T>
T* allocateAndCopy(T* array, int length) {
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

template <typename T>
T* allocateAndNull(int length) {
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

template float* allocateAndCopy<float>(float* array, int length);
template float3* allocateAndCopy<float3>(float3* array, int length);
template Particle* allocateAndCopy<Particle>(Particle* array, int length);

template float* allocateAndNull<float>(int length);
template float3* allocateAndNull<float3>(int length);
template Particle* allocateAndNull<Particle>(int length);