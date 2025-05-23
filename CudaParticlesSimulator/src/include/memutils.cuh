#ifndef MEMUTILS_CUH
#define MEMUTILS_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

using namespace std;

template <typename T>
T* allocateAndCopy(T* array, int length);

template <typename T>
T* allocateAndNull(int length);

#endif