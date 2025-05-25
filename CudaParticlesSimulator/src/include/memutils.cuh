#ifndef MEMUTILS_CUH
#define MEMUTILS_CUH

#include <cuda_runtime.h>
#include <iostream>

using namespace std;

template <typename T>
T* allocateAndCopy(T* array, int length);

template <typename T>
T* allocateAndNull(int length);

#endif