#ifndef MEMUTILS_CUH
#define MEMUTILS_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;
double* allocateAndCopy(double* array, int length);
double* allocateAndNull(int length);

#endif