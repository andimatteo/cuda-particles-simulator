#ifndef CUDA_PARTICLE_SIMULATOR_CUH
#define CUDA_PARTICLE_SIMULATOR_CUH

#include <cuda.h>

__global__ void newState(
    const uint64_t particleNum,
    const float64_t* masses,
    const float64_t* x_pos_old,
    const float64_t* y_pos_old,
    const float64_t* z_pos_old,
    const float64_t* x_vel_old,
    const float64_t* y_vel_old,
    const float64_t* z_vel_old,
    float64_t* x_pos_new,
    float64_t* y_pos_new,
    float64_t* z_pos_new,
    float64_t* x_vel_new,
    float64_t* y_vel_new,
    float64_t* z_vel_new,
    float64_t* x_acc,
    float64_t* y_acc,
    float64_t* z_acc
);

#endif