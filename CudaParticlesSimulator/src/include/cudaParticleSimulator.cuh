#ifndef CUDA_PARTICLE_SIMULATOR_CUH
#define CUDA_PARTICLE_SIMULATOR_CUH

#include "Particle.cuh"
#include <cuda.h>

#ifndef VERSION
    #define VERSION 0
#endif

__global__ void newState(
    const uint64_t particleNum,
#if VERSION == 0
    const double* masses,
    const double* x_pos_old,
    const double* y_pos_old,
    const double* z_pos_old,
    const double* x_vel_old,
    const double* y_vel_old,
    const double* z_vel_old,
    double* x_pos_new,
    double* y_pos_new,
    double* z_pos_new,
    double* x_vel_new,
    double* y_vel_new,
    double* z_vel_new,
    double* x_acc,
    double* y_acc,
    double* z_acc
#else
    // const double3* pos_old,
    // const double3* vel_old,
    // double3* pos_new,
    // double3* vel_new,
    // double3* acc
    Particle* particles_old,
    Particle* particles_new
#endif
);

#endif