#ifndef CUDA_PARTICLE_SIMULATOR_CUH
#define CUDA_PARTICLE_SIMULATOR_CUH

#include "Particle.cuh"
#include <cuda.h>

#ifndef VERSION
    #define VERSION 0
#endif

#ifndef PARTICLE_NUM
    #define PARTICLE_NUM 10000
#endif

#ifndef THREADS_PER_BLOCK
    #define THREADS_PER_BLOCK 128
#endif

#define SHARED_SIZE 48 * 1024 // 48 KB shared memory per block

__global__ void newState_0(
    Particle* particles_old,
    Particle* particles_new
);

__global__ void newState_1(
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
);

__global__ void newState_2(
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
);

#endif