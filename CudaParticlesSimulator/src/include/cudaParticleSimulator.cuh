#ifndef CUDA_PARTICLE_SIMULATOR_CUH
#define CUDA_PARTICLE_SIMULATOR_CUH

#include "Particle.cuh"
#include <cuda.h>
#include <math.h>

#ifndef VERSION
    #define VERSION 0
#endif

#ifndef PARTICLE_NUM
    #define PARTICLE_NUM 10000
#endif

#ifndef THREADS_PER_BLOCK
    #define THREADS_PER_BLOCK 128
#endif

constexpr int PARTICLE_NUM_PADDING = (PARTICLE_NUM + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK * THREADS_PER_BLOCK;
constexpr int NUM_TILES = (PARTICLE_NUM + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

__global__ void newState_0(
    Particle* particles_old,
    Particle* particles_new
);

__global__ void newState_1(
    Particle* particles_old,
    Particle* particles_new
);

__global__ void newState_2(
    const float* masses,
    const float* x_pos_old,
    const float* y_pos_old,
    const float* z_pos_old,
    const float* x_vel_old,
    const float* y_vel_old,
    const float* z_vel_old,
    float* x_pos_new,
    float* y_pos_new,
    float* z_pos_new,
    float* x_vel_new,
    float* y_vel_new,
    float* z_vel_new
);

__global__ void newState_3(
    const float* masses,
    const float* x_pos_old,
    const float* y_pos_old,
    const float* z_pos_old,
    const float* x_vel_old,
    const float* y_vel_old,
    const float* z_vel_old,
    float* x_pos_new,
    float* y_pos_new,
    float* z_pos_new,
    float* x_vel_new,
    float* y_vel_new,
    float* z_vel_new
);

__global__ void newState_4(
    const float* masses,
    const float* x_pos_old,
    const float* y_pos_old,
    const float* z_pos_old,
    const float* x_vel_old,
    const float* y_vel_old,
    const float* z_vel_old,
    float* x_pos_new,
    float* y_pos_new,
    float* z_pos_new,
    float* x_vel_new,
    float* y_vel_new,
    float* z_vel_new
);

__global__ void newState_5(
    const float* masses,
    const float* x_pos_old,
    const float* y_pos_old,
    const float* z_pos_old,
    const float* x_vel_old,
    const float* y_vel_old,
    const float* z_vel_old,
    float* x_pos_new,
    float* y_pos_new,
    float* z_pos_new,
    float* x_vel_new,
    float* y_vel_new,
    float* z_vel_new
);

#endif