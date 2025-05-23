#include "memutils.cuh"
#include "cudaParticleSimulator.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include <cstdlib>

#ifndef PARTICLE_NUM
    #define PARTICLE_NUM 10000
#endif

#ifndef THREADS_PER_BLOCK
    #define THREADS_PER_BLOCK 128
#endif

#ifndef DURATION
    #define DURATION 10
#endif



int main(int argc, char** argv) {
    // TODO: manage arguments

    // host memory
    float64_t *h_masses = (float64_t*) malloc(PARTICLE_NUM * sizeof(float64_t));

    float64_t *h_x_pos = (float64_t*) malloc(PARTICLE_NUM * sizeof(float64_t));
    float64_t *h_y_pos = (float64_t*) malloc(PARTICLE_NUM * sizeof(float64_t));
    float64_t *h_z_pos = (float64_t*) malloc(PARTICLE_NUM * sizeof(float64_t));

    float64_t *h_x_vel = (float64_t*) malloc(PARTICLE_NUM * sizeof(float64_t));
    float64_t *h_y_vel = (float64_t*) malloc(PARTICLE_NUM * sizeof(float64_t));
    float64_t *h_z_vel = (float64_t*) malloc(PARTICLE_NUM * sizeof(float64_t));


    // initialize host memory
    int unused;
    cin >> unused;
    cin >> unused;

    for (int i = 0; i < PARTICLE_NUM; i++) {
        cin >> h_masses[i] 
            >> h_x_pos[i] >> h_y_pos[i] >> h_z_pos[i]
            >> h_x_vel[i] >> h_y_vel[i] >> h_z_vel[i];
    }

    // allocate and initialize device memory
    // TODO: declare __constant__
    float64_t *d_masses = allocateAndCopy<float64_t>(h_masses, PARTICLE_NUM);

    float64_t *d_x_pos_old = allocateAndCopy<float64_t>(h_x_pos, PARTICLE_NUM);
    float64_t *d_y_pos_old = allocateAndCopy<float64_t>(h_y_pos, PARTICLE_NUM);
    float64_t *d_z_pos_old = allocateAndCopy<float64_t>(h_z_pos, PARTICLE_NUM);

    float64_t *d_x_vel_old = allocateAndCopy<float64_t>(h_x_vel, PARTICLE_NUM); 
    float64_t *d_y_vel_old = allocateAndCopy<float64_t>(h_y_vel, PARTICLE_NUM);
    float64_t *d_z_vel_old = allocateAndCopy<float64_t>(h_z_vel, PARTICLE_NUM);

    float64_t *d_x_pos_new = allocateAndNull<float64_t>(PARTICLE_NUM);
    float64_t *d_y_pos_new = allocateAndNull<float64_t>(PARTICLE_NUM);
    float64_t *d_z_pos_new = allocateAndNull<float64_t>(PARTICLE_NUM);

    float64_t *d_x_vel_new = allocateAndNull<float64_t>(PARTICLE_NUM); 
    float64_t *d_y_vel_new = allocateAndNull<float64_t>(PARTICLE_NUM);
    float64_t *d_z_vel_new = allocateAndNull<float64_t>(PARTICLE_NUM);

    float64_t *d_x_acc = allocateAndNull<float64_t>(PARTICLE_NUM);
    float64_t *d_y_acc = allocateAndNull<float64_t>(PARTICLE_NUM);
    float64_t *d_z_acc = allocateAndNull<float64_t>(PARTICLE_NUM);

    // copy masses from host to device
    cudaError_t result = cudaMemcpy(d_masses, h_masses, sizeof(float64_t) * PARTICLE_NUM,
        cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cerr << "Could not copy the masses array to the device \n";
        return 0;
    }

    // copy positions and velocities from host to device
    result = cudaMemcpy(d_x_pos_old, h_x_pos, sizeof(float64_t) * PARTICLE_NUM,
        cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cerr << "Could not copy the x_pos array to the device \n";
        return 0;
    }

    result = cudaMemcpy(d_y_pos_old, h_y_pos, sizeof(float64_t) * PARTICLE_NUM,
    cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cerr << "Could not copy the y_pos array to the device \n";
        return 0;
    }

    result = cudaMemcpy(d_z_pos_old, h_z_pos, sizeof(float64_t) * PARTICLE_NUM,
    cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cerr << "Could not copy the z_pos array to the device \n";
        return 0;
    }

    result = cudaMemcpy(d_x_vel_old, h_x_vel, sizeof(float64_t) * PARTICLE_NUM,
    cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cerr << "Could not copy the x_vel array to the device \n";
        return 0;
    }

    result = cudaMemcpy(d_y_vel_old, h_y_vel, sizeof(float64_t) * PARTICLE_NUM,
    cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cerr << "Could not copy the y_vel array to the device \n";
        return 0;
    }

    result = cudaMemcpy(d_z_vel_old, h_z_vel, sizeof(float64_t) * PARTICLE_NUM,
    cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cerr << "Could not copy the z_vel array to the device \n";
        return 0;
    }

    for (int iter = 0; iter < DURATION; iter++) {
        // TODO: check time
        // set up the kernel launch parameters
        newState << <(PARTICLE_NUM + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(
            PARTICLE_NUM,
            d_masses,
            d_x_pos_old,
            d_y_pos_old,
            d_z_pos_old,
            d_x_vel_old,
            d_y_vel_old,
            d_z_vel_old,
            d_x_pos_new,
            d_y_pos_new,
            d_z_pos_new,
            d_x_vel_new,
            d_y_vel_new,
            d_z_vel_new,
            d_x_acc,
            d_y_acc,
            d_z_acc
        );

        // log results
        result = cudaMemcpy(h_x_pos, d_x_pos_new, sizeof(float64_t) * PARTICLE_NUM,
            cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            cerr << "Could not copy the x_pos array to the host \n";
            return 0;
        }
        result = cudaMemcpy(h_y_pos, d_y_pos_new, sizeof(float64_t) * PARTICLE_NUM,
            cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            cerr << "Could not copy the y_pos array to the host \n";
            return 0;
        }
        result = cudaMemcpy(h_z_pos, d_z_pos_new, sizeof(float64_t) * PARTICLE_NUM,
            cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            cerr << "Could not copy the z_pos array to the host \n";
            return 0;
        }

        for (int particle = 0; particle < PARTICLE_NUM; particle++) {
            cout << h_x_pos[particle] << " "
                << h_y_pos[particle] << " "
                << h_z_pos[particle] << " "
                << endl;
        }
        cout << endl;
        // also velocities?

        // swap the old and new positions and velocities
        swap(d_x_pos_old, d_x_pos_new);
        swap(d_y_pos_old, d_y_pos_new);
        swap(d_z_pos_old, d_z_pos_new);

        swap(d_x_vel_old, d_x_vel_new);
        swap(d_y_vel_old, d_y_vel_new);
        swap(d_z_vel_old, d_z_vel_new);
    }
    
}