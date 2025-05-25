#include "memutils.cuh"
#include "cudaParticleSimulator.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
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
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <version> <time_file>\n";
        return 1;
    }

    // Read the version and time file
    int version = atoi(argv[1]);
    char *time_file = argv[2];

    ofstream time_stream(time_file, ios::out | ios::app);
    if (!time_stream.is_open()) {
        cerr << "Could not open time file: " << time_file << endl;
        return 1;
    }

    // host memory
    double *h_masses = (double*) malloc(PARTICLE_NUM * sizeof(double));

    // TODO: use float3
    double *h_x_pos = (double*) malloc(PARTICLE_NUM * sizeof(double));
    double *h_y_pos = (double*) malloc(PARTICLE_NUM * sizeof(double));
    double *h_z_pos = (double*) malloc(PARTICLE_NUM * sizeof(double));

    double *h_x_vel = (double*) malloc(PARTICLE_NUM * sizeof(double));
    double *h_y_vel = (double*) malloc(PARTICLE_NUM * sizeof(double));
    double *h_z_vel = (double*) malloc(PARTICLE_NUM * sizeof(double));


    // initialize host memory
    int unused;
    cin >> unused;
    cin >> unused;

    for (int i = 0; i < PARTICLE_NUM; i++) {
        cin >> h_x_pos[i] >> h_y_pos[i] >> h_z_pos[i]
            >> h_x_vel[i] >> h_y_vel[i] >> h_z_vel[i]
            >> h_masses[i];
    }

    // allocate and initialize device memory
    // TODO: declare __constant__
    double *d_masses = allocateAndCopy(h_masses, PARTICLE_NUM);

    double *d_x_pos_old = allocateAndCopy(h_x_pos, PARTICLE_NUM);
    double *d_y_pos_old = allocateAndCopy(h_y_pos, PARTICLE_NUM);
    double *d_z_pos_old = allocateAndCopy(h_z_pos, PARTICLE_NUM);

    double *d_x_vel_old = allocateAndCopy(h_x_vel, PARTICLE_NUM); 
    double *d_y_vel_old = allocateAndCopy(h_y_vel, PARTICLE_NUM);
    double *d_z_vel_old = allocateAndCopy(h_z_vel, PARTICLE_NUM);

    double *d_x_pos_new = allocateAndNull(PARTICLE_NUM);
    double *d_y_pos_new = allocateAndNull(PARTICLE_NUM);
    double *d_z_pos_new = allocateAndNull(PARTICLE_NUM);

    double *d_x_vel_new = allocateAndNull(PARTICLE_NUM); 
    double *d_y_vel_new = allocateAndNull(PARTICLE_NUM);
    double *d_z_vel_new = allocateAndNull(PARTICLE_NUM);

    double *d_x_acc = allocateAndNull(PARTICLE_NUM);
    double *d_y_acc = allocateAndNull(PARTICLE_NUM);
    double *d_z_acc = allocateAndNull(PARTICLE_NUM);

    // copy masses from host to device
    cudaError_t result = cudaMemcpy(d_masses, h_masses, sizeof(double) * PARTICLE_NUM,
        cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cerr << "Could not copy the masses array to the device \n";
        return 0;
    }

    // copy positions and velocities from host to device
    result = cudaMemcpy(d_x_pos_old, h_x_pos, sizeof(double) * PARTICLE_NUM,
        cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cerr << "Could not copy the x_pos array to the device \n";
        return 0;
    }

    result = cudaMemcpy(d_y_pos_old, h_y_pos, sizeof(double) * PARTICLE_NUM,
    cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cerr << "Could not copy the y_pos array to the device \n";
        return 0;
    }

    result = cudaMemcpy(d_z_pos_old, h_z_pos, sizeof(double) * PARTICLE_NUM,
    cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cerr << "Could not copy the z_pos array to the device \n";
        return 0;
    }

    result = cudaMemcpy(d_x_vel_old, h_x_vel, sizeof(double) * PARTICLE_NUM,
    cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cerr << "Could not copy the x_vel array to the device \n";
        return 0;
    }

    result = cudaMemcpy(d_y_vel_old, h_y_vel, sizeof(double) * PARTICLE_NUM,
    cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cerr << "Could not copy the y_vel array to the device \n";
        return 0;
    }

    result = cudaMemcpy(d_z_vel_old, h_z_vel, sizeof(double) * PARTICLE_NUM,
    cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cerr << "Could not copy the z_vel array to the device \n";
        return 0;
    }

    //set up Cuda Event for timing
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int iter = 0; iter < DURATION; iter++) {
        cudaEventRecord(start);
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
        result = cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&milliseconds, start, stop);
        if (result != cudaSuccess) {
            cerr << "Kernel launch failed with error: " << cudaGetErrorString(result) << endl;
            return 0;
        }
        // print the time taken for this iteration
        time_stream << version << " " << THREADS_PER_BLOCK << " " << PARTICLE_NUM << " " << iter << ": " << milliseconds << "ms" << endl;

        // log results
        result = cudaMemcpy(h_x_pos, d_x_pos_new, sizeof(double) * PARTICLE_NUM,
            cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            cerr << "Could not copy the x_pos array to the host \n";
            return 0;
        }
        result = cudaMemcpy(h_y_pos, d_y_pos_new, sizeof(double) * PARTICLE_NUM,
            cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            cerr << "Could not copy the y_pos array to the host \n";
            return 0;
        }
        result = cudaMemcpy(h_z_pos, d_z_pos_new, sizeof(double) * PARTICLE_NUM,
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

    // Free memory
    free(h_masses);

    free(h_x_pos);
    free(h_y_pos);
    free(h_z_pos);

    free(h_x_vel);
    free(h_y_vel);
    free(h_z_vel);

    // free device memory
    cudaFree(d_masses);

    cudaFree(d_x_pos_old);
    cudaFree(d_y_pos_old);
    cudaFree(d_z_pos_old);

    cudaFree(d_x_vel_old);
    cudaFree(d_y_vel_old);
    cudaFree(d_z_vel_old);

    cudaFree(d_x_pos_new);
    cudaFree(d_y_pos_new);
    cudaFree(d_z_pos_new);

    cudaFree(d_x_vel_new);
    cudaFree(d_y_vel_new);
    cudaFree(d_z_vel_new);

    cudaFree(d_x_acc);
    cudaFree(d_y_acc);
    cudaFree(d_z_acc);

    // destroy Cuda Event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    time_stream.close();
    return 0;
}