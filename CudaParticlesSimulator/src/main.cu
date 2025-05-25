#include "memutils.cuh"
#include "cudaParticleSimulator.cuh"

#include <fstream>
#include <cstdlib>
#include <type_traits>

#ifndef PARTICLE_NUM
    #define PARTICLE_NUM 10000
#endif

#ifndef THREADS_PER_BLOCK
    #define THREADS_PER_BLOCK 128
#endif

#ifndef DURATION
    #define DURATION 10
#endif

// 0->naive, 1->float3, 2->__constant__, 3->shared memory, 4->...
#ifndef VERSION
    #define VERSION 0
#endif

#if VERSION == 0
    using Real = double;
#else
    using Real = double3;
#endif

int main(int argc, char** argv) {

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <time_file> < <load_file> > <output_file>\n";
        return 1;
    }

    char *time_file = argv[1];

    ofstream time_stream(time_file, ios::out | ios::app);
    if (!time_stream.is_open()) {
        cerr << "Could not open time file: " << time_file << endl;
        return 1;
    }

    // host memory
    double *h_masses = (double*) malloc(PARTICLE_NUM * sizeof(double));

#if VERSION == 0
    double *h_x_pos = (double*) malloc(PARTICLE_NUM * sizeof(double));
    double *h_y_pos = (double*) malloc(PARTICLE_NUM * sizeof(double));
    double *h_z_pos = (double*) malloc(PARTICLE_NUM * sizeof(double));

    double *h_x_vel = (double*) malloc(PARTICLE_NUM * sizeof(double));
    double *h_y_vel = (double*) malloc(PARTICLE_NUM * sizeof(double));
    double *h_z_vel = (double*) malloc(PARTICLE_NUM * sizeof(double));
#else
    double3 *h_pos = (double3*) malloc(PARTICLE_NUM * sizeof(double3));
    double3 *h_vel = (double3*) malloc(PARTICLE_NUM * sizeof(double3));
#endif

    // initialize host memory
    int unused;
    cin >> unused;
    cin >> unused;

    for (int i = 0; i < PARTICLE_NUM; i++) {
#if VERSION == 0
        cin >> h_x_pos[i] >> h_y_pos[i] >> h_z_pos[i]
            >> h_x_vel[i] >> h_y_vel[i] >> h_z_vel[i]
#else
        cin >> h_pos[i].x >> h_pos[i].y >> h_pos[i].z
            >> h_vel[i].x >> h_vel[i].y >> h_vel[i].z
#endif
            >> h_masses[i];
    }

    // allocate and initialize device memory
    // TODO: declare __constant__
    double *d_masses = allocateAndCopy<double>(h_masses, PARTICLE_NUM);

#if VERSION == 0
    double *d_x_pos_old = allocateAndCopy<double>(h_x_pos, PARTICLE_NUM);
    double *d_y_pos_old = allocateAndCopy<double>(h_y_pos, PARTICLE_NUM);
    double *d_z_pos_old = allocateAndCopy<double>(h_z_pos, PARTICLE_NUM);

    double *d_x_vel_old = allocateAndCopy<double>(h_x_vel, PARTICLE_NUM); 
    double *d_y_vel_old = allocateAndCopy<double>(h_y_vel, PARTICLE_NUM);
    double *d_z_vel_old = allocateAndCopy<double>(h_z_vel, PARTICLE_NUM);

    double *d_x_pos_new = allocateAndNull<double>(PARTICLE_NUM);
    double *d_y_pos_new = allocateAndNull<double>(PARTICLE_NUM);
    double *d_z_pos_new = allocateAndNull<double>(PARTICLE_NUM);

    double *d_x_vel_new = allocateAndNull<double>(PARTICLE_NUM); 
    double *d_y_vel_new = allocateAndNull<double>(PARTICLE_NUM);
    double *d_z_vel_new = allocateAndNull<double>(PARTICLE_NUM);

    double *d_x_acc = allocateAndNull<double>(PARTICLE_NUM);
    double *d_y_acc = allocateAndNull<double>(PARTICLE_NUM);
    double *d_z_acc = allocateAndNull<double>(PARTICLE_NUM);
#else
    double3 *d_pos_old = allocateAndCopy<double3>(h_pos, PARTICLE_NUM);
    double3 *d_vel_old = allocateAndCopy<double3>(h_vel, PARTICLE_NUM);
    double3 *d_pos_new = allocateAndNull<double3>(PARTICLE_NUM);
    double3 *d_vel_new = allocateAndNull<double3>(PARTICLE_NUM);
    double3 *d_acc = allocateAndNull<double3>(PARTICLE_NUM);
#endif

    // copy masses from host to device
    cudaError_t result = cudaMemcpy(d_masses, h_masses, sizeof(double) * PARTICLE_NUM,
        cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cerr << "Could not copy the masses array to the device \n";
        return 0;
    }

    // copy positions and velocities from host to device
#if VERSION == 0
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
#else
    result = cudaMemcpy(d_pos_old, h_pos, sizeof(double3) * PARTICLE_NUM,
        cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cerr << "Could not copy the pos array to the device \n";
        return 0;
    }
    result = cudaMemcpy(d_vel_old, h_vel, sizeof(double3) * PARTICLE_NUM,
        cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cerr << "Could not copy the vel array to the device \n";
        return 0;
    }
#endif

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
#if VERSION == 0
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
#else
            d_pos_old,
            d_vel_old,
            d_pos_new,
            d_vel_new,
            d_acc
#endif
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
        time_stream << VERSION << " " << THREADS_PER_BLOCK << " " << PARTICLE_NUM << " " << iter << ": " << milliseconds << "ms" << endl;

        // log results
#if VERSION == 0
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
#else
        result = cudaMemcpy(h_pos, d_pos_new, sizeof(double3) * PARTICLE_NUM,
            cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            cerr << "Could not copy the pos array to the host \n";
            return 0;
        }
#endif

        for (int particle = 0; particle < PARTICLE_NUM; particle++) {
#if VERSION == 0
            cout << h_x_pos[particle] << " "
                << h_y_pos[particle] << " "
                << h_z_pos[particle] << " "
#else 
            cout << h_pos[particle].x << " "
                << h_pos[particle].y << " "
                << h_pos[particle].z << " "
#endif
                << endl;
        }
        cout << endl;
        // also velocities?

        // swap the old and new positions and velocities
#if VERSION == 0
        swap(d_x_pos_old, d_x_pos_new);
        swap(d_y_pos_old, d_y_pos_new);
        swap(d_z_pos_old, d_z_pos_new);

        swap(d_x_vel_old, d_x_vel_new);
        swap(d_y_vel_old, d_y_vel_new);
        swap(d_z_vel_old, d_z_vel_new);
#else
        swap(d_pos_old, d_pos_new);
        swap(d_vel_old, d_vel_new);
#endif
    }

    // Free memory
    free(h_masses);
    cudaFree(d_masses);

#if VERSION == 0
    free(h_x_pos);
    free(h_y_pos);
    free(h_z_pos);

    free(h_x_vel);
    free(h_y_vel);
    free(h_z_vel);

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
#else
    free(h_pos);
    free(h_vel);
    cudaFree(d_pos_old);
    cudaFree(d_vel_old);
    cudaFree(d_pos_new);
    cudaFree(d_vel_new);
    cudaFree(d_acc);
#endif

    // destroy Cuda Event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    time_stream.close();
    return 0;
}