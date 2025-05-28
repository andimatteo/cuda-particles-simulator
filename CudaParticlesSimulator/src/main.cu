#include "memutils.cuh"
#include "cudaParticleSimulator.cuh"

#include <fstream>
#include <cstdlib>
#include <type_traits>

#ifndef DURATION
    #define DURATION 10
#endif

// 0->Particle global, 1->SoA global, 3->shared memory AoS, 4->...
#ifndef VERSION
    #define VERSION 0
#endif

int main_0_1(ofstream &time_stream) {

    Particle *h_particles = (Particle*) malloc(PARTICLE_NUM * sizeof(Particle));

    // initialize host memory
    int unused;
    cin >> unused;
    cin >> unused;

    for (int i = 0; i < PARTICLE_NUM; i++) {
        cin >> h_particles[i].pos.x >> h_particles[i].pos.y >> h_particles[i].pos.z
            >> h_particles[i].vel.x >> h_particles[i].vel.y >> h_particles[i].vel.z
            >> h_particles[i].mass;
    }

    // allocate and initialize device memory
    // TODO: declare __constant__

    Particle *d_particles_old = allocateAndCopy<Particle>(h_particles, PARTICLE_NUM);
    Particle *d_particles_new = allocateAndNull<Particle>(PARTICLE_NUM);

    //set up Cuda Event for timing
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int iter = 0; iter < DURATION; iter++) {
        cudaEventRecord(start);
        // set up the kernel launch parameters
#if VERSION == 0
        newState_0 << <(PARTICLE_NUM + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(
#else 
        newState_1 << <(PARTICLE_NUM + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(
#endif
            d_particles_old,
            d_particles_new
        );
        cudaError_t result = cudaDeviceSynchronize();
        if (result != cudaSuccess) {
            cerr << "Kernel launch failed with error: " << cudaGetErrorString(result) << endl;
            return 0;
        }

        // log results
        result = cudaMemcpy(h_particles, d_particles_new, sizeof(Particle) * PARTICLE_NUM,
            cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            cerr << "Could not copy the particles array to the host \n";
            return 0;
        }

        // swap the old and new positions and velocities
        swap(d_particles_old, d_particles_new);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&milliseconds, start, stop);
        // print the time taken for this iteration
        time_stream << VERSION << " " << THREADS_PER_BLOCK << " " << PARTICLE_NUM << " " << iter << ": " << milliseconds << "ms" << endl;

        for (int particle = 0; particle < PARTICLE_NUM; particle++) {
            cout << h_particles[particle].pos.x << " "
                << h_particles[particle].pos.y << " "
                << h_particles[particle].pos.z << " "
                << endl;
        }
        cout << endl;
        // also velocities?
    }

    // Free memory
    free(h_particles);
    cudaFree(d_particles_old);
    cudaFree(d_particles_new);

    // destroy Cuda Event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    time_stream.close();
    return 0;
}



int main_2_3(ofstream &time_stream) {

    float *h_masses = (float*) malloc(PARTICLE_NUM * sizeof(float));
    float *h_x_pos = (float*) malloc(PARTICLE_NUM * sizeof(float));
    float *h_y_pos = (float*) malloc(PARTICLE_NUM * sizeof(float));
    float *h_z_pos = (float*) malloc(PARTICLE_NUM * sizeof(float));

    float *h_x_vel = (float*) malloc(PARTICLE_NUM * sizeof(float));
    float *h_y_vel = (float*) malloc(PARTICLE_NUM * sizeof(float));
    float *h_z_vel = (float*) malloc(PARTICLE_NUM * sizeof(float));

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

    float *d_masses = allocateAndCopy<float>(h_masses, PARTICLE_NUM);
    float *d_x_pos_old = allocateAndCopy<float>(h_x_pos, PARTICLE_NUM);
    float *d_y_pos_old = allocateAndCopy<float>(h_y_pos, PARTICLE_NUM);
    float *d_z_pos_old = allocateAndCopy<float>(h_z_pos, PARTICLE_NUM);

    float *d_x_vel_old = allocateAndCopy<float>(h_x_vel, PARTICLE_NUM); 
    float *d_y_vel_old = allocateAndCopy<float>(h_y_vel, PARTICLE_NUM);
    float *d_z_vel_old = allocateAndCopy<float>(h_z_vel, PARTICLE_NUM);

    float *d_x_pos_new = allocateAndNull<float>(PARTICLE_NUM);
    float *d_y_pos_new = allocateAndNull<float>(PARTICLE_NUM);
    float *d_z_pos_new = allocateAndNull<float>(PARTICLE_NUM);

    float *d_x_vel_new = allocateAndNull<float>(PARTICLE_NUM); 
    float *d_y_vel_new = allocateAndNull<float>(PARTICLE_NUM);
    float *d_z_vel_new = allocateAndNull<float>(PARTICLE_NUM);

    //set up Cuda Event for timing
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int iter = 0; iter < DURATION; iter++) {
        cudaEventRecord(start);
        // set up the kernel launch parameters
#if VERSION == 2
        newState_2 << <(PARTICLE_NUM + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(
#else
        newState_3 << <(PARTICLE_NUM + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(
#endif
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
            d_z_vel_new
        );
        cudaError_t result = cudaDeviceSynchronize();
        if (result != cudaSuccess) {
            cerr << "Kernel launch failed with error: " << cudaGetErrorString(result) << endl;
            return 0;
        }

        // log results
        result = cudaMemcpy(h_x_pos, d_x_pos_new, sizeof(float) * PARTICLE_NUM,
            cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            cerr << "Could not copy the x_pos array to the host \n";
            return 0;
        }
        result = cudaMemcpy(h_y_pos, d_y_pos_new, sizeof(float) * PARTICLE_NUM,
            cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            cerr << "Could not copy the y_pos array to the host \n";
            return 0;
        }
        result = cudaMemcpy(h_z_pos, d_z_pos_new, sizeof(float) * PARTICLE_NUM,
            cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            cerr << "Could not copy the z_pos array to the host \n";
            return 0;
        }

        // swap the old and new positions and velocities
        swap(d_x_pos_old, d_x_pos_new);
        swap(d_y_pos_old, d_y_pos_new);
        swap(d_z_pos_old, d_z_pos_new);

        swap(d_x_vel_old, d_x_vel_new);
        swap(d_y_vel_old, d_y_vel_new);
        swap(d_z_vel_old, d_z_vel_new);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&milliseconds, start, stop);

        // print the time taken for this iteration
        time_stream << VERSION << " " << THREADS_PER_BLOCK << " " << PARTICLE_NUM << " " << iter << ": " << milliseconds << "ms" << endl;

        for (int particle = 0; particle < PARTICLE_NUM; particle++) {
            cout << h_x_pos[particle] << " "
                << h_y_pos[particle] << " "
                << h_z_pos[particle] << " "
                << endl;
        }
        cout << endl;
    }

    // Free memory

    free(h_masses);
    cudaFree(d_masses);

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

    // destroy Cuda Event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    time_stream.close();
    return 0;
}

int main_4_5(ofstream &time_stream) {

    float *h_masses = (float*) malloc(PARTICLE_NUM_PADDING * sizeof(float));
    float *h_x_pos = (float*) malloc(PARTICLE_NUM_PADDING * sizeof(float));
    float *h_y_pos = (float*) malloc(PARTICLE_NUM_PADDING * sizeof(float));
    float *h_z_pos = (float*) malloc(PARTICLE_NUM_PADDING * sizeof(float));

    float *h_x_vel = (float*) malloc(PARTICLE_NUM_PADDING * sizeof(float));
    float *h_y_vel = (float*) malloc(PARTICLE_NUM_PADDING * sizeof(float));
    float *h_z_vel = (float*) malloc(PARTICLE_NUM_PADDING * sizeof(float));

    // initialize host memory
    int unused;
    cin >> unused;
    cin >> unused;

    for (int i = 0; i < PARTICLE_NUM; i++) {
        cin >> h_x_pos[i] >> h_y_pos[i] >> h_z_pos[i]
            >> h_x_vel[i] >> h_y_vel[i] >> h_z_vel[i]
            >> h_masses[i];
    }
    memset(h_masses + PARTICLE_NUM, 0, (PARTICLE_NUM_PADDING - PARTICLE_NUM) * sizeof(float));

    // allocate and initialize device memory

    float *d_masses = allocateAndCopy<float>(h_masses, PARTICLE_NUM_PADDING);
    float *d_x_pos_old = allocateAndCopy<float>(h_x_pos, PARTICLE_NUM_PADDING);
    float *d_y_pos_old = allocateAndCopy<float>(h_y_pos, PARTICLE_NUM_PADDING);
    float *d_z_pos_old = allocateAndCopy<float>(h_z_pos, PARTICLE_NUM_PADDING);

    float *d_x_vel_old = allocateAndCopy<float>(h_x_vel, PARTICLE_NUM_PADDING); 
    float *d_y_vel_old = allocateAndCopy<float>(h_y_vel, PARTICLE_NUM_PADDING);
    float *d_z_vel_old = allocateAndCopy<float>(h_z_vel, PARTICLE_NUM_PADDING);

    float *d_x_pos_new = allocateAndNull<float>(PARTICLE_NUM_PADDING);
    float *d_y_pos_new = allocateAndNull<float>(PARTICLE_NUM_PADDING);
    float *d_z_pos_new = allocateAndNull<float>(PARTICLE_NUM_PADDING);

    float *d_x_vel_new = allocateAndNull<float>(PARTICLE_NUM_PADDING); 
    float *d_y_vel_new = allocateAndNull<float>(PARTICLE_NUM_PADDING);
    float *d_z_vel_new = allocateAndNull<float>(PARTICLE_NUM_PADDING);

    //set up Cuda Event for timing
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int iter = 0; iter < DURATION; iter++) {
        cudaEventRecord(start);
        // set up the kernel launch parameters
#if VERSION == 4
        newState_4 << <NUM_TILES, THREADS_PER_BLOCK >> >(
#else
        newState_5 << <NUM_TILES, THREADS_PER_BLOCK >> >(
#endif
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
            d_z_vel_new
        );
        cudaError_t result = cudaDeviceSynchronize();
        if (result != cudaSuccess) {
            cerr << "Kernel launch failed with error: " << cudaGetErrorString(result) << endl;
            return 0;
        }

        // log results
        result = cudaMemcpy(h_x_pos, d_x_pos_new, sizeof(float) * PARTICLE_NUM,
            cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            cerr << "Could not copy the x_pos array to the host \n";
            return 0;
        }
        result = cudaMemcpy(h_y_pos, d_y_pos_new, sizeof(float) * PARTICLE_NUM,
            cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            cerr << "Could not copy the y_pos array to the host \n";
            return 0;
        }
        result = cudaMemcpy(h_z_pos, d_z_pos_new, sizeof(float) * PARTICLE_NUM,
            cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) {
            cerr << "Could not copy the z_pos array to the host \n";
            return 0;
        }

        // swap the old and new positions and velocities
        swap(d_x_pos_old, d_x_pos_new);
        swap(d_y_pos_old, d_y_pos_new);
        swap(d_z_pos_old, d_z_pos_new);

        swap(d_x_vel_old, d_x_vel_new);
        swap(d_y_vel_old, d_y_vel_new);
        swap(d_z_vel_old, d_z_vel_new);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&milliseconds, start, stop);

        // print the time taken for this iteration
        time_stream << VERSION << " " << THREADS_PER_BLOCK << " " << PARTICLE_NUM << " " << iter << ": " << milliseconds << "ms" << endl;        

        for (int particle = 0; particle < PARTICLE_NUM; particle++) {
            cout << h_x_pos[particle] << " "
                << h_y_pos[particle] << " "
                << h_z_pos[particle] << " "
                << endl;
        }
        cout << endl;
        // also velocities?
    }

    // Free memory

    free(h_masses);
    cudaFree(d_masses);

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

    // destroy Cuda Event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    time_stream.close();
    return 0;
}

int main(int argc, char** argv) {

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <time_file> < <load_file> > <output_file>\n";
        return 1;
    }

    char *time_file = argv[1];

    ofstream time_stream(time_file, ios::out | ios::app);
    if (!time_stream.is_open()) {
        cerr << "Could not open time file: " << time_file << endl;
        return 1;
    }

// 3 versions
#if VERSION == 0 || VERSION == 1
    return main_0_1(time_stream);
#elif VERSION == 2 || VERSION == 3
    return main_2_3(time_stream);
#elif VERSION == 4 || VERSION == 5
    return main_4_5(time_stream);
#else
    cerr << "Invalid VERSION defined. Please set VERSION to 0, 1, or 2." << endl;
    return 1;
#endif
}