#include "cudaParticleSimulator.cuh"

#ifndef EPS
    #define EPS 1e-10
#endif

#ifndef G
    #define G 6.674e-11
#endif

#ifndef STEP_TIME
    #define STEP_TIME 0.01
#endif

#define TILE_WIDTH ((SHARED_SIZE - THREADS_PER_BLOCK * 6 * 8) / (4 * 8))

// AoS global
__global__ void newState_0(
    Particle* particles_old,
    Particle* particles_new
) {
    // calculate the index of the current thread
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // set acc to zero
    particles_old[idx].acc = make_double3(0.0, 0.0, 0.0);

    // calculate the acceleration of the current particle 
    // cycle through all other particles
    for (uint64_t particle = 0; particle < PARTICLE_NUM; particle++) {
        if (idx != particle) {
            double dx = particles_old[particle].pos.x - particles_old[idx].pos.x;
            double dy = particles_old[particle].pos.y - particles_old[idx].pos.y;
            double dz = particles_old[particle].pos.z - particles_old[idx].pos.z;

            double dist = sqrt(dx * dx + dy * dy + dz * dz + EPS * EPS);
            
            double acc_mod = G * particles_old[particle].mass / (dist * dist);
            particles_old[idx].acc.x += acc_mod * dx / dist;
            particles_old[idx].acc.y += acc_mod * dy / dist;
            particles_old[idx].acc.z += acc_mod * dz / dist;
        }
    }

    // update the velocity and position of the current particle
    particles_new[idx].vel.x = particles_old[idx].vel.x + particles_old[idx].acc.x * STEP_TIME;
    particles_new[idx].vel.y = particles_old[idx].vel.y + particles_old[idx].acc.y * STEP_TIME;
    particles_new[idx].vel.z = particles_old[idx].vel.z + particles_old[idx].acc.z * STEP_TIME;

    particles_new[idx].pos.x = particles_old[idx].pos.x + particles_old[idx].vel.x * STEP_TIME + 0.5 * particles_old[idx].acc.x * STEP_TIME * STEP_TIME;
    particles_new[idx].pos.y = particles_old[idx].pos.y + particles_old[idx].vel.y * STEP_TIME + 0.5 * particles_old[idx].acc.y * STEP_TIME * STEP_TIME;
    particles_new[idx].pos.z = particles_old[idx].pos.z + particles_old[idx].vel.z * STEP_TIME + 0.5 * particles_old[idx].acc.z * STEP_TIME * STEP_TIME;
}




// SoA global
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
) {

    // calculate the index of the current thread
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // set acc to zero
    x_acc[idx] = 0;
    y_acc[idx] = 0;
    z_acc[idx] = 0;

    // calculate the acceleration of the current particle 
    // cycle through all other particles
    for (uint64_t particle = 0; particle < PARTICLE_NUM; particle++) {
        if (idx != particle) {
            double dx = x_pos_old[particle] - x_pos_old[idx];
            double dy = y_pos_old[particle] - y_pos_old[idx];
            double dz = z_pos_old[particle] - z_pos_old[idx];

            double dist = sqrt(dx * dx + dy * dy + dz * dz + EPS * EPS);
            
            double acc_mod = G * masses[particle] / (dist * dist);
            x_acc[idx] += acc_mod * dx / dist;
            y_acc[idx] += acc_mod * dy / dist;
            z_acc[idx] += acc_mod * dz / dist;
        }
    }

    // update the velocity and position of the current particle
    x_vel_new[idx] = x_vel_old[idx] + x_acc[idx] * STEP_TIME;
    y_vel_new[idx] = y_vel_old[idx] + y_acc[idx] * STEP_TIME;
    z_vel_new[idx] = z_vel_old[idx] + z_acc[idx] * STEP_TIME;

    x_pos_new[idx] = x_pos_old[idx] + x_vel_new[idx] * STEP_TIME + 0.5 * x_acc[idx] * STEP_TIME * STEP_TIME;
    y_pos_new[idx] = y_pos_old[idx] + y_vel_new[idx] * STEP_TIME + 0.5 * y_acc[idx] * STEP_TIME * STEP_TIME;
    z_pos_new[idx] = z_pos_old[idx] + z_vel_new[idx] * STEP_TIME + 0.5 * z_acc[idx] * STEP_TIME * STEP_TIME;
}



// __shared__ is 48KB
// save blockDim.x [32-1024] pos_old, acc (6 double) -> blockDim.x * 6 * 8 = [1536-49,152] bytes
// remaining 48KB - [1536-49,152] = [47616-0] bytes
// save pos_old, masses of TILE particles -> TILE * 4 * 8 = [47616-0] bytes
// TILE_WIDTH = remaining / 32 = [1488-0] particles

// SoA shared memory
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
) {

    // calculate the index of the current thread
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t tidx = threadIdx.x;

    __shared__ double thread_x_pos_old_shared[THREADS_PER_BLOCK];
    __shared__ double thread_y_pos_old_shared[THREADS_PER_BLOCK];
    __shared__ double thread_z_pos_old_shared[THREADS_PER_BLOCK];
    __shared__ double thread_x_acc_shared[THREADS_PER_BLOCK];
    __shared__ double thread_y_acc_shared[THREADS_PER_BLOCK];
    __shared__ double thread_z_acc_shared[THREADS_PER_BLOCK];

#if TILE_WIDTH > 0
    __shared__ double cycle_x_pos_old_shared[TILE_WIDTH];
    __shared__ double cycle_y_pos_old_shared[TILE_WIDTH];
    __shared__ double cycle_z_pos_old_shared[TILE_WIDTH];
    __shared__ double cycle_masses_shared[TILE_WIDTH];
#endif


    // set acc to zero
    thread_x_acc_shared[tidx] = 0.0;
    thread_y_acc_shared[tidx] = 0.0;
    thread_z_acc_shared[tidx] = 0.0;

    // load position of the thread into shared memory
    thread_x_pos_old_shared[tidx] = x_pos_old[idx];
    thread_y_pos_old_shared[tidx] = y_pos_old[idx];
    thread_z_pos_old_shared[tidx] = z_pos_old[idx];

#if TILE_WIDTH > 0
    // iterate over all tiles in PARTICLE_NUM particles
    for (uint64_t tile = 0; tile < (PARTICLE_NUM + TILE_WIDTH - 1) / TILE_WIDTH; tile++) {


        // load the current tile into shared memory
        // each thread loads TILE_WIDTH/THREADS_PER_BLOCK particles in coalesced mode
        for (uint64_t i = tidx; i < TILE_WIDTH; i += blockDim.x) {
            uint64_t particle = tile * TILE_WIDTH + i;

            // check if the particle is within bounds
            if (particle < PARTICLE_NUM) {
                cycle_x_pos_old_shared[i] = x_pos_old[particle];
                cycle_y_pos_old_shared[i] = y_pos_old[particle];
                cycle_z_pos_old_shared[i] = z_pos_old[particle];
                cycle_masses_shared[i] = masses[particle];
            }
        }

        // synchronize threads to ensure all data is loaded
        __syncthreads();

        // cycle all particles in the tile and compute the acceleration
        for (uint64_t particle = 0; particle < TILE_WIDTH; particle++) {
            // check if the particle is within bounds
            if (tile * TILE_WIDTH + particle < PARTICLE_NUM && idx != tile * TILE_WIDTH + particle) {
                double dx = cycle_x_pos_old_shared[particle] - thread_x_pos_old_shared[tidx];
                double dy = cycle_y_pos_old_shared[particle] - thread_y_pos_old_shared[tidx];
                double dz = cycle_z_pos_old_shared[particle] - thread_z_pos_old_shared[tidx];

                double dist = sqrt(dx * dx + dy * dy + dz * dz + EPS * EPS);
                
                double acc_mod = G * cycle_masses_shared[particle] / (dist * dist);
                thread_x_acc_shared[tidx] += acc_mod * dx / dist;
                thread_y_acc_shared[tidx] += acc_mod * dy / dist;
                thread_z_acc_shared[tidx] += acc_mod * dz / dist;
            }
        }

        // synchronize threads to ensure all data of the tile has been processed
        __syncthreads();
    }
#else
    // calculate the acceleration of the current particle 
    // cycle through all other particles
    for (uint64_t particle = 0; particle < PARTICLE_NUM; particle++) {
        if (idx != particle) {
            double dx = x_pos_old[particle] - thread_x_pos_old_shared[idx];
            double dy = y_pos_old[particle] - thread_y_pos_old_shared[idx];
            double dz = z_pos_old[particle] - thread_z_pos_old_shared[idx];

            double dist = sqrt(dx * dx + dy * dy + dz * dz + EPS * EPS);
            
            double acc_mod = G * masses[particle] / (dist * dist);
            thread_x_acc_shared[idx] += acc_mod * dx / dist;
            thread_y_acc_shared[idx] += acc_mod * dy / dist;
            thread_z_acc_shared[idx] += acc_mod * dz / dist;
        }
    }
#endif

    // update the velocity and position of the current particle
    double register_x_vel_new = x_vel_old[idx] + thread_x_acc_shared[tidx] * STEP_TIME;
    x_vel_new[idx] = register_x_vel_new;
    double register_y_vel_new = y_vel_old[idx] + thread_y_acc_shared[tidx] * STEP_TIME;
    y_vel_new[idx] = register_y_vel_new;
    double register_z_vel_new = z_vel_old[idx] + thread_z_acc_shared[tidx] * STEP_TIME;
    z_vel_new[idx] = register_z_vel_new;

    x_pos_new[idx] = thread_x_pos_old_shared[tidx] + register_x_vel_new * STEP_TIME + 0.5 * thread_x_acc_shared[tidx] * STEP_TIME * STEP_TIME;
    y_pos_new[idx] = thread_y_pos_old_shared[tidx] + register_y_vel_new * STEP_TIME + 0.5 * thread_y_acc_shared[tidx] * STEP_TIME * STEP_TIME;
    z_pos_new[idx] = thread_z_pos_old_shared[tidx] + register_z_vel_new * STEP_TIME + 0.5 * thread_z_acc_shared[tidx] * STEP_TIME * STEP_TIME;
}