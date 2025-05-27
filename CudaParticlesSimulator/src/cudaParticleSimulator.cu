#include "cudaParticleSimulator.cuh"

#ifndef EPS
    #define EPS 1e-10f
#endif

#ifndef G
    #define G 6.674e-11f
#endif

#ifndef STEP_TIME
    #define STEP_TIME 0.01f
#endif

#define TILE_WIDTH ((SHARED_SIZE) / (4 * 4 * 64))

// AoS global
__global__ void newState_0(
    Particle* particles_old,
    Particle* particles_new
) {
    // calculate the index of the current thread
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // set acc to zero
    particles_old[idx].acc = make_float3(0.0f, 0.0f, 0.0f);

    // calculate the acceleration of the current particle 
    // cycle through all other particles
    for (uint64_t particle = 0; particle < PARTICLE_NUM; particle++) {
        if (idx != particle) {
            float dx = particles_old[particle].pos.x - particles_old[idx].pos.x;
            float dy = particles_old[particle].pos.y - particles_old[idx].pos.y;
            float dz = particles_old[particle].pos.z - particles_old[idx].pos.z;

            float dist = sqrtf(dx * dx + dy * dy + dz * dz + EPS * EPS);
            
            float acc_mod = G * particles_old[particle].mass / (dist * dist);
            particles_old[idx].acc.x += acc_mod * dx / dist;
            particles_old[idx].acc.y += acc_mod * dy / dist;
            particles_old[idx].acc.z += acc_mod * dz / dist;
        }
    }

    // update the velocity and position of the current particle
    particles_new[idx].vel.x = particles_old[idx].vel.x + particles_old[idx].acc.x * STEP_TIME;
    particles_new[idx].vel.y = particles_old[idx].vel.y + particles_old[idx].acc.y * STEP_TIME;
    particles_new[idx].vel.z = particles_old[idx].vel.z + particles_old[idx].acc.z * STEP_TIME;

    particles_new[idx].pos.x = particles_old[idx].pos.x + particles_old[idx].vel.x * STEP_TIME + 0.5f * particles_old[idx].acc.x * STEP_TIME * STEP_TIME;
    particles_new[idx].pos.y = particles_old[idx].pos.y + particles_old[idx].vel.y * STEP_TIME + 0.5f * particles_old[idx].acc.y * STEP_TIME * STEP_TIME;
    particles_new[idx].pos.z = particles_old[idx].pos.z + particles_old[idx].vel.z * STEP_TIME + 0.5f * particles_old[idx].acc.z * STEP_TIME * STEP_TIME;
}




// SoA global
__global__ void newState_1(
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
    float* z_vel_new,
    float* x_acc,
    float* y_acc,
    float* z_acc
) {

    // calculate the index of the current thread
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // set acc to zero
    x_acc[idx] = 0.0f;
    y_acc[idx] = 0.0f;
    z_acc[idx] = 0.0f;

    // calculate the acceleration of the current particle 
    // cycle through all other particles
    for (uint64_t particle = 0; particle < PARTICLE_NUM; particle++) {
        if (idx != particle) {
            float dx = x_pos_old[particle] - x_pos_old[idx];
            float dy = y_pos_old[particle] - y_pos_old[idx];
            float dz = z_pos_old[particle] - z_pos_old[idx];

            float dist = sqrtf(dx * dx + dy * dy + dz * dz + EPS * EPS);
            
            float acc_mod = G * masses[particle] / (dist * dist);
            x_acc[idx] += acc_mod * dx / dist;
            y_acc[idx] += acc_mod * dy / dist;
            z_acc[idx] += acc_mod * dz / dist;
        }
    }

    // update the velocity and position of the current particle
    x_vel_new[idx] = x_vel_old[idx] + x_acc[idx] * STEP_TIME;
    y_vel_new[idx] = y_vel_old[idx] + y_acc[idx] * STEP_TIME;
    z_vel_new[idx] = z_vel_old[idx] + z_acc[idx] * STEP_TIME;

    x_pos_new[idx] = x_pos_old[idx] + x_vel_new[idx] * STEP_TIME + 0.5f * x_acc[idx] * STEP_TIME * STEP_TIME;
    y_pos_new[idx] = y_pos_old[idx] + y_vel_new[idx] * STEP_TIME + 0.5f * y_acc[idx] * STEP_TIME * STEP_TIME;
    z_pos_new[idx] = z_pos_old[idx] + z_vel_new[idx] * STEP_TIME + 0.5f * z_acc[idx] * STEP_TIME * STEP_TIME;
}



// __shared__ is 48KB
// save blockDim.x [32-1024] pos_old, acc (6 float) -> blockDim.x * 6 * 4 = [1536-49,152] bytes
// remaining 48KB - [1536-49,152] = [47616-0] bytes
// save pos_old, masses of TILE particles -> TILE * 4 * 4 = [47616-0] bytes
// TILE_WIDTH = remaining / 32 = [1488-0] particles

// SoA shared memory
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
    float* z_vel_new,
    float* x_acc,
    float* y_acc,
    float* z_acc
) {

    // calculate the index of the current thread
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t tidx = threadIdx.x;

    float register_x_pos_old;
    float register_y_pos_old;
    float register_z_pos_old;
    float register_x_acc;
    float register_y_acc;
    float register_z_acc;

#if TILE_WIDTH > 0
    __shared__ float cycle_x_pos_old_shared[TILE_WIDTH];
    __shared__ float cycle_y_pos_old_shared[TILE_WIDTH];
    __shared__ float cycle_z_pos_old_shared[TILE_WIDTH];
    __shared__ float cycle_masses_shared[TILE_WIDTH];
#endif


    // set acc to zero
    register_x_acc = 0.0f;
    register_y_acc = 0.0f;
    register_z_acc = 0.0f;

    // load position of the thread into shared memory
    register_x_pos_old = x_pos_old[idx];
    register_y_pos_old = y_pos_old[idx];
    register_z_pos_old = z_pos_old[idx];

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
                float dx = cycle_x_pos_old_shared[particle] - register_x_pos_old;
                float dy = cycle_y_pos_old_shared[particle] - register_y_pos_old;
                float dz = cycle_z_pos_old_shared[particle] - register_z_pos_old;

                float dist = sqrtf(dx * dx + dy * dy + dz * dz + EPS * EPS);
                
                float acc_mod = G * cycle_masses_shared[particle] / (dist * dist);
                register_x_acc += acc_mod * dx / dist;
                register_y_acc += acc_mod * dy / dist;
                register_z_acc += acc_mod * dz / dist;
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
            float dx = x_pos_old[particle] - register_x_pos_old;
            float dy = y_pos_old[particle] - register_y_pos_old;
            float dz = z_pos_old[particle] - register_z_pos_old;

            float dist = sqrtf(dx * dx + dy * dy + dz * dz + EPS * EPS);
            
            float acc_mod = G * masses[particle] / (dist * dist);
            register_x_acc += acc_mod * dx / dist;
            register_y_acc += acc_mod * dy / dist;
            register_z_acc += acc_mod * dz / dist;
        }
    }
#endif

    // update the velocity and position of the current particle
    float register_x_vel_new = x_vel_old[idx] + register_x_acc * STEP_TIME;
    x_vel_new[idx] = register_x_vel_new;
    float register_y_vel_new = y_vel_old[idx] + register_y_acc * STEP_TIME;
    y_vel_new[idx] = register_y_vel_new;
    float register_z_vel_new = z_vel_old[idx] + register_z_acc * STEP_TIME;
    z_vel_new[idx] = register_z_vel_new;

    x_pos_new[idx] = register_x_pos_old + register_x_vel_new * STEP_TIME + 0.5f * register_x_acc * STEP_TIME * STEP_TIME;
    y_pos_new[idx] = register_y_pos_old + register_y_vel_new * STEP_TIME + 0.5f * register_y_acc * STEP_TIME * STEP_TIME;
    z_pos_new[idx] = register_z_pos_old + register_z_vel_new * STEP_TIME + 0.5f * register_z_acc * STEP_TIME * STEP_TIME;
}