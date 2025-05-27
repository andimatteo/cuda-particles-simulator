#include "cudaParticleSimulator.cuh"

#if VERSION == 0
    #define EPS 1e-10
    #define G 6.674e-11
    #define STEP_TIME 10.0
#else
    #define EPS 1e-10f
    #define G 6.674e-11f
    #define STEP_TIME 10.0f
#endif

// AoS global FP64
__global__ void newState_0(
    Particle* particles_old,
    Particle* particles_new
) {
    // calculate the index of the current thread
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    Particle thread_particle = particles_old[idx];
    double3 acc = make_double3(0.0, 0.0, 0.0);

    // calculate the acceleration of the current particle 
    // cycle through all other particles
    for (uint64_t particle = 0; particle < PARTICLE_NUM; particle++) {
        if (idx != particle) {
            double dx = particles_old[particle].pos.x - thread_particle.pos.x;
            double dy = particles_old[particle].pos.y - thread_particle.pos.y;
            double dz = particles_old[particle].pos.z - thread_particle.pos.z;

            double dist = sqrt(dx * dx + dy * dy + dz * dz + EPS * EPS);
            
            double acc_mod = G * particles_old[particle].mass / (dist * dist);
            acc.x += acc_mod * dx / dist;
            acc.y += acc_mod * dy / dist;
            acc.z += acc_mod * dz / dist;
        }
    }

    // update the velocity and position of the current particle
    particles_new[idx].vel.x = thread_particle.vel.x + acc.x * STEP_TIME;
    particles_new[idx].vel.y = thread_particle.vel.y + acc.y * STEP_TIME;
    particles_new[idx].vel.z = thread_particle.vel.z + acc.z * STEP_TIME;

    particles_new[idx].pos.x = thread_particle.pos.x + thread_particle.vel.x * STEP_TIME + 0.5 * acc.x * STEP_TIME * STEP_TIME;
    particles_new[idx].pos.y = thread_particle.pos.y + thread_particle.vel.y * STEP_TIME + 0.5 * acc.y * STEP_TIME * STEP_TIME;
    particles_new[idx].pos.z = thread_particle.pos.z + thread_particle.vel.z * STEP_TIME + 0.5 * acc.z * STEP_TIME * STEP_TIME;
}


// AoS global FP32 e int
__global__ void newState_1(
    Particle* particles_old,
    Particle* particles_new
) {
    // calculate the index of the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    Particle thread_particle = particles_old[idx];
    float3 acc = make_float3(0.0f, 0.0f, 0.0f);

    // calculate the acceleration of the current particle 
    // cycle through all other particles
    for (int particle = 0; particle < PARTICLE_NUM; particle++) {
        if (idx != particle) {
            float dx = particles_old[particle].pos.x - thread_particle.pos.x;
            float dy = particles_old[particle].pos.y - thread_particle.pos.y;
            float dz = particles_old[particle].pos.z - thread_particle.pos.z;

            float dist = sqrtf(dx * dx + dy * dy + dz * dz + EPS * EPS);
            
            float acc_mod = G * particles_old[particle].mass / (dist * dist);
            acc.x += acc_mod * dx / dist;
            acc.y += acc_mod * dy / dist;
            acc.z += acc_mod * dz / dist;
        }
    }

    // update the velocity and position of the current particle
    particles_new[idx].vel.x = thread_particle.vel.x + acc.x * STEP_TIME;
    particles_new[idx].vel.y = thread_particle.vel.y + acc.y * STEP_TIME;
    particles_new[idx].vel.z = thread_particle.vel.z + acc.z * STEP_TIME;

    particles_new[idx].pos.x = thread_particle.pos.x + thread_particle.vel.x * STEP_TIME + 0.5f * acc.x * STEP_TIME * STEP_TIME;
    particles_new[idx].pos.y = thread_particle.pos.y + thread_particle.vel.y * STEP_TIME + 0.5f * acc.y * STEP_TIME * STEP_TIME;
    particles_new[idx].pos.z = thread_particle.pos.z + thread_particle.vel.z * STEP_TIME + 0.5f * acc.z * STEP_TIME * STEP_TIME;
}


// SoA global
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
) {

    // calculate the index of the current thread
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // set acc to zero
    float x_acc = 0.0f;
    float y_acc = 0.0f;
    float z_acc = 0.0f;

    float thread_x_pos_old = x_pos_old[idx];
    float thread_y_pos_old = y_pos_old[idx];
    float thread_z_pos_old = z_pos_old[idx];

    // calculate the acceleration of the current particle 
    // cycle through all other particles
    for (int particle = 0; particle < PARTICLE_NUM; particle++) {
        if (idx != particle) {
            float dx = x_pos_old[particle] - thread_x_pos_old;
            float dy = y_pos_old[particle] - thread_y_pos_old;
            float dz = z_pos_old[particle] - thread_z_pos_old;

            float dist = sqrtf(dx * dx + dy * dy + dz * dz + EPS * EPS);
            
            float acc_mod = G * masses[particle] / (dist * dist);
            x_acc += acc_mod * dx / dist;
            y_acc += acc_mod * dy / dist;
            z_acc += acc_mod * dz / dist;
        }
    }


    float thread_x_vel_old = x_vel_old[idx];
    float thread_y_vel_old = y_vel_old[idx];
    float thread_z_vel_old = z_vel_old[idx];

    // update the velocity and position of the current particle
    x_vel_new[idx] = thread_x_vel_old + x_acc * STEP_TIME;
    y_vel_new[idx] = thread_y_vel_old + y_acc * STEP_TIME;
    z_vel_new[idx] = thread_z_vel_old + z_acc * STEP_TIME;

    x_pos_new[idx] = thread_x_pos_old + thread_x_vel_old * STEP_TIME + 0.5f * x_acc * STEP_TIME * STEP_TIME;
    y_pos_new[idx] = thread_y_pos_old + thread_y_vel_old * STEP_TIME + 0.5f * y_acc * STEP_TIME * STEP_TIME;
    z_pos_new[idx] = thread_z_pos_old + thread_z_vel_old * STEP_TIME + 0.5f * z_acc * STEP_TIME * STEP_TIME;
}



// each thread loads 1 particle into shared memory
// SoA shared memory
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
) {

    // calculate the index of the current thread
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t tidx = threadIdx.x;

    float thread_x_pos_old = x_pos_old[idx];
    float thread_y_pos_old = y_pos_old[idx];
    float thread_z_pos_old = z_pos_old[idx];
    float x_acc = 0.0f;
    float y_acc = 0.0f;
    float z_acc = 0.0f;

    __shared__ float tile_x_pos_old_shared[THREADS_PER_BLOCK];
    __shared__ float tile_y_pos_old_shared[THREADS_PER_BLOCK];
    __shared__ float tile_z_pos_old_shared[THREADS_PER_BLOCK];
    __shared__ float tile_masses_shared[THREADS_PER_BLOCK];

    // iterate over all tiles in PARTICLE_NUM particles
    for (int tile = 0; tile < (PARTICLE_NUM + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; tile++) {
        // load the current tile into shared memory
        // each thread loads the corresponding element
        int shared_idx = tile * THREADS_PER_BLOCK + tidx;
        if (shared_idx < PARTICLE_NUM) {
            tile_x_pos_old_shared[tidx] = x_pos_old[shared_idx];
            tile_y_pos_old_shared[tidx] = y_pos_old[shared_idx];
            tile_z_pos_old_shared[tidx] = z_pos_old[shared_idx];
            tile_masses_shared[tidx] = masses[shared_idx];
        } else {
            // if the shared index is out of bounds, set to zero
            tile_x_pos_old_shared[tidx] = 0.0f;
            tile_y_pos_old_shared[tidx] = 0.0f;
            tile_z_pos_old_shared[tidx] = 0.0f;
            tile_masses_shared[tidx] = 0.0f;
        }

        // synchronize threads to ensure all data is loaded
        __syncthreads();

        // tile all particles in the tile and compute the acceleration
        for (int particle = 0; particle < THREADS_PER_BLOCK; particle++) {
            // check if the particle is within bounds
            if (tile * THREADS_PER_BLOCK + particle < PARTICLE_NUM && idx != tile * THREADS_PER_BLOCK + particle) {
                float dx = tile_x_pos_old_shared[particle] - thread_x_pos_old;
                float dy = tile_y_pos_old_shared[particle] - thread_y_pos_old;
                float dz = tile_z_pos_old_shared[particle] - thread_z_pos_old;

                float dist = sqrtf(dx * dx + dy * dy + dz * dz + EPS * EPS);
                
                float acc_mod = G * tile_masses_shared[particle] / (dist * dist);
                x_acc += acc_mod * dx / dist;
                y_acc += acc_mod * dy / dist;
                z_acc += acc_mod * dz / dist;
            }
        }

        // synchronize threads to ensure all data of the tile has been processed
        __syncthreads();
    }

    float thread_x_vel_old = x_vel_old[idx];
    float thread_y_vel_old = y_vel_old[idx];
    float thread_z_vel_old = z_vel_old[idx];

    // update the velocity and position of the current particle
    x_vel_new[idx] = thread_x_vel_old + x_acc * STEP_TIME;
    y_vel_new[idx] = thread_y_vel_old + y_acc * STEP_TIME;
    z_vel_new[idx] = thread_z_vel_old + z_acc * STEP_TIME;

    x_pos_new[idx] = thread_x_pos_old + thread_x_vel_old * STEP_TIME + 0.5f * x_acc * STEP_TIME * STEP_TIME;
    y_pos_new[idx] = thread_y_pos_old + thread_y_vel_old * STEP_TIME + 0.5f * y_acc * STEP_TIME * STEP_TIME;
    z_pos_new[idx] = thread_z_pos_old + thread_z_vel_old * STEP_TIME + 0.5f * z_acc * STEP_TIME * STEP_TIME;
}