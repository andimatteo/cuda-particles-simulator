#include "cudaParticleSimulator.cuh"

#ifndef EPS
    #define EPS 1e-10
#endif

#ifndef G
    #define G 6.674e-11
#endif

#ifndef STEP
    #define STEP 0.01
#endif

__global__ void newState(
    const uint64_t particleNum,
    const float64_t* masses,
    const float64_t* x_pos_old,
    const float64_t* y_pos_old,
    const float64_t* z_pos_old,
    const float64_t* x_vel_old,
    const float64_t* y_vel_old,
    const float64_t* z_vel_old,
    float64_t* x_pos_new,
    float64_t* y_pos_new,
    float64_t* z_pos_new,
    float64_t* x_vel_new,
    float64_t* y_vel_new,
    float64_t* z_vel_new,
    float64_t* x_acc,
    float64_t* y_acc,
    float64_t* z_acc
) {

    // calculate the index of the current thread
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // set acc to zero
    x_acc[idx] = 0;
    y_acc[idx] = 0;
    z_acc[idx] = 0;

    // calculate the acceleration of the current particle 
    // cycle through all other particles
    for (uint64_t particle = 0; particle < particleNum; particle++) {
        if (idx != particle) {
            float64_t dx = x_pos_old[particle] - x_pos_old[idx];
            float64_t dy = y_pos_old[particle] - y_pos_old[idx];
            float64_t dz = z_pos_old[particle] - z_pos_old[idx];

            float64_t dist = sqrt(dx * dx + dy * dy + dz * dz + EPS * EPS);
            
            float64_t force = G * masses[particle] / (dist * dist);
            x_acc[idx] += force * dx / dist;
            y_acc[idx] += force * dy / dist;
            z_acc[idx] += force * dz / dist;
        }
    }

    // update the velocity and position of the current particle
    x_vel_new[idx] = x_vel_old[idx] + x_acc[idx] * STEP;
    y_vel_new[idx] = y_vel_old[idx] + y_acc[idx] * STEP;
    z_vel_new[idx] = z_vel_old[idx] + z_acc[idx] * STEP;

    x_pos_new[idx] = x_pos_old[idx] + x_vel_new[idx] * STEP + 0.5 * x_acc[idx] * STEP * STEP;
    y_pos_new[idx] = y_pos_old[idx] + y_vel_new[idx] * STEP + 0.5 * y_acc[idx] * STEP * STEP;
    z_pos_new[idx] = z_pos_old[idx] + z_vel_new[idx] * STEP + 0.5 * z_acc[idx] * STEP * STEP;
}