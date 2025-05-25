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

__global__ void newState(
    const uint64_t particleNum,
    const double* masses,
#if VERSION == 0
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
#else
    const double3* pos_old,
    const double3* vel_old,
    double3* pos_new,
    double3* vel_new,
    double3* acc
#endif
) {

    // calculate the index of the current thread
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // set acc to zero
#if VERSION == 0
    x_acc[idx] = 0;
    y_acc[idx] = 0;
    z_acc[idx] = 0;
#else
    acc[idx] = make_double3(0.0, 0.0, 0.0);
#endif

    // calculate the acceleration of the current particle 
    // cycle through all other particles
    for (uint64_t particle = 0; particle < particleNum; particle++) {
        if (idx != particle) {
#if VERSION == 0
            double dx = x_pos_old[particle] - x_pos_old[idx];
            double dy = y_pos_old[particle] - y_pos_old[idx];
            double dz = z_pos_old[particle] - z_pos_old[idx];
#else
            double dx = pos_old[particle].x - pos_old[idx].x;
            double dy = pos_old[particle].y - pos_old[idx].y;
            double dz = pos_old[particle].z - pos_old[idx].z;
#endif

            double dist = sqrt(dx * dx + dy * dy + dz * dz + EPS * EPS);
            
            double acc_mod = G * masses[particle] / (dist * dist);
#if VERSION == 0
            x_acc[idx] += acc_mod * dx / dist;
            y_acc[idx] += acc_mod * dy / dist;
            z_acc[idx] += acc_mod * dz / dist;
#else
            acc[idx].x += acc_mod * dx / dist;
            acc[idx].y += acc_mod * dy / dist;
            acc[idx].z += acc_mod * dz / dist;
#endif
        }
    }

    // update the velocity and position of the current particle
#if VERSION == 0
    x_vel_new[idx] = x_vel_old[idx] + x_acc[idx] * STEP_TIME;
    y_vel_new[idx] = y_vel_old[idx] + y_acc[idx] * STEP_TIME;
    z_vel_new[idx] = z_vel_old[idx] + z_acc[idx] * STEP_TIME;

    x_pos_new[idx] = x_pos_old[idx] + x_vel_new[idx] * STEP_TIME + 0.5 * x_acc[idx] * STEP_TIME * STEP_TIME;
    y_pos_new[idx] = y_pos_old[idx] + y_vel_new[idx] * STEP_TIME + 0.5 * y_acc[idx] * STEP_TIME * STEP_TIME;
    z_pos_new[idx] = z_pos_old[idx] + z_vel_new[idx] * STEP_TIME + 0.5 * z_acc[idx] * STEP_TIME * STEP_TIME;
#else
    vel_new[idx].x = vel_old[idx].x + acc[idx].x * STEP_TIME;
    vel_new[idx].y = vel_old[idx].y + acc[idx].y * STEP_TIME;
    vel_new[idx].z = vel_old[idx].z + acc[idx].z * STEP_TIME;

    pos_new[idx].x = pos_old[idx].x + vel_new[idx].x * STEP_TIME + 0.5 * acc[idx].x * STEP_TIME * STEP_TIME;
    pos_new[idx].y = pos_old[idx].y + vel_new[idx].y * STEP_TIME + 0.5 * acc[idx].y * STEP_TIME * STEP_TIME;
    pos_new[idx].z = pos_old[idx].z + vel_new[idx].z * STEP_TIME + 0.5 * acc[idx].z * STEP_TIME * STEP_TIME;
#endif
}