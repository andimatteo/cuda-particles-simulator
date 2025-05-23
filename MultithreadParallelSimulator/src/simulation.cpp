#include "simulation.h"
#include <omp.h>

simulation::simulation(int duration, int particleNum){
    this->particleNum = particleNum;
    this->duration = duration;
    this->particles = new particle[particleNum];
    this->oldParticles = new particle[particleNum];
}

void simulation::load_particles(){
    for (int i = 0; i < particleNum; ++i){
        cin >> oldParticles[i];
        particles[i].mass = oldParticles[i].mass;
   }
}

simulation::~simulation(){
    delete [] particles;
    delete [] oldParticles;
}

void sequentialSimulation::start_simulation(){
    for (int t = 0; t < duration; ++t){
        for (int i = 0; i < particleNum; ++i){
            double3 acceleration = {0,0,0};
            for (int j = 0; j < particleNum; ++j) {
                if (i != j) [[likely]]
                            oldParticles[i].calcAcceleration(oldParticles[j], acceleration);
            }
            oldParticles[i].newState(particles[i], acceleration);
        }
#ifdef DEBUG
        for (int i = 0; i < particleNum; ++i){
            cout << particles[i] << endl;
        }
        cout << endl;
#endif
    }
}


void parallelSimulation::start_simulation(){
    # pragma omp parallel num_threads(THREAD_NUM) proc_bind(close) shared(particles, oldParticles, particleNum, duration)
    {
        for (int t = 0; t < duration; ++t){
            # pragma omp for schedule(static)        
            for (int i = 0; i < particleNum; ++i){
                double3 acceleration = {0,0,0};
                for (int j = 0; j < particleNum; ++j) {
                    if (i != j) [[likely]]
                        oldParticles[i].calcAcceleration(oldParticles[j], acceleration);
                }
                oldParticles[i].newState(particles[i], acceleration);
            }
#ifdef DEBUG
            for (int i = 0; i < particleNum; ++i){
                cout << particles[i] << endl;
            }
            cout << endl;
#endif
        }
    }
}

void chunkSimulation::start_simulation(){
    # pragma omp parallel num_threads(THREAD_NUM) proc_bind(close) shared(particles, oldParticles, particleNum, duration)
    {
        for (int t = 0; t < duration; ++t){
            # pragma omp for schedule(static)
            for (int i = 0; i < particleNum; i += CHUNK_SIZE){
                double3 acceleration[CHUNK_SIZE];
                for (int k = 0; (k < CHUNK_SIZE) && (i + k < particleNum); ++k) {
                    acceleration[k].x = 0;
                    acceleration[k].y = 0;
                    acceleration[k].z = 0;
                }

                for (int j = 0; j < particleNum; ++j) {
                    for (int k = 0; (k < CHUNK_SIZE) && (i + k < particleNum); ++k) {
                        if (i + k != j) [[likely]]
                            oldParticles[i+k].calcAcceleration(oldParticles[j], acceleration[k]);
                    }
                }

                for (int k = 0; (k < CHUNK_SIZE) && (i + k < particleNum); ++k) {
                    oldParticles[i+k].newState(particles[i+k], acceleration[k]);
                }
            }
#ifdef DEBUG
            for (int i = 0; i < particleNum; ++i){
                cout << particles[i] << endl;
            }
            cout << endl;
#endif
        }
    }
}

