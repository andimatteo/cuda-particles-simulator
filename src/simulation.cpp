#include "simulation.h"

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
            oldParticles[i].calcAcceleration(oldParticles, particleNum, acceleration);
            oldParticles[i].newState(particles[i], acceleration);

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
    for (int t = 0; t < duration; ++t){
        # pragma omp parallel for num_threads(8) schedule(static) shared(particles, oldParticles)
        for (int i = 0; i < particleNum; ++i){
            double3 acceleration = {0,0,0};
            oldParticles[i].calcAcceleration(oldParticles, particleNum, acceleration);
            oldParticles[i].newState(particles[i], acceleration);

            #ifdef DEBUG
            for (int i = 0; i < particleNum; ++i){
                cout << particles[i] << endl;
            }
            cout << endl;
            #endif

        }
    }
}
