#include "particle.h"
#include <iostream>
using namespace std;

class simulation{
public:
    int duration;
    int particleNum;
    particle * particles;
    particle * oldParticles;

    virtual void start_simulation() = 0;
    
    void load_particles();
    simulation();
    simulation(int duration, int particleNum);
    ~simulation();
};

class sequentialSimulation : public simulation {
public:
    sequentialSimulation(int duration, int particleNum) : simulation(duration, particleNum) {}
    void start_simulation();  
};

class parallelSimulation : public simulation {
public:
    parallelSimulation(int duration, int particleNum) : simulation(duration, particleNum) {}
    void start_simulation();
};

class chunkSimulation : public simulation {
public:
    chunkSimulation(int duration, int particleNum) : simulation(duration, particleNum) {}
    void start_simulation();
};

