#include "particle.h"
#include <iostream>
#include <fstream>

#define THREAD_NUM 12
#define CHUNK_SIZE 250

using namespace std;

class simulation{
public:
    int duration;
    int particleNum;
    int version;
    particle * particles;
    particle * oldParticles;

    virtual void start_simulation() = 0;
    
    void save_state(double t0, double t1, int it, const std::string& filename = "out.txt") const;

    void load_particles();
    simulation();
    simulation(int duration, int particleNum, int version);
    ~simulation();
};

class sequentialSimulation : public simulation {
public:
    sequentialSimulation(int duration, int particleNum) : simulation(duration, particleNum,0) {}
    void start_simulation();  
};

class parallelSimulation : public simulation {
public:
    parallelSimulation(int duration, int particleNum) : simulation(duration, particleNum,1) {}
    void start_simulation();
};

class chunkSimulation : public simulation {
public:
    chunkSimulation(int duration, int particleNum) : simulation(duration, particleNum,2) {}
    void start_simulation();
};

