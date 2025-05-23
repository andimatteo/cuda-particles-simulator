#ifndef CODICE_PARTICLE_H
#define CODICE_PARTICLE_H

#define G 1.0f
#define STEP_TIME 1.0f
#define DIST_THRESHOLD 0.1f

#include <fstream>
#include <cmath>
using namespace std;


struct double3{
    double x,y,z;
};

class particle {
public:
    double3 position;
    double3 velocity;
    double mass;

    void calcAcceleration(const particle* const particles, const int particleNum, double3& acceleration) const;

    void newState(particle& target, double3 acceleration) const;
};

istream& operator>> (istream& is, particle& particle);

ostream& operator<< (ostream& os, const particle& particle);

#endif
