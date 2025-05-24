#ifndef CODICE_PARTICLE_H
#define CODICE_PARTICLE_H

#ifndef EPS
    #define EPS 1e-10
#endif

#ifndef G
    #define G 6.674e-11
#endif

#ifndef STEP_TIME
    #define STEP_TIME 0.01
#endif

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

    void calcAcceleration(const particle particle, double3& acceleration) const;

    void newState(particle& target, double3 acceleration) const;
};

istream& operator>> (istream& is, particle& particle);

ostream& operator<< (ostream& os, const particle& particle);

#endif
