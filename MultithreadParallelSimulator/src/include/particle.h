#ifndef CODICE_PARTICLE_H
#define CODICE_PARTICLE_H

#ifndef EPS
    #define EPS 1e-10f
#endif

#ifndef G
    #define G 6.674e-11f
#endif

#ifndef STEP_TIME
    #define STEP_TIME 0.01f
#endif

#include <fstream>
#include <cmath>
using namespace std;


struct float3{
    float x,y,z;
};

class particle {
public:
    float3 position;
    float3 velocity;
    float mass;

    void calcAcceleration(const particle particle, float3& acceleration) const;

    void newState(particle& target, float3 acceleration) const;
};

istream& operator>> (istream& is, particle& particle);

ostream& operator<< (ostream& os, const particle& particle);

#endif
