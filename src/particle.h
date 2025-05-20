#ifndef CODICE_PARTICLE_H
#define CODICE_PARTICLE_H

#define G 1.0f
#define STEP_TIME 1.0f
#define DIST_THRESHOLD 0.1f

#include <fstream>
#include <cmath>
using namespace std;

class particle {
public:
    float x;
    float y;
    float z;
    float v_x;
    float v_y;
    float v_z;
    float mass;

    void calcAcceleration(const particle* const particles, const int particleNum, float& a_x, float& a_y, float& a_z) const;

    void newState(particle& target, float a_x, float a_y, float a_z) const;
};

istream& operator>> (istream& is, particle& particle);

ostream& operator<< (ostream& os, const particle& particle);

#endif
