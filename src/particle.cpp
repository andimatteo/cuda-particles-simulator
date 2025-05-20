#include "particle.h"

void particle::calcAcceleration(const particle* const particles, const int particleNum, float& a_x, float& a_y, float& a_z) const {

    for (int i = 0; i < particleNum; ++i) {

        float dist = sqrt(
                (particles[i].x - this->x) * (particles[i].x - this->x) +
                (particles[i].y - this->y) * (particles[i].y - this->y) +
                (particles[i].z - this->z) * (particles[i].z - this->z)
            );

        if (dist <= DIST_THRESHOLD) continue;

        a_x += G * particles[i].mass * this->mass * (particles[i].x - this->x) / (dist*dist*dist);
        a_y += G * particles[i].mass * this->mass * (particles[i].y - this->y) / (dist*dist*dist);
        a_z += G * particles[i].mass * this->mass * (particles[i].z - this->z) / (dist*dist*dist);

    }

}

void particle::newState(particle& target, float a_x, float a_y, float a_z) const {

    target.x = this->x + this->v_x * STEP_TIME + 0.5f * a_x * STEP_TIME * STEP_TIME;
    target.y = this->y + this->v_y * STEP_TIME + 0.5f * a_y * STEP_TIME * STEP_TIME;
    target.z = this->z + this->v_z * STEP_TIME + 0.5f * a_z * STEP_TIME * STEP_TIME;

    target.v_x = this->v_x + a_x * STEP_TIME;
    target.v_y = this->v_y + a_y * STEP_TIME;
    target.v_z = this->v_z + a_z * STEP_TIME;

}

istream& operator>> (istream& is, particle& particle)
{
    is  >> particle.x
        >> particle.y
        >> particle.z
        >> particle.v_x
        >> particle.v_y
        >> particle.v_z
        >> particle.mass;

    return is;
}

ostream& operator<< (ostream& os, const particle& particle)
{
    os  << particle.x << " "
        << particle.y << " "
        << particle.z << " "
        << particle.v_x << " "
        << particle.v_y << " "
        << particle.v_z << " "
        << particle.mass;

    return os;
}