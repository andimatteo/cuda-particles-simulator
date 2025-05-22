#include "particle.h"

void particle::calcAcceleration(const particle* const particles, const int particleNum, double3& acceleration) const {
    
    for (int i = 0; i < particleNum; ++i) {

        float dist = sqrt(
                (particles[i].position.x - this->position.x) * (particles[i].position.x - this->position.x) +
                (particles[i].position.y - this->position.y) * (particles[i].position.y - this->position.y) +
                (particles[i].position.z - this->position.z) * (particles[i].position.z - this->position.z)
            );

        if (dist <= DIST_THRESHOLD) continue;

        acceleration.x += G * particles[i].mass * (particles[i].position.x - this->position.x) / (dist*dist*dist);
        acceleration.y += G * particles[i].mass * (particles[i].position.y - this->position.y) / (dist*dist*dist);
        acceleration.z += G * particles[i].mass * (particles[i].position.z - this->position.z) / (dist*dist*dist);

    }

}

void particle::newState(particle& target, double3 acceleration) const {

    target.position.x = this->position.x + this->velocity.x * STEP_TIME + 0.5 * acceleration.x * STEP_TIME * STEP_TIME;
    target.position.y = this->position.y + this->velocity.y * STEP_TIME + 0.5 * acceleration.y * STEP_TIME * STEP_TIME;
    target.position.z = this->position.z + this->velocity.z * STEP_TIME + 0.5 * acceleration.z * STEP_TIME * STEP_TIME;

    target.velocity.x = this->velocity.x + acceleration.x * STEP_TIME;
    target.velocity.y = this->velocity.y + acceleration.y * STEP_TIME;
    target.velocity.z = this->velocity.z + acceleration.z * STEP_TIME;
    
}

istream& operator>> (istream& is, particle& particle)
{
    is  >> particle.position.x
        >> particle.position.y
        >> particle.position.z
        >> particle.velocity.x
        >> particle.velocity.y
        >> particle.velocity.z
        >> particle.mass;

    return is;
}

ostream& operator<< (ostream& os, const particle& particle)
{
    os  << particle.position.x << " "
        << particle.position.y << " "
        << particle.position.z << " "
        << particle.velocity.x << " "
        << particle.velocity.y << " "
        << particle.velocity.z << " "
        << particle.mass;

    return os;
}

