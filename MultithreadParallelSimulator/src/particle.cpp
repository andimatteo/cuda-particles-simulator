#include "particle.h"

void particle::calcAcceleration(const particle particle, double3& acceleration) const {

    float dist = sqrt(
            (particle.position.x - this->position.x) * (particle.position.x - this->position.x) +
            (particle.position.y - this->position.y) * (particle.position.y - this->position.y) +
            (particle.position.z - this->position.z) * (particle.position.z - this->position.z)
        );

    if (dist <= DIST_THRESHOLD) [[unlikely]] return;

    acceleration.x += G * particle.mass * (particle.position.x - this->position.x) / (dist*dist*dist);
    acceleration.y += G * particle.mass * (particle.position.y - this->position.y) / (dist*dist*dist);
    acceleration.z += G * particle.mass * (particle.position.z - this->position.z) / (dist*dist*dist);

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

