#include "particle.h"

#include <iostream>
#include <string>
using namespace std;


int main(int argc, char *argv[]) {
    int duration;
    int particleNum;
    particle* particles;
    particle* oldParticles;

    cin >> duration;

    cin >> particleNum;

    particles = new particle[particleNum];
    oldParticles = new particle[particleNum];

    for (int i = 0; i < particleNum; ++i) {
        cin >> oldParticles[i];
        particles[i].mass = oldParticles[i].mass;
    }

    for (int t = 0; t < duration; ++t) {

        for (int i = 0; i < particleNum; ++i) {

            float a_x  = 0;
            float a_y = 0;
            float a_z = 0;
            oldParticles[i].calcAcceleration(oldParticles, particleNum, a_x, a_y, a_z);

            oldParticles[i].newState(particles[i], a_x, a_y, a_z);

        }

        for (int i = 0; i < particleNum; ++i) {
            cout << particles[i] << endl;
        }
        cout << endl;

        particle* tmp = oldParticles;
        oldParticles = particles;
        particles = tmp;

    }

}