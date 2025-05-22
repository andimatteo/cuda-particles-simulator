#include "simulation.h"

#include <iostream>
#include <string>
using namespace std;


int main(int, char *argv[]) {
    
    int choice = stoi(argv[1]);
    int duration;
    int particleNum;

    cin >> duration;
    cin >> particleNum;

    simulation * Sim;

    switch (choice) {
        case (0):
            Sim = new sequentialSimulation(duration, particleNum);
            break;
        case (1):
            Sim = new chunkSimulation(duration, particleNum);
            break;
    }

    Sim->load_particles();

    Sim->start_simulation();

}
