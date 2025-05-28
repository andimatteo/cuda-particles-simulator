#include "simulation.h"

#include <iostream>
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
            Sim = new parallelSimulation(duration, particleNum);
            break;
        case (2):
            Sim = new chunkSimulation(duration, particleNum);
            break;
        default:
            return 1;
    }

    Sim->load_particles();

    Sim->start_simulation();

}
