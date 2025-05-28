#include "simulation.h"
#include <omp.h>

simulation::simulation(int duration, int particleNum, int version){
    this->particleNum = particleNum;
    this->duration = duration;
    this->particles = new particle[particleNum];
    this->oldParticles = new particle[particleNum];
    this->version = version;
}

void simulation::load_particles(){
    for (int i = 0; i < particleNum; ++i){
        cin >> oldParticles[i];
        particles[i].mass = oldParticles[i].mass;
   }
}

simulation::~simulation(){
    delete [] particles;
    delete [] oldParticles;
}

void sequentialSimulation::start_simulation(){
    for (int t = 0; t < duration; ++t){

        double t0 = omp_get_wtime();

        for (int i = 0; i < particleNum; ++i){
            float3 acceleration = {0.0f,0.0f,0.0f};
            for (int j = 0; j < particleNum; ++j) {
                //if (i != j) [[likely]]
                    oldParticles[i].calcAcceleration(oldParticles[j], acceleration);
            }
#ifdef FAST
            acceleration.x *= G;
            acceleration.y *= G;
            acceleration.z *= G;
#endif
            oldParticles[i].newState(particles[i], acceleration);
        }
        
        double t1 = omp_get_wtime();

        save_state(t0,t1,t);
#ifdef DEBUG
        for (int i = 0; i < particleNum; ++i){
            cout << particles[i] << endl;
        }
        cout << endl;
#endif
    }
}


void parallelSimulation::start_simulation(){
    for (int t = 0; t < duration; ++t){
 
        double t0 = omp_get_wtime();

        # pragma omp parallel for schedule(static) num_threads(THREAD_NUM) proc_bind(close) shared(particles, oldParticles, particleNum, duration)
        for (int i = 0; i < particleNum; ++i){
            float3 acceleration = {0.0f,0.0f,0.0f};
            for (int j = 0; j < particleNum; ++j) {
                //if (i != j) [[likely]]
                    oldParticles[i].calcAcceleration(oldParticles[j], acceleration);
            }
#ifdef FAST
            acceleration.x *= G;
            acceleration.y *= G;
            acceleration.z *= G;
#endif
            oldParticles[i].newState(particles[i], acceleration);
        }

        double t1 = omp_get_wtime();

        save_state(t0,t1,t);
#ifdef DEBUG
        for (int i = 0; i < particleNum; ++i){
            cout << particles[i] << endl;
        }
        cout << endl;
#endif

    }
}

void chunkSimulation::start_simulation(){
    
    for (int t = 0; t < duration; ++t){

        double t0 = omp_get_wtime();

        # pragma omp parallel for schedule(static) num_threads(THREAD_NUM) proc_bind(close) shared(particles, oldParticles, particleNum, duration)
        for (int i = 0; i < particleNum; i += CHUNK_SIZE){
            int size = CHUNK_SIZE < particleNum-i ? CHUNK_SIZE : particleNum-i;

            float3 acceleration[CHUNK_SIZE];
            for (int k = 0; k < size; ++k) {
                acceleration[k].x = 0.0f;
                acceleration[k].y = 0.0f;
                acceleration[k].z = 0.0f;
            }

            for (int j = 0; j < particleNum; ++j) {
                for (int k = 0; k < size; ++k) {
                    //if (i + k != j) [[likely]]
                        oldParticles[i+k].calcAcceleration(oldParticles[j], acceleration[k]);
                }
            }

            for (int k = 0; k < size; ++k) {
#ifdef FAST
                acceleration[k].x *= G;
                acceleration[k].y *= G;
                acceleration[k].z *= G;
#endif
                oldParticles[i+k].newState(particles[i+k], acceleration[k]);
            }
        }

        
        double t1 = omp_get_wtime();
        

        save_state(t0,t1,t);
#ifdef DEBUG
        for (int i = 0; i < particleNum; ++i){
            cout << particles[i] << endl;
        }
        cout << endl;
#endif

    }
}


void simulation::save_state(double t0, double t1, int it, const std::string& filename) const {
    ofstream out(filename, std::ios::out | std::ios::app);
    out << version << " " << THREAD_NUM << " " << CHUNK_SIZE << " " 
    #ifdef FAST
        << "F"
    #else
        << "S"
    #endif
        << " " << particleNum << " " << it << ':' << t1 - t0 << endl;
}

