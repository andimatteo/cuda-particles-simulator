#ifndef PARTICLE_CUH
#define PARTICLE_CUH

#ifndef VERSION
    #define VERSION 0
#endif

struct Particle {
#if VERSION == 0
    double mass;
    double3 pos;
    double3 vel;
#else 
    float mass;
    float3 pos;
    float3 vel;   
#endif
};

#endif