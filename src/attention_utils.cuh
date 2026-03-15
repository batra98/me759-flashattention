#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define CUDA_CHECK(call) \
    do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
        fprintf(stderr,"CUDA error %s:%d %s\n",__FILE__,__LINE__, \
        cudaGetErrorString(_e)); exit(1); } } while(0)

static constexpr int D_HEAD = 64;
static constexpr int BR     = 32;
static constexpr int BC     = 32;

inline void fill_random(float* h, int n, unsigned seed=42) {
    srand(seed);
    for (int i=0;i<n;++i) h[i]=((float)rand()/(float)RAND_MAX)*0.1f;
}
