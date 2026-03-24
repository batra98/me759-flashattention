#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>

#define CUDA_CHECK(call) \
    do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
        fprintf(stderr,"CUDA error at %s:%d – %s\n",__FILE__,__LINE__, \
        cudaGetErrorString(_e)); exit(1); } } while(0)

struct GpuTimer {
    cudaEvent_t s_,e_;
    GpuTimer()  { cudaEventCreate(&s_); cudaEventCreate(&e_); }
    ~GpuTimer() { cudaEventDestroy(s_); cudaEventDestroy(e_); }
    void Start() { cudaEventRecord(s_,0); }
    float Stop() {
        cudaEventRecord(e_,0); cudaEventSynchronize(e_);
        float ms=0.f; cudaEventElapsedTime(&ms,s_,e_); return ms;
    }
};

static constexpr int D_HEAD = 64;
static constexpr int BR     = 32;
static constexpr int BC     = 32;

inline void fill_random(float* h, int n, unsigned seed=42) {
    srand(seed);
    for(int i=0;i<n;++i) h[i]=((float)rand()/(float)RAND_MAX)*0.1f;
}

inline double rmse(const float* a, const float* b, int n) {
    double s=0.0;
    for(int i=0;i<n;++i){ double d=a[i]-b[i]; s+=d*d; }
    return sqrt(s/n);
}
