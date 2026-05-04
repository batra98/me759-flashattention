#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>

// ── Error-checking macro ────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d – %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ── GPU Timer ──────────────────────────────────────────────────────────────
struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer()  {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void Start() { CUDA_CHECK(cudaEventRecord(start_, 0)); }
    float Stop() {
        CUDA_CHECK(cudaEventRecord(stop_, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// ── Compile-time kernel constants ─────────────────────────────────────────
static constexpr int D_HEAD = 64;  // head dimension (fixed for sm_75 shared-mem budget)
static constexpr int BR     = 32;  // query tile height (FlashAttention v1)
static constexpr int BC     = 32;  // key/value tile height (v1)
// Shared-mem usage: (BR + 2*BC) * D_HEAD * 4 = 96 * 64 * 4 = 24 576 B < 48 KB ✓

// FlashAttention v2 — one warp per row; BR=16 keeps shared memory (~26 KB) so more blocks
// can reside per SM on T4 than BR=32 (~41 KB). Tradeoff: 2× more Q tiles vs v1 (BR=32).
static constexpr int BR_V2 = 16;
static constexpr int BC_V2 = 32;

// WMMA path — 16×16 tiles, D=64 (four K-chunks)
static constexpr int BR_WMMA = 16;
static constexpr int BC_WMMA = 16;

// ── Host helpers ───────────────────────────────────────────────────────────

/// Fill array with small random floats (stable softmax inputs)
inline void fill_random(float* h, int n, unsigned seed = 42) {
    srand(seed);
    for (int i = 0; i < n; ++i)
        h[i] = ((float)rand() / (float)RAND_MAX) * 0.1f;
}

/// Root mean square error between two float arrays
inline double rmse(const float* a, const float* b, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = (double)a[i] - (double)b[i];
        acc += d * d;
    }
    return sqrt(acc / n);
}
