#include "attention_utils.cuh"
#include <cuda_runtime.h>
#include <cmath>

// ═══════════════════════════════════════════════════════════════════════════
//  Naive Attention — three-kernel baseline
//  Memory traffic: O(N²) — the N×N attention matrix lives in HBM throughout
// ═══════════════════════════════════════════════════════════════════════════

// Kernel 1: S = (Q @ K^T) / sqrt(d)   — writes full N×N to HBM
__global__ void qkt_kernel(const float* __restrict__ Q,
                            const float* __restrict__ K,
                            float*       __restrict__ S,
                            int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // query index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // key   index
    if (row >= N || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < d; ++k)
        acc += Q[row * d + k] * K[col * d + k];

    S[row * N + col] = acc * rsqrtf((float)d);
}

// Kernel 2: in-place row-wise numerically stable softmax on S
__global__ void softmax_kernel(float* __restrict__ S, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    float* row_ptr = S + row * N;

    // Pass 1: row max (numerical stability)
    float mx = -FLT_MAX;
    for (int j = 0; j < N; ++j)
        mx = fmaxf(mx, row_ptr[j]);

    // Pass 2: exponentiate and accumulate
    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        row_ptr[j] = expf(row_ptr[j] - mx);
        sum += row_ptr[j];
    }

    // Pass 3: normalize
    float inv_sum = 1.0f / sum;
    for (int j = 0; j < N; ++j)
        row_ptr[j] *= inv_sum;
}

// Kernel 3: O = P @ V   where P = softmax(S)
__global__ void pv_kernel(const float* __restrict__ P,
                           const float* __restrict__ V,
                           float*       __restrict__ O,
                           int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // output row
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // output col (head dim)
    if (row >= N || col >= d) return;

    float acc = 0.0f;
    for (int j = 0; j < N; ++j)
        acc += P[row * N + j] * V[j * d + col];

    O[row * d + col] = acc;
}

// ── Public entry point ─────────────────────────────────────────────────────
void naive_attention(const float* dQ, const float* dK, const float* dV,
                     float* dO, float* dS, int N, int d) {
    // Kernel 1: QK^T / sqrt(d)
    dim3 blk1(16, 16);
    dim3 grd1((N + 15) / 16, (N + 15) / 16);
    qkt_kernel<<<grd1, blk1>>>(dQ, dK, dS, N, d);

    // Kernel 2: softmax (row-wise)
    int blk2 = 256;
    int grd2  = (N + blk2 - 1) / blk2;
    softmax_kernel<<<grd2, blk2>>>(dS, N);

    // Kernel 3: P @ V
    dim3 blk3(16, 16);
    dim3 grd3((d + 15) / 16, (N + 15) / 16);
    pv_kernel<<<grd3, blk3>>>(dS, dV, dO, N, d);

    CUDA_CHECK(cudaDeviceSynchronize());
}
