#include "attention_utils.cuh"
#include <cuda_runtime.h>

// FlashAttention v1 — Dao et al. 2022, Algorithm 1
// TODO: implement tiled kernel with online softmax

template <int BR, int BC, int D>
__global__ void flash_attn_fwd(
        const float* __restrict__ Q,
        const float* __restrict__ K,
        const float* __restrict__ V,
        float*       __restrict__ O,
        int N)
{
    // TODO: shared-memory tiling + online softmax
    // Tile budget: (BR + 2*BC)*D*4 = 24576 B  < 48 KB (sm_75)
    (void)Q; (void)K; (void)V; (void)O; (void)N;
}

void flash_attention_v1(const float* dQ, const float* dK, const float* dV,
                         float* dO, int N)
{
    const int Tr = (N + BR - 1) / BR;
    flash_attn_fwd<BR, BC, D_HEAD><<<Tr, BR>>>(dQ, dK, dV, dO, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}
