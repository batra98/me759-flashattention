#include "attention_utils.cuh"
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// FlashAttention v2 — one warp per query row. Per-row O_acc / (m,l) live in shared memory
// so only ~32 lane-0 threads per block touch them (not 1024 × O_acc[D] registers).

static __device__ inline float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

template <int BR, int BC, int D>
__global__ void __launch_bounds__(1024, 2)
flash_attn_fwd_v2(
        const float* __restrict__ Q,
        const float* __restrict__ K,
        const float* __restrict__ V,
        float*       __restrict__ O,
        int N)
{
    __shared__ float Q_smem[BR][D];
    __shared__ float K_smem[BC][D];
    __shared__ float V_smem[BC][D];
    __shared__ float logits_tile[BR][BC];
    __shared__ float S_weight[BR][BC];
    __shared__ float O_acc_smem[BR][D];
    __shared__ float m_i_smem[BR];
    __shared__ float l_i_smem[BR];

    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int q_tile  = blockIdx.x;
    const int q_idx   = q_tile * BR + warp_id;

    if (warp_id < BR && lane == 0) {
        m_i_smem[warp_id] = -FLT_MAX;
        l_i_smem[warp_id] = 0.0f;
        #pragma unroll
        for (int k = 0; k < D; ++k)
            O_acc_smem[warp_id][k] = 0.0f;
    }

    if (lane == 0 && q_idx < N) {
        #pragma unroll
        for (int k = 0; k < D; ++k)
            Q_smem[warp_id][k] = Q[q_idx * D + k];
    } else if (lane == 0) {
        #pragma unroll
        for (int k = 0; k < D; ++k)
            Q_smem[warp_id][k] = 0.0f;
    }
    __syncthreads();

    const float scale = rsqrtf((float)D);
    const int Tc = (N + BC - 1) / BC;

    for (int j = 0; j < Tc; ++j) {
        const int kv_start  = j * BC;
        const int bc_actual = (kv_start + BC <= N) ? BC : (N - kv_start);

        for (int row = threadIdx.x; row < BC; row += blockDim.x) {
            const int kv_idx = kv_start + row;
            if (kv_idx < N) {
                #pragma unroll
                for (int k = 0; k < D; ++k) {
                    K_smem[row][k] = K[kv_idx * D + k];
                    V_smem[row][k] = V[kv_idx * D + k];
                }
            } else {
                #pragma unroll
                for (int k = 0; k < D; ++k) {
                    K_smem[row][k] = 0.0f;
                    V_smem[row][k] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (q_idx < N) {
            for (int jj = 0; jj < bc_actual; ++jj) {
                float partial = 0.0f;
                #pragma unroll
                for (int k = lane; k < D; k += 32)
                    partial += Q_smem[warp_id][k] * K_smem[jj][k];
                partial = warp_reduce_sum(partial);
                if (lane == 0)
                    logits_tile[warp_id][jj] = partial * scale;
            }
        }
        __syncthreads();

        if (lane == 0 && q_idx < N) {
            float m_i = m_i_smem[warp_id];
            float l_i = l_i_smem[warp_id];

            float m_tilde = -FLT_MAX;
            for (int jj = 0; jj < bc_actual; ++jj)
                m_tilde = fmaxf(m_tilde, logits_tile[warp_id][jj]);

            float l_tilde = 0.0f;
            for (int jj = 0; jj < bc_actual; ++jj) {
                float w = expf(logits_tile[warp_id][jj] - m_tilde);
                S_weight[warp_id][jj] = w;
                l_tilde += w;
            }

            float m_new = fmaxf(m_i, m_tilde);
            float alpha = expf(m_i - m_new);
            float beta  = expf(m_tilde - m_new);
            float l_new = alpha * l_i + beta * l_tilde;

            #pragma unroll
            for (int k = 0; k < D; ++k) {
                float pv_k = 0.0f;
                for (int jj = 0; jj < bc_actual; ++jj)
                    pv_k += S_weight[warp_id][jj] * V_smem[jj][k];
                O_acc_smem[warp_id][k] =
                    alpha * O_acc_smem[warp_id][k] + beta * pv_k;
            }
            m_i_smem[warp_id] = m_new;
            l_i_smem[warp_id] = l_new;
        }
        __syncthreads();
    }

    if (q_idx < N && lane == 0) {
        float inv_l = 1.0f / l_i_smem[warp_id];
        #pragma unroll
        for (int k = 0; k < D; ++k)
            O[q_idx * D + k] = O_acc_smem[warp_id][k] * inv_l;
    }
}

void flash_attention_v2(const float* dQ, const float* dK, const float* dV,
                        float* dO, int N) {
    constexpr int BR_ = BR_V2;
    constexpr int BC_ = BC_V2;
    constexpr int D_  = D_HEAD;
    const int Tr = (N + BR_ - 1) / BR_;
    dim3 block(BR_ * 32);
    flash_attn_fwd_v2<BR_, BC_, D_><<<Tr, block>>>(dQ, dK, dV, dO, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}
