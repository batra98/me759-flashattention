#include "attention_utils.cuh"
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// FlashAttention v1 — causal forward (lower-triangular mask).
// Skips K/V tiles entirely after the last query row in the block can attend.

template <int BR, int BC, int D>
__global__ void flash_attn_fwd_causal(
        const float* __restrict__ Q,
        const float* __restrict__ K,
        const float* __restrict__ V,
        float*       __restrict__ O,
        int N)
{
    __shared__ float Q_smem[BR][D];
    __shared__ float K_smem[BC][D];
    __shared__ float V_smem[BC][D];

    const int q_tile = blockIdx.x;
    const int t      = threadIdx.x;
    const int q_idx  = q_tile * BR + t;

    if (q_idx < N) {
        #pragma unroll
        for (int k = 0; k < D; ++k)
            Q_smem[t][k] = Q[q_idx * D + k];
    } else {
        #pragma unroll
        for (int k = 0; k < D; ++k)
            Q_smem[t][k] = 0.0f;
    }
    __syncthreads();

    float O_acc[D];
    #pragma unroll
    for (int k = 0; k < D; ++k) O_acc[k] = 0.0f;
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    const float scale = rsqrtf((float)D);
    const int Tc    = (N + BC - 1) / BC;

    const int q_row_max_in_block = min(q_tile * BR + BR - 1, N - 1);

    for (int j = 0; j < Tc; ++j) {
        const int kv_start  = j * BC;
        const int bc_actual = (kv_start + BC <= N) ? BC : (N - kv_start);

        // Entire tile is strictly after every query row in this block → done.
        if (kv_start > q_row_max_in_block)
            break;

        for (int row = t; row < BC; row += BR) {
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

        float raw[BC];
        for (int jj = 0; jj < bc_actual; ++jj) {
            const int key_idx = kv_start + jj;
            float dot = 0.0f;
            #pragma unroll
            for (int k = 0; k < D; ++k)
                dot += Q_smem[t][k] * K_smem[jj][k];
            raw[jj] = (key_idx > q_idx) ? -INFINITY : (dot * scale);
        }

        bool any_valid = false;
        for (int jj = 0; jj < bc_actual; ++jj) {
            if (!isinf(raw[jj]))
                any_valid = true;
        }
        if (!any_valid) {
            __syncthreads();
            continue;
        }

        float m_tilde = -FLT_MAX;
        for (int jj = 0; jj < bc_actual; ++jj) {
            if (!isinf(raw[jj]))
                m_tilde = fmaxf(m_tilde, raw[jj]);
        }

        float S_row[BC];
        float l_tilde = 0.0f;
        for (int jj = 0; jj < bc_actual; ++jj) {
            if (isinf(raw[jj])) {
                S_row[jj] = 0.0f;
                continue;
            }
            S_row[jj] = expf(raw[jj] - m_tilde);
            l_tilde  += S_row[jj];
        }

        float m_new = fmaxf(m_i, m_tilde);
        float alpha = expf(m_i     - m_new);
        float beta  = expf(m_tilde - m_new);
        float l_new = alpha * l_i + beta * l_tilde;

        #pragma unroll
        for (int k = 0; k < D; ++k) {
            float pv_k = 0.0f;
            for (int jj = 0; jj < bc_actual; ++jj)
                pv_k += S_row[jj] * V_smem[jj][k];
            O_acc[k] = alpha * O_acc[k] + beta * pv_k;
        }

        m_i = m_new;
        l_i = l_new;
        __syncthreads();
    }

    if (q_idx < N) {
        float inv_l = 1.0f / l_i;
        #pragma unroll
        for (int k = 0; k < D; ++k)
            O[q_idx * D + k] = O_acc[k] * inv_l;
    }
}

void flash_attention_v1_causal(const float* dQ, const float* dK, const float* dV,
                               float* dO, int N) {
    constexpr int BR_ = BR;
    constexpr int BC_ = BC;
    constexpr int D_  = D_HEAD;
    const int Tr = (N + BR_ - 1) / BR_;
    flash_attn_fwd_causal<BR_, BC_, D_><<<Tr, BR_>>>(dQ, dK, dV, dO, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}
