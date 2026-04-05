#include "attention_utils.cuh"
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

// ═══════════════════════════════════════════════════════════════════════════
//  FlashAttention v1 — Forward Pass
//  Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
//  with IO-Awareness", NeurIPS 2022, Algorithm 1.
//
//  Key idea: tile Q, K, V so the N×N score matrix NEVER touches HBM.
//  HBM traffic: O(N · d) instead of O(N²).
//
//  Layout : all matrices row-major, shape [N, D_HEAD]
//  Compile: sm_75, BR = BC = 32, D = 64
//           Shared-mem per block = (BR + 2·BC)·D·4 = 24 576 B  < 48 KB ✓
// ═══════════════════════════════════════════════════════════════════════════

template <int BR, int BC, int D>
__global__ void flash_attn_fwd(
        const float* __restrict__ Q,   // [N, D]
        const float* __restrict__ K,   // [N, D]
        const float* __restrict__ V,   // [N, D]
        float*       __restrict__ O,   // [N, D]
        int N)
{
    // ── Shared memory ─────────────────────────────────────────────────────
    __shared__ float Q_smem[BR][D];   // current Q tile  (stays for entire outer loop)
    __shared__ float K_smem[BC][D];   // current K tile  (reloaded each j)
    __shared__ float V_smem[BC][D];   // current V tile  (reloaded each j)

    const int q_tile = blockIdx.x;          // Q tile index 0..Tr-1
    const int t      = threadIdx.x;         // lane index   0..BR-1
    const int q_idx  = q_tile * BR + t;     // global row in Q

    // ── Load Q tile (persistent across entire outer loop) ─────────────────
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

    // ── Per-thread accumulators (in registers) ─────────────────────────────
    float O_acc[D];                  // unnormalized output accumulator
    #pragma unroll
    for (int k = 0; k < D; ++k) O_acc[k] = 0.0f;
    float m_i = -FLT_MAX;            // running row-max
    float l_i = 0.0f;                // running sum of exp(S - m)

    const float scale = rsqrtf((float)D);
    // Tile sizes BR=BC=32 tuned for sm_75 48KB SRAM budget
    const int   Tc    = (N + BC - 1) / BC;

    // ── Outer loop: iterate over K/V column tiles ─────────────────────────
    for (int j = 0; j < Tc; ++j) {
        const int kv_start  = j * BC;
        const int bc_actual = (kv_start + BC <= N) ? BC : (N - kv_start);

        // Load K_j and V_j — thread t loads rows {t, t+BR, t+2BR, …}
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
        __syncthreads();   // K_smem / V_smem ready

        // ── Compute S_row[jj] = dot(Q[q_idx], K[kv_start+jj]) * scale ────
        float S_row[BC];
        float m_tilde = -FLT_MAX;

        for (int jj = 0; jj < bc_actual; ++jj) {
            float dot = 0.0f;
            #pragma unroll
            for (int k = 0; k < D; ++k)
                dot += Q_smem[t][k] * K_smem[jj][k];
            S_row[jj] = dot * scale;
            m_tilde   = fmaxf(m_tilde, S_row[jj]);
        }

        // ── P_tilde = exp(S_row - m_tilde)  and  l_tilde = sum(P_tilde) ──
        float l_tilde = 0.0f;
        for (int jj = 0; jj < bc_actual; ++jj) {
            S_row[jj]  = expf(S_row[jj] - m_tilde);   // S_row now holds P_tilde
            l_tilde   += S_row[jj];
        }

        // ── Online softmax rescale ─────────────────────────────────────────
        //   m_new = max(m_i, m_tilde)
        //   l_new = exp(m_i - m_new)·l_i  +  exp(m_tilde - m_new)·l_tilde
        //   O_acc = alpha·O_acc  +  beta·(P_tilde @ V_j)
        float m_new = fmaxf(m_i, m_tilde);
        float alpha = expf(m_i     - m_new);   // rescale for old contribution
        float beta  = expf(m_tilde - m_new);   // rescale for new contribution
        float l_new = alpha * l_i + beta * l_tilde;

        // Update O_acc (unnormalized)
        #pragma unroll
        for (int k = 0; k < D; ++k) {
            float pv_k = 0.0f;
            for (int jj = 0; jj < bc_actual; ++jj)
                pv_k += S_row[jj] * V_smem[jj][k];
            O_acc[k] = alpha * O_acc[k] + beta * pv_k;
        }

        m_i = m_new;
        l_i = l_new;
        __syncthreads();   // guard smem before next tile load
    }

    // ── Write output: O = O_acc / l_i  (normalize once at the end) ────────
    if (q_idx < N) {
        float inv_l = 1.0f / l_i;
        #pragma unroll
        for (int k = 0; k < D; ++k)
            O[q_idx * D + k] = O_acc[k] * inv_l;
    }
}

// ── Public entry point ─────────────────────────────────────────────────────
void flash_attention_v1(const float* dQ, const float* dK, const float* dV,
                         float* dO, int N) {
    constexpr int BR_ = BR;   // from attention_utils.cuh  (32)
    constexpr int BC_ = BC;   // from attention_utils.cuh  (32)
    constexpr int D_  = D_HEAD;  // 64

    const int Tr = (N + BR_ - 1) / BR_;

    flash_attn_fwd<BR_, BC_, D_><<<Tr, BR_>>>(dQ, dK, dV, dO, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}
