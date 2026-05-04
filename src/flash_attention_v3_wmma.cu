#include "attention_utils.cuh"
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

namespace wmma = nvcuda::wmma;

#ifndef CUDART_INF_F
#define CUDART_INF_F __int_as_float(0x7f800000)
#endif

// FlashAttention WMMA: 16×16 score tiles, FP16 Tensor Core path, FP32 softmax state.

template <int BR, int BC, int D>
__global__ void __launch_bounds__(128)
flash_attn_fwd_wmma(
        const float* __restrict__ Q,
        const float* __restrict__ K,
        const float* __restrict__ V,
        float*       __restrict__ O,
        int N)
{
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 16;
    constexpr int MMA_K = 16;

    __shared__ half  Qh[BR][D];
    __shared__ half  Kh[BC][D];
    __shared__ float Vf[BC][D];
    __shared__ float Stile[BR][BC];
    __shared__ half  Bpack[MMA_K * MMA_N];

    const int q_base = blockIdx.x * BR;
    const int lane   = threadIdx.x;

    float O_acc[D];
    float m_row = -FLT_MAX;
    float l_row = 0.0f;

    const int my_row = lane;
    const int q_idx  = q_base + my_row;

    if (my_row < BR && q_idx < N) {
        #pragma unroll
        for (int k = 0; k < D; ++k)
            O_acc[k] = 0.0f;
    }

    const float scale = rsqrtf((float)D);
    const int Tc = (N + BC - 1) / BC;

    // Load Q tile once (matches FlashAttention outer-loop reuse of Q block)
    for (int idx = lane; idx < BR * D; idx += 32) {
        int r = idx / D;
        int c = idx % D;
        int qi = q_base + r;
        Qh[r][c] = (qi < N) ? __float2half(Q[qi * D + c]) : __float2half(0.f);
    }
    __syncwarp();

    for (int j = 0; j < Tc; ++j) {
        const int kv_start  = j * BC;
        const int bc_actual = (kv_start + BC <= N) ? BC : (N - kv_start);
        if (bc_actual <= 0) break;

        for (int idx = lane; idx < BC * D; idx += 32) {
            int r = idx / D;
            int c = idx % D;
            int kj = kv_start + r;
            Kh[r][c] = (kj < N) ? __float2half(K[kj * D + c]) : __float2half(0.f);
            Vf[r][c] = (kj < N) ? V[kj * D + c] : 0.f;
        }
        __syncwarp();

        wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> c_frag;

        wmma::fill_fragment(c_frag, 0.0f);

        for (int kk = 0; kk < D; kk += MMA_K) {
            wmma::load_matrix_sync(a_frag, &Qh[0][kk], D);

            for (int idx = lane; idx < MMA_K * MMA_N; idx += 32) {
                int t  = idx % MMA_K;
                int jc = idx / MMA_K;
                Bpack[idx] = Kh[jc][kk + t];
            }
            __syncwarp();
            wmma::load_matrix_sync(b_frag, Bpack, MMA_K);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        wmma::store_matrix_sync(&Stile[0][0], c_frag, BC, wmma::mem_row_major);

        __syncwarp();

        for (int idx = lane; idx < BR * BC; idx += 32) {
            int r = idx / BC;
            int c = idx % BC;
            int qi = q_base + r;
            int key_ix = kv_start + c;
            if (c < bc_actual && qi < N && key_ix < N)
                Stile[r][c] = Stile[r][c] * scale;
            else
                Stile[r][c] = -CUDART_INF_F;
        }
        __syncwarp();

        if (lane < BR) {
            const int qi = q_base + lane;
            if (qi < N) {
                float S_row[BC];
                float m_tilde = -FLT_MAX;
                for (int jj = 0; jj < bc_actual; ++jj) {
                    float s = Stile[lane][jj];
                    if (!isinf(s))
                        m_tilde = fmaxf(m_tilde, s);
                }

                float l_tilde = 0.f;
                for (int jj = 0; jj < bc_actual; ++jj) {
                    float s = Stile[lane][jj];
                    if (isinf(s)) {
                        S_row[jj] = 0.f;
                        continue;
                    }
                    float e = expf(s - m_tilde);
                    S_row[jj] = e;
                    l_tilde += e;
                }

                float m_new = fmaxf(m_row, m_tilde);
                float alpha = expf(m_row - m_new);
                float beta  = expf(m_tilde - m_new);
                float l_new = alpha * l_row + beta * l_tilde;

                #pragma unroll
                for (int k = 0; k < D; ++k) {
                    float pv_k = 0.f;
                    for (int jj = 0; jj < bc_actual; ++jj)
                        pv_k += S_row[jj] * Vf[jj][k];
                    O_acc[k] = alpha * O_acc[k] + beta * pv_k;
                }
                m_row = m_new;
                l_row = l_new;
            }
        }
        __syncwarp();
    }

    if (lane < BR) {
        const int qi = q_base + lane;
        if (qi < N) {
            float inv_l = 1.0f / l_row;
            #pragma unroll
            for (int k = 0; k < D; ++k)
                O[qi * D + k] = O_acc[k] * inv_l;
        }
    }
}

void flash_attention_v3_wmma(const float* dQ, const float* dK, const float* dV,
                             float* dO, int N)
{
    constexpr int BR_ = BR_WMMA;
    constexpr int BC_ = BC_WMMA;
    constexpr int D_  = D_HEAD;
    const int Tr = (N + BR_ - 1) / BR_;
    dim3 block(32);
    flash_attn_fwd_wmma<BR_, BC_, D_><<<Tr, block>>>(dQ, dK, dV, dO, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void flash_attention_v3_wmma_db(const float* dQ, const float* dK, const float* dV,
                                float* dO, int N)
{
    flash_attention_v3_wmma(dQ, dK, dV, dO, N);
}
