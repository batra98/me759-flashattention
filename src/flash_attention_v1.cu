#include "attention_utils.cuh"
#include <cuda_runtime.h>
#include <cfloat>

template <int BR, int BC, int D>
__global__ void flash_attn_fwd(
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
        for (int k = 0; k < D; ++k) Q_smem[t][k] = Q[q_idx*D+k];
    } else {
        #pragma unroll
        for (int k = 0; k < D; ++k) Q_smem[t][k] = 0.f;
    }
    __syncthreads();

    float O_acc[D];
    #pragma unroll
    for (int k=0;k<D;++k) O_acc[k]=0.f;
    float m_i=-FLT_MAX, l_i=0.f;
    const float scale=rsqrtf((float)D);
    const int Tc=(N+BC-1)/BC;

    for (int j=0;j<Tc;++j) {
        const int kv_start=j*BC;
        const int bc_actual=(kv_start+BC<=N)?BC:(N-kv_start);

        for (int row=t;row<BC;row+=BR) {
            const int kv_idx=kv_start+row;
            if (kv_idx<N) {
                #pragma unroll
                for (int k=0;k<D;++k){
                    K_smem[row][k]=K[kv_idx*D+k];
                    V_smem[row][k]=V[kv_idx*D+k];
                }
            } else {
                #pragma unroll
                for (int k=0;k<D;++k){ K_smem[row][k]=0.f; V_smem[row][k]=0.f; }
            }
        }
        __syncthreads();

        float S_row[BC], m_tilde=-FLT_MAX;
        for (int jj=0;jj<bc_actual;++jj){
            float dot=0.f;
            #pragma unroll
            for (int k=0;k<D;++k) dot+=Q_smem[t][k]*K_smem[jj][k];
            S_row[jj]=dot*scale;
            m_tilde=fmaxf(m_tilde,S_row[jj]);
        }

        float l_tilde=0.f;
        for (int jj=0;jj<bc_actual;++jj){
            S_row[jj]=expf(S_row[jj]-m_tilde);
            l_tilde+=S_row[jj];
        }

        float m_new=fmaxf(m_i,m_tilde);
        float alpha=expf(m_i-m_new), beta=expf(m_tilde-m_new);
        float l_new=alpha*l_i+beta*l_tilde;

        #pragma unroll
        for (int k=0;k<D;++k){
            float pv=0.f;
            for (int jj=0;jj<bc_actual;++jj) pv+=S_row[jj]*V_smem[jj][k];
            O_acc[k]=alpha*O_acc[k]+beta*pv;
        }
        m_i=m_new; l_i=l_new;
        __syncthreads();
    }

    if (q_idx<N){
        float inv=1.f/l_i;
        #pragma unroll
        for (int k=0;k<D;++k) O[q_idx*D+k]=O_acc[k]*inv;
    }
}

void flash_attention_v1(const float* dQ,const float* dK,const float* dV,
                         float* dO,int N){
    const int Tr=(N+BR-1)/BR;
    flash_attn_fwd<BR,BC,D_HEAD><<<Tr,BR>>>(dQ,dK,dV,dO,N);
    CUDA_CHECK(cudaDeviceSynchronize());
}
