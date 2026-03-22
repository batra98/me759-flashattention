#include "attention_utils.cuh"
#include <cuda_runtime.h>
#include <cfloat>

__global__ void qkt_kernel(const float* Q,const float* K,float* S,int N,int d){
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=N||col>=N) return;
    float acc=0.f;
    for(int k=0;k<d;++k) acc+=Q[row*d+k]*K[col*d+k];
    S[row*N+col]=acc*rsqrtf((float)d);
}

// Fixed: subtract row max for numerical stability
__global__ void softmax_kernel(float* S,int N){
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=N) return;
    float* rp=S+row*N;
    float mx=-FLT_MAX;
    for(int j=0;j<N;++j) mx=fmaxf(mx,rp[j]);
    float sum=0.f;
    for(int j=0;j<N;++j){ rp[j]=expf(rp[j]-mx); sum+=rp[j]; }
    float inv=1.f/sum;
    for(int j=0;j<N;++j) rp[j]*=inv;
}

__global__ void pv_kernel(const float* P,const float* V,float* O,int N,int d){
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=N||col>=d) return;
    float acc=0.f;
    for(int j=0;j<N;++j) acc+=P[row*N+j]*V[j*d+col];
    O[row*d+col]=acc;
}

void naive_attention(const float* dQ,const float* dK,const float* dV,
                     float* dO,float* dS,int N,int d){
    dim3 b1(16,16); dim3 g1((N+15)/16,(N+15)/16);
    qkt_kernel<<<g1,b1>>>(dQ,dK,dS,N,d);
    softmax_kernel<<<(N+255)/256,256>>>(dS,N);
    dim3 b3(16,16); dim3 g3((d+15)/16,(N+15)/16);
    pv_kernel<<<g3,b3>>>(dS,dV,dO,N,d);
    CUDA_CHECK(cudaDeviceSynchronize());
}
