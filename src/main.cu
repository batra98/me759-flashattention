#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include "attention_utils.cuh"

void naive_attention(const float*,const float*,const float*,float*,float*,int,int);
void flash_attention_v1(const float*,const float*,const float*,float*,int);

int main(int argc,char** argv){
    const char* mode="flash";
    int N=1024,d=D_HEAD,warmup=3,iters=10;
    bool csv=false;
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"--mode")    &&i+1<argc) mode  =argv[++i];
        if(!strcmp(argv[i],"--seq_len") &&i+1<argc) N     =atoi(argv[++i]);
        if(!strcmp(argv[i],"--warmup")  &&i+1<argc) warmup=atoi(argv[++i]);
        if(!strcmp(argv[i],"--iters")   &&i+1<argc) iters =atoi(argv[++i]);
        if(!strcmp(argv[i],"--csv"))                csv   =true;
    }
    int qkv=N*d;
    float *hQ=new float[qkv],*hK=new float[qkv],*hV=new float[qkv];
    fill_random(hQ,qkv,42); fill_random(hK,qkv,43); fill_random(hV,qkv,44);
    float *dQ,*dK,*dV,*dO;
    CUDA_CHECK(cudaMalloc(&dQ,qkv*4)); CUDA_CHECK(cudaMalloc(&dK,qkv*4));
    CUDA_CHECK(cudaMalloc(&dV,qkv*4)); CUDA_CHECK(cudaMalloc(&dO,qkv*4));
    CUDA_CHECK(cudaMemcpy(dQ,hQ,qkv*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK,hK,qkv*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV,hV,qkv*4,cudaMemcpyHostToDevice));

    if(!strcmp(mode,"correctness")){
        float *dOn,*dOf,*dS;
        CUDA_CHECK(cudaMalloc(&dOn,qkv*4));CUDA_CHECK(cudaMalloc(&dOf,qkv*4));
        CUDA_CHECK(cudaMalloc(&dS,(long)N*N*4));
        naive_attention(dQ,dK,dV,dOn,dS,N,d);
        flash_attention_v1(dQ,dK,dV,dOf,N);
        float *hn=new float[qkv],*hf=new float[qkv];
        CUDA_CHECK(cudaMemcpy(hn,dOn,qkv*4,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hf,dOf,qkv*4,cudaMemcpyDeviceToHost));
        double err=rmse(hn,hf,qkv);
        printf("N=%d RMSE=%.6e  %s\n",N,err,err<1e-3?"PASS":"FAIL");
        cudaFree(dOn);cudaFree(dOf);cudaFree(dS);delete[]hn;delete[]hf;
        goto done;
    }

    {
        float *dS=nullptr;
        if(!strcmp(mode,"naive")) CUDA_CHECK(cudaMalloc(&dS,(long)N*N*4));
        auto run=[&](){ if(!strcmp(mode,"naive")) naive_attention(dQ,dK,dV,dO,dS,N,d);
                        else flash_attention_v1(dQ,dK,dV,dO,N); };
        for(int w=0;w<warmup;++w) run();
        GpuTimer t; t.Start();
        for(int i=0;i<iters;++i) run();
        float ms=t.Stop()/iters;
        if(csv) printf("%s,%d,%d,%.4f\n",mode,N,d,ms);
        else    printf("mode=%-5s N=%5d d=%d time=%.4fms\n",mode,N,d,ms);
        if(dS) cudaFree(dS);
    }
done:
    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
    delete[]hQ;delete[]hK;delete[]hV;
    return 0;
}
