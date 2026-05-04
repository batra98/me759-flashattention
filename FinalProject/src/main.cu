#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include "attention_utils.cuh"

void naive_attention(const float* dQ, const float* dK, const float* dV,
                     float* dO, float* dS, int N, int d);
void naive_attention_causal(const float* dQ, const float* dK, const float* dV,
                            float* dO, float* dS, int N, int d);
void flash_attention_v1(const float* dQ, const float* dK, const float* dV,
                          float* dO, int N);
void flash_attention_v1_causal(const float* dQ, const float* dK, const float* dV,
                                 float* dO, int N);
void flash_attention_v2(const float* dQ, const float* dK, const float* dV,
                        float* dO, int N);
void flash_attention_v3_wmma(const float* dQ, const float* dK, const float* dV,
                               float* dO, int N);
void flash_attention_v3_wmma_db(const float* dQ, const float* dK, const float* dV,
                                float* dO, int N);

static bool is_causal_target(const char* mode) {
    return strstr(mode, "causal") != nullptr;
}

static void run_target(const char* mode,
                       const float* dQ, const float* dK, const float* dV,
                       float* dO, int N, int d) {
    if (!strcmp(mode, "naive"))
        ; // handled elsewhere with dS
    else if (!strcmp(mode, "flash"))
        flash_attention_v1(dQ, dK, dV, dO, N);
    else if (!strcmp(mode, "flash_causal"))
        flash_attention_v1_causal(dQ, dK, dV, dO, N);
    else if (!strcmp(mode, "flash_v2"))
        flash_attention_v2(dQ, dK, dV, dO, N);
    else if (!strcmp(mode, "flash_wmma"))
        flash_attention_v3_wmma(dQ, dK, dV, dO, N);
    else if (!strcmp(mode, "flash_wmma_db"))
        flash_attention_v3_wmma_db(dQ, dK, dV, dO, N);
    else {
        fprintf(stderr, "Unknown --mode %s\n", mode);
        exit(1);
    }
}

static double rmse_threshold_for(const char* target) {
    if (strstr(target, "wmma"))
        return 5e-3;
    return 1e-3;
}

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --mode <kernel> [options]       # timing\n"
        "  %s --mode correctness --target <k>   # RMSE vs naive reference\n"
        "Kernels: naive | naive_causal | flash | flash_causal | flash_v2 |\n"
        "         flash_wmma | flash_wmma_db\n"
        "Options:\n"
        "  --seq_len  N    Sequence length       (default: 1024)\n"
        "  --d_head   D    Head dimension        (default: 64, fixed in kernel)\n"
        "  --warmup   W    Warmup iterations     (default: 3)\n"
        "  --iters    I    Timing iterations     (default: 10)\n"
        "  --csv           Print CSV line (mode,N,d,ms)\n"
        "  --target   K    correctness reference pair (default: flash)\n",
        prog, prog);
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    const char* mode    = "flash";
    const char* target  = "flash";
    int  N      = 1024;
    int  d      = D_HEAD;
    int  warmup = 3;
    int  iters  = 10;
    bool csv    = false;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--mode")    && i+1 < argc) mode   = argv[++i];
        if (!strcmp(argv[i], "--target")  && i+1 < argc) target = argv[++i];
        if (!strcmp(argv[i], "--seq_len") && i+1 < argc) N      = atoi(argv[++i]);
        if (!strcmp(argv[i], "--d_head")  && i+1 < argc) d      = atoi(argv[++i]);
        if (!strcmp(argv[i], "--warmup")  && i+1 < argc) warmup = atoi(argv[++i]);
        if (!strcmp(argv[i], "--iters")   && i+1 < argc) iters  = atoi(argv[++i]);
        if (!strcmp(argv[i], "--csv"))                   csv    = true;
        if (!strcmp(argv[i], "--help"))  { usage(argv[0]); return 0; }
    }

    const int qkv_n = N * d;
    float* hQ = new float[qkv_n];
    float* hK = new float[qkv_n];
    float* hV = new float[qkv_n];
    float* hO = new float[qkv_n];

    fill_random(hQ, qkv_n, 42);
    fill_random(hK, qkv_n, 43);
    fill_random(hV, qkv_n, 44);

    float *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ, hQ, qkv_n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK, qkv_n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV, qkv_n*sizeof(float), cudaMemcpyHostToDevice));

    if (!strcmp(mode, "correctness")) {
        float *dO_ref, *dO_tst, *dS;
        CUDA_CHECK(cudaMalloc(&dO_ref, qkv_n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dO_tst, qkv_n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dS,   (long)N * N * sizeof(float)));

        const bool causal = is_causal_target(target);
        if (causal) {
            naive_attention_causal(dQ, dK, dV, dO_ref, dS, N, d);
            if (!strcmp(target, "flash_causal"))
                flash_attention_v1_causal(dQ, dK, dV, dO_tst, N);
            else {
                fprintf(stderr, "correctness --target must be flash_causal for causal RMSE\n");
                return 1;
            }
        } else {
            naive_attention(dQ, dK, dV, dO_ref, dS, N, d);
            run_target(target, dQ, dK, dV, dO_tst, N, d);
        }

        float* h_ref = new float[qkv_n];
        float* h_tst = new float[qkv_n];
        CUDA_CHECK(cudaMemcpy(h_ref, dO_ref, qkv_n*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_tst, dO_tst, qkv_n*sizeof(float), cudaMemcpyDeviceToHost));

        double err = rmse(h_ref, h_tst, qkv_n);
        double tol = rmse_threshold_for(target);
        printf("target=%s  N=%d  d=%d  RMSE(ref,tgt) = %.6e  (tol %.1e)  %s\n",
               target, N, d, err, tol, (err < tol) ? "PASS" : "FAIL");

        cudaFree(dO_ref); cudaFree(dO_tst); cudaFree(dS);
        delete[] h_ref; delete[] h_tst;
        goto cleanup;
    }

    {
        float* dS = nullptr;
        if (!strcmp(mode, "naive") || !strcmp(mode, "naive_causal"))
            CUDA_CHECK(cudaMalloc(&dS, (long)N * N * sizeof(float)));

        auto run_once = [&]() {
            if (!strcmp(mode, "naive"))
                naive_attention(dQ, dK, dV, dO, dS, N, d);
            else if (!strcmp(mode, "naive_causal"))
                naive_attention_causal(dQ, dK, dV, dO, dS, N, d);
            else
                run_target(mode, dQ, dK, dV, dO, N, d);
        };

        for (int w = 0; w < warmup; ++w) run_once();

        GpuTimer timer;
        timer.Start();
        for (int it = 0; it < iters; ++it) run_once();
        float ms = timer.Stop() / iters;

        if (csv)
            printf("%s,%d,%d,%.4f\n", mode, N, d, ms);
        else
            printf("mode=%-14s  N=%5d  d=%2d  time=%.4f ms\n", mode, N, d, ms);

        if (dS) cudaFree(dS);
    }

cleanup:
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    delete[] hQ; delete[] hK; delete[] hV; delete[] hO;
    return 0;
}
