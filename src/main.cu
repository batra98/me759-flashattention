#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include "attention_utils.cuh"

// ── Forward declarations ───────────────────────────────────────────────────
void naive_attention(const float* dQ, const float* dK, const float* dV,
                     float* dO, float* dS, int N, int d);
void flash_attention_v1(const float* dQ, const float* dK, const float* dV,
                         float* dO, int N);

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --mode [naive|flash|correctness] [options]\n"
        "Options:\n"
        "  --seq_len  N    Sequence length       (default: 1024)\n"
        "  --d_head   D    Head dimension        (default: 64, fixed in kernel)\n"
        "  --warmup   W    Warmup iterations     (default: 3)\n"
        "  --iters    I    Timing iterations     (default: 10)\n"
        "  --csv           Print CSV line (mode,N,d,ms)\n",
        prog);
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    // ── Parse arguments ───────────────────────────────────────────────────
    const char* mode = "flash";
    int  N      = 1024;
    int  d      = D_HEAD;
    int  warmup = 3;
    int  iters  = 10;
    bool csv    = false;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--mode")    && i+1 < argc) mode   = argv[++i];
        if (!strcmp(argv[i], "--seq_len") && i+1 < argc) N      = atoi(argv[++i]);
        if (!strcmp(argv[i], "--d_head")  && i+1 < argc) d      = atoi(argv[++i]);
        if (!strcmp(argv[i], "--warmup")  && i+1 < argc) warmup = atoi(argv[++i]);
        if (!strcmp(argv[i], "--iters")   && i+1 < argc) iters  = atoi(argv[++i]);
        if (!strcmp(argv[i], "--csv"))                   csv    = true;
        if (!strcmp(argv[i], "--help"))  { usage(argv[0]); return 0; }
    }

    // ── Allocate and initialise host buffers ──────────────────────────────
    const int qkv_n = N * d;
    float* hQ = new float[qkv_n];
    float* hK = new float[qkv_n];
    float* hV = new float[qkv_n];
    float* hO = new float[qkv_n];

    fill_random(hQ, qkv_n, 42);
    fill_random(hK, qkv_n, 43);
    fill_random(hV, qkv_n, 44);

    // ── Allocate device buffers ───────────────────────────────────────────
    float *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ, hQ, qkv_n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK, qkv_n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV, qkv_n*sizeof(float), cudaMemcpyHostToDevice));

    // ═══════════════════════════════════════════════════════════════════════
    //  CORRECTNESS MODE: run both kernels, compare outputs
    // ═══════════════════════════════════════════════════════════════════════
    if (!strcmp(mode, "correctness")) {
        float *dO_n, *dO_f, *dS;
        CUDA_CHECK(cudaMalloc(&dO_n, qkv_n    * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dO_f, qkv_n    * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dS,   (long)N*N * sizeof(float)));

        naive_attention      (dQ, dK, dV, dO_n, dS,  N, d);
        flash_attention_v1   (dQ, dK, dV, dO_f,      N);

        float* hO_n = new float[qkv_n];
        float* hO_f = new float[qkv_n];
        CUDA_CHECK(cudaMemcpy(hO_n, dO_n, qkv_n*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hO_f, dO_f, qkv_n*sizeof(float), cudaMemcpyDeviceToHost));

        double err = rmse(hO_n, hO_f, qkv_n);
        printf("N=%d  d=%d  RMSE(naive, flash) = %.6e  →  %s\n",
               N, d, err, (err < 1e-3) ? "PASS ✓" : "FAIL ✗");

        cudaFree(dO_n); cudaFree(dO_f); cudaFree(dS);
        delete[] hO_n; delete[] hO_f;
        goto cleanup;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  TIMING MODE: warmup + timed iterations
    // ═══════════════════════════════════════════════════════════════════════
    {
        float* dS = nullptr;
        if (!strcmp(mode, "naive"))
            CUDA_CHECK(cudaMalloc(&dS, (long)N*N * sizeof(float)));

        auto run_once = [&]() {
            if (!strcmp(mode, "naive")) naive_attention(dQ, dK, dV, dO, dS, N, d);
            else                       flash_attention_v1(dQ, dK, dV, dO, N);
        };

        // Warmup
        for (int w = 0; w < warmup; ++w) run_once();

        // Timed iterations
        GpuTimer timer;
        timer.Start();
        for (int it = 0; it < iters; ++it) run_once();
        float ms = timer.Stop() / iters;

        if (csv)
            printf("%s,%d,%d,%.4f\n", mode, N, d, ms);
        else
            printf("mode=%-5s  N=%5d  d=%2d  time=%.4f ms\n", mode, N, d, ms);

        if (dS) cudaFree(dS);
    }

cleanup:
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    delete[] hQ; delete[] hK; delete[] hV; delete[] hO;
    return 0;
}
