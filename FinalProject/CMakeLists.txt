cmake_minimum_required(VERSION 3.18)

# Default matches Tesla T4 / Turing (sm_75). Override when targeting another GPU, e.g.
#   cmake -DFLASHATTN_CUDA_ARCH=80 ..
set(FLASHATTN_CUDA_ARCH "75" CACHE STRING "CUDA real arch for nvcc (e.g. 75, 80, 89)")
set(CMAKE_CUDA_ARCHITECTURES "${FLASHATTN_CUDA_ARCH}")

project(FlashAttentionME759 CUDA CXX)

set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(flash_attn
    src/main.cu
    src/naive_attention.cu
    src/flash_attention_v1.cu
    src/flash_attention_v1_causal.cu
    src/flash_attention_v2.cu
    src/flash_attention_v3_wmma.cu
)

target_include_directories(flash_attn PRIVATE src)

target_compile_options(flash_attn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --extended-lambda
        --expt-relaxed-constexpr
        -lineinfo
        --ptxas-options=-v
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -Wall
    >
)
