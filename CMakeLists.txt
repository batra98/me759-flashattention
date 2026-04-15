cmake_minimum_required(VERSION 3.18)
project(FlashAttentionME759 CUDA CXX)

set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)

# Target: NVIDIA RTX 2080 Ti — Turing, sm_75
set(CMAKE_CUDA_ARCHITECTURES 75)

add_executable(flash_attn
    src/main.cu
    src/naive_attention.cu
    src/flash_attention_v1.cu
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
