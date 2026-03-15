cmake_minimum_required(VERSION 3.18)
project(FlashAttentionME759 CUDA CXX)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_ARCHITECTURES 75)

add_executable(flash_attn
    src/main.cu
    src/naive_attention.cu
    src/flash_attention_v1.cu
)
target_include_directories(flash_attn PRIVATE src)
target_compile_options(flash_attn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extended-lambda -lineinfo>
)
