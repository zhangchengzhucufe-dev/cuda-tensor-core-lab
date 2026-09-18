// Problem: Query your GPU device 

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU model           : %s\n", prop.name);
    printf("compute capability  : %d.%d\n", prop.major, prop.minor);
    // ====== Blank 1: Number of SMs (hint: field name starts with multiProcessor) ======
    printf("SM count            : %d\n", prop.multiProcessorCount);
    // ====== Blank 2: warp size ======
    printf("warp size           : %d\n", prop.warpSize);
    // ====== Blank 3: Maximum shared memory available per block (bytes) ======
    printf("shared mem / block  : %zu\n", (size_t)prop.sharedMemPerBlock);
    // ====== Blank 4: Maximum resident threads per SM ======
    printf("max threads / SM    : %d\n", prop.maxThreadsPerMultiProcessor);
    // ====== Blank 5: Total global memory (bytes) ======
    printf("global mem          : %zu\n", (size_t)prop.totalGlobalMem);
    printf("max threads / block : %d\n", prop.maxThreadsPerBlock);
    return 0;
}