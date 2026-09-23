#ifndef GPURANDOM_H
#define GPURANDOM_H
#include <curand_kernel.h>

__global__ void GpuRandF(unsigned long long seed, const int n, float *arr){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n) return;
    curandState state;
    curand_init(seed, tid, 0, &state);
    arr[tid] = curand_uniform(&state);
}

__global__ void GpuRandI(unsigned long long seed, const int n, int *arr){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n) return;
    curandState state;
    curand_init(seed, tid, 0, &state);
    arr[tid] = (int)(curand_uniform(&state) * 10000);
}
#endif
