#include <stdio.h>
#include "GPUTimer.h"
#include "GPURandom.h"
#include <cstdlib>
#include <cuda_runtime.h>
#define BINS 16

__global__ void histogram_naive(const int n, int *d_out, int *d_in){
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; gid < n; gid += stride)
        atomicAdd(&d_out[d_in[gid] % BINS], 1);
}


__global__ void histogram_pri(const int n, int *d_in, int *d_out){
    __shared__ int s_out[BINS];
    int tid = threadIdx.x;
    if (tid < BINS) s_out[tid] = 0;
    __syncthreads();

    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; gid < n; gid += stride)
        atomicAdd(&s_out[d_in[gid] % BINS], 1);
    __syncthreads();

    if (tid < BINS) atomicAdd(&d_out[tid], s_out[tid]);

}


int main(){
    const int n = 1 << 27;
    int *d_in;
    size_t bytes = n * sizeof(int);
    cudaMalloc((void **)&d_in, bytes);
    GpuRandI<<<120, 256>>>(1ULL, n, d_in);

    int *d_out1;
    cudaMalloc((void **)&d_out1, BINS * sizeof(int));
    int *d_out2;
    cudaMalloc((void **)&d_out2, BINS * sizeof(int));

    GPUTimer timerNai;
    timerNai.Start();
    histogram_naive<<<120, 256>>>(n, d_out1, d_in);
    timerNai.End();
    printf("naive consumes %f ms\n", timerNai.Elapsed());

    GPUTimer timerPri;
    timerPri.Start();
    histogram_pri<<<120, 256>>>(n, d_in, d_out2);
    timerPri.End();
    printf("private consumes %f ms\n", timerPri.Elapsed());

    cudaFree(d_out1);
    cudaFree(d_out2);
    cudaFree(d_in);

}