#include "CPURandom.h"
#include <chrono>
#include <stdio.h>
#include <cstdlib>
#include "GPURandom.h"
#include <cuda_runtime.h>
#include "GPUTimer.h"
#define BLOCK 256

__global__ void reduce_interleave(float *d_in, const int n){
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    
    extern __shared__ float cache[];
    if (gid < n)
        cache[tid] = d_in[gid];
    else
        cache[tid] = 0.0f;
    __syncthreads();

    for (int i = 1; i < blockDim.x; i *=2){
        if (tid % (i * 2) == 0) cache[tid] += cache[tid + i];
        __syncthreads();
    }
}

__global__ void reduce_contiguous(float *d_in, const int n){
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    extern __shared__ float cache[];
    if (gid < n)
        cache[tid] = d_in[gid];
    else
        cache[tid] = 0.0f;
    __syncthreads();
    for (int i = blockDim.x / 2; i != 0; i >>= 1){
        if (tid < i) cache[tid] += cache[tid + i];
        __syncthreads();
    }

}

int main() {
    const size_t n = 1 << 26;
    float *d_in, *h_in;
    cudaMalloc((void **)&d_in, (size_t)n * sizeof(float));



    int grid = (n + BLOCK - 1) / BLOCK;

    GPUTimer timer;
    timer.Start();
    GPUrandF<<<grid, BLOCK>>>((unsigned long long)666, n, d_in);
    cudaDeviceSynchronize();
    timer.End();
    printf("GPUrandom consums %f ms\n", timer.Elapsed());

    float *d_conti, *d_inter;
    cudaMalloc((void **)&d_conti, n * sizeof(float));
    cudaMalloc((void **)&d_inter, n * sizeof(float));
    cudaMemcpy(d_conti, d_in, n * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_inter, d_in, n * sizeof(float), cudaMemcpyDeviceToDevice);

    h_in = (float *)malloc((size_t)n * sizeof(float));
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) h_in[i] = dist_float(gen);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (t2 - t1);
    printf("CPU random fill consumes %lld ms\n", duration.count());

    GPUTimer timerConti;
    timerConti.Start();
    reduce_interleave<<<grid, BLOCK, BLOCK * sizeof(float)>>>(d_inter, n);
    cudaDeviceSynchronize();
    timerConti.End();
    printf("interleave reduce consums %f ms\n", timerConti.Elapsed());
    
    GPUTimer timerInter;
    timerInter.Start();
    reduce_contiguous<<<grid, BLOCK, BLOCK * sizeof(float)>>>(d_conti, n);
    cudaDeviceSynchronize();
    timerInter.End();
    printf("contiguous reduce consums %f ms\n", timerInter.Elapsed());

    cudaFree(d_in);
    free(h_in);
    cudaFree(d_conti);
    cudaFree(d_inter);

    return 0;
}