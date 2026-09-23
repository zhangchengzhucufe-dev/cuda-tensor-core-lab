#include <stdio.h>
#include <cuda_runtime.h>
#include "GPURandom.h"
#include "GPUTimer.h"
#define BLOCK 256

//warp 内用 shuffle 寄存器归约，warp 之间再归约一次，比 shared memory 互扣快
__global__ void reduce_shuffle(const float *d_in, float *d_out, const int n){
    __shared__ float warpSum[BLOCK / 32];
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    float v = (gid < n) ? d_in[gid] : 0.0f;

    //蝴蝶式折半，5 步把一个 warp 的 32 个值聚到 0 号线程
    for (int i = 16; i > 0; i >>= 1)
        v += __shfl_down_sync(0xffffffff, v, i);

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) warpSum[warp] = v;
    __syncthreads();

    //每个 block 只有 8 个 warp，第二次归约只有前 8 个线程参加，mask 用 0xff
    if (threadIdx.x < BLOCK / 32){
        v = warpSum[threadIdx.x];
        for (int i = BLOCK / 64; i > 0; i >>= 1)
            v += __shfl_down_sync((1u << (BLOCK / 32)) - 1u, v, i);
        if (threadIdx.x == 0) atomicAdd(d_out, v);
    }
}

int main(){
    const int n = 1 << 26;
    float *d_in, *d_out;
    cudaMalloc((void **)&d_in, (size_t)n * sizeof(float));
    cudaMalloc((void **)&d_out, sizeof(float));
    cudaMemset(d_out, 0, sizeof(float));

    int grid = (n + BLOCK - 1) / BLOCK;
    GpuRandF<<<120, 256>>>(666ULL, n, d_in);

    reduce_shuffle<<<grid, BLOCK>>>(d_in, d_out, n);   //空跑一次预热，消除 JIT 开销
    cudaDeviceSynchronize();
    cudaMemset(d_out, 0, sizeof(float));

    GPUTimer timer;
    timer.Start();
    reduce_shuffle<<<grid, BLOCK>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    timer.End();
    float gpuSum;
    cudaMemcpy(&gpuSum, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("shuffle reduce consumes %f ms, gpu sum = %f\n", timer.Elapsed(), gpuSum);

    //CPU 对账，float 累加顺序不同，对不上精确相等是正常的
    float *h_in = new float[n];
    cudaMemcpy(h_in, d_in, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    double cpuSum = 0.0;
    for (int i = 0; i < n; i++) cpuSum += h_in[i];
    printf("cpu sum = %f\n", (float)cpuSum);

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    return 0;
}
