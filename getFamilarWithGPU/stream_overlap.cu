#include <stdio.h>
#include <cuda_runtime.h>
#include "GPUTimer.h"

//故意写重一点，计算才有得和拷贝重叠，不然瓶颈全在 PCIe 上
__global__ void heavy_add(float *arr, const int n){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < n){
        float x = arr[tid];
        for (int i = 0; i < 3000; i++) x = fmaf(x, 1.0000001f, 1e-7f);
        arr[tid] = x;
        tid += blockDim.x * gridDim.x;
    }
}

//大块的拷贝-计算-拷回，拆成 4 块放 4 条流里，拷贝和计算能互相重叠
//注意：WSL2 的半虚拟化驱动会把不同流的 kernel 串行执行（实测两个 10ms 的核在两条流上要跑 20ms），
//所以本机看不出 4 流的收益；原生 Linux/Windows 驱动上这版会明显更快
int main(){
    const int n = 1 << 24;
    const size_t bytes = (size_t)n * sizeof(float);
    const int chunks = 4;
    const int per = n / chunks;

    float *h_in, *d_in;
    cudaMallocHost((void **)&h_in, bytes);   //异步拷贝必须用页锁定内存
    cudaMalloc((void **)&d_in, bytes);
    for (int i = 0; i < n; i++) h_in[i] = 1.0f;

    heavy_add<<<120, 256>>>(d_in, per);   //先空跑一次预热，消除 JIT 开销
    cudaDeviceSynchronize();

    GPUTimer timer;
    timer.Start();
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    heavy_add<<<120, 256>>>(d_in, n);
    cudaMemcpy(h_in, d_in, bytes, cudaMemcpyDeviceToHost);
    timer.End();
    printf("single stream consumes %f ms\n", timer.Elapsed());

    cudaStream_t st[chunks];
    for (int i = 0; i < chunks; i++) cudaStreamCreate(&st[i]);
    timer.Start();
    for (int i = 0; i < chunks; i++){
        int offset = i * per;
        cudaMemcpyAsync(d_in + offset, h_in + offset, per * sizeof(float), cudaMemcpyHostToDevice, st[i]);
        heavy_add<<<120, 256, 0, st[i]>>>(d_in + offset, per);
        cudaMemcpyAsync(h_in + offset, d_in + offset, per * sizeof(float), cudaMemcpyDeviceToHost, st[i]);
    }
    timer.End();   //默认流上的 event 会等所有阻塞流跑完
    printf("4 streams overlap consumes %f ms\n", timer.Elapsed());

    for (int i = 0; i < chunks; i++) cudaStreamDestroy(st[i]);
    cudaFreeHost(h_in);
    cudaFree(d_in);
    return 0;
}
