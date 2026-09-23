#include <stdio.h>
#include <cuda_runtime.h>
#include "GPUTimer.h"
#define REPEAT 20

//同一个拷贝测试，对比可分页内存和页锁定内存，pinned 省掉一次中转拷贝
int main(){
    const int n = 1 << 24;
    const size_t bytes = (size_t)n * sizeof(float);
    float *d_buf;
    cudaMalloc((void **)&d_buf, bytes);

    float *h_page = new float[n];
    float *h_pin;
    cudaMallocHost((void **)&h_pin, bytes);
    for (int i = 0; i < n; i++) h_page[i] = 1.0f;
    for (int i = 0; i < n; i++) h_pin[i] = 1.0f;

    //先空跑一次预热
    cudaMemcpy(d_buf, h_page, bytes, cudaMemcpyHostToDevice);

    GPUTimer timer;
    timer.Start();
    for (int r = 0; r < REPEAT; r++){
        cudaMemcpy(d_buf, h_page, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(h_page, d_buf, bytes, cudaMemcpyDeviceToHost);
    }
    timer.End();
    float tPage = timer.Elapsed();

    timer.Start();
    for (int r = 0; r < REPEAT; r++){
        cudaMemcpy(d_buf, h_pin, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(h_pin, d_buf, bytes, cudaMemcpyDeviceToHost);
    }
    timer.End();
    float tPin = timer.Elapsed();

    double totalGB = 2.0 * REPEAT * bytes / 1e9;
    printf("pageable memory consumes %f ms, bandwidth %f GB/s\n", tPage, totalGB / (tPage / 1000));
    printf("pinned memory    consumes %f ms, bandwidth %f GB/s\n", tPin, totalGB / (tPin / 1000));

    cudaFreeHost(h_pin);
    cudaFree(d_buf);
    delete[] h_page;
    return 0;
}
