#include "GPUTimer.h"
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist_float(0.0f, 10.5f);

__global__ void one_thread(float *arrG1, const int n, float *d_arrC1, float *d_arrC2){
    for (int i = 0; i < n; i++) arrG1[i] = d_arrC1[i] + d_arrC2[i];
}

__global__ void one_block(const int n, float *arrG2, float *d_arrC1, float *d_arrC2){
    for (int i = threadIdx.x; i < n; i += blockDim.x) arrG2[i] = d_arrC1[i] + d_arrC2[i];
}

__global__ void add_grid(const int n, float *arrG3, float *d_arrC1, float *d_arrC2){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (true){
        if (tid >= n) return;
        arrG3[tid] = d_arrC1[tid] + d_arrC2[tid];
        tid += blockDim.x * gridDim.x;
    }

}

int main(){
    const int n = 1 << 22;
    float *arrC1 = new float[n]{0.0f};
    float *arrC2 = new float[n]{0.0f};
    float *arrC = new float[n]{0.0f};

    for (int i = 0; i < n; i++) arrC1[i] = dist_float(gen);
    for (int i = 0; i < n; i++) arrC2[i] = dist_float(gen);

    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < 10; j++)
        for (int i = 0; i < n; i++) arrC[i] = arrC1[i] + arrC2[i];
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);
    std::cout << "CPU " << duration.count() / 10 << "ms" <<std::endl;

    const size_t bytes = n * sizeof(float);    
    float *d_arrC1, *d_arrC2;
    cudaMalloc(&d_arrC1, bytes);
    cudaMalloc(&d_arrC2, bytes);

    cudaMemcpy(d_arrC1, arrC1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arrC2, arrC2, bytes, cudaMemcpyHostToDevice);
    
    float *arrG1, *arrG2, *arrG3;
    cudaMalloc(&arrG1, bytes);
    cudaMalloc(&arrG2, bytes);
    cudaMalloc(&arrG3, bytes);

    add_grid<<<128, 256>>>(n, arrG3, d_arrC1, d_arrC2);

    GPUTimer timerOneThread;
    timerOneThread.Start();
    for (int i = 0; i < 10; i++) one_thread<<<1, 1>>>(arrG1, n, d_arrC1, d_arrC2);
    timerOneThread.End();
    std::cout <<  "oneThread " <<timerOneThread.Elapsed() / 10 <<"ms" << std::endl;

    GPUTimer timerOneBlock;
    timerOneBlock.Start();
    for (int i = 0; i < 10; i++) one_block<<<1, 256>>>(n, arrG2, d_arrC1, d_arrC2);
    timerOneBlock.End();
    std::cout <<  "oneBlock " <<timerOneBlock.Elapsed() / 10 << "ms" <<std::endl;
    
    GPUTimer timerGrid;
    timerGrid.Start();
    for (int i = 0; i < 10; i++) add_grid<<<128, 256>>>(n, arrG3, d_arrC1, d_arrC2);
    timerGrid.End();
    std::cout << "add_grid " <<timerGrid.Elapsed() / 10 << "ms" <<std::endl;

    delete[] arrC1;
    delete[] arrC2;
    delete[] arrC;

    cudaFree(d_arrC1);
    cudaFree(d_arrC2);
    cudaFree(arrG1);
    cudaFree(arrG2);
    cudaFree(arrG3);
}

