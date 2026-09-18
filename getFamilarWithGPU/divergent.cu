#include <stdio.h>
#include <chrono>
#include <cstddef>
#include "CPURandom.h"
#include <cuda_runtime.h>

__global__ void diverge_in_wrap(float *arr, float *arr1, float *arr2){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid % 2 == 0){
        arr[tid] = arr1[tid] * arr2[tid] - arr1[tid];
    }else{
        arr[tid] = arr1[tid] / arr2[tid] + arr2[tid];
    }
}


__global__ void diverge_by_wrap(float *arr, float *arr1, float *arr2){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if ((tid / 32 ) % 2){
        arr[tid] = arr1[tid] * arr2[tid] - arr1[tid];
    }else {
        arr[tid] = arr1[tid] / arr2[tid] - arr2[tid];      
    }
}

int main(){
    const int n  = 1 << 26;
    size_t bytes = n * sizeof(float);
    float *arr1 = (float *)malloc(bytes);
    float *arr2 = (float *)malloc(bytes);
    float *d_arr;
    cudaMalloc((void **)&d_arr, bytes);

    for (int i = 0; i < n; i++) arr1[i] = dist_float(gen);
    for (int i = 0; i < n; i++) arr2[i] = dist_float(gen);

    float *d_arr1, *d_arr2;
    cudaMalloc((void **)&d_arr1, bytes);
    cudaMalloc((void **)&d_arr2, bytes);

    cudaMemcpy(d_arr1, arr1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, arr2, bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(256);
    dim3 gridDim((n + 255) / 256);

    auto t1 = std::chrono::high_resolution_clock::now();
    diverge_by_wrap<<<gridDim, blockDim>>>(d_arr, d_arr1, d_arr2);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto durationByWrap = std::chrono::duration_cast<std::chrono::milliseconds> (t2 - t1);

    auto t3 = std::chrono::high_resolution_clock::now();
    diverge_in_wrap<<<gridDim, blockDim>>>(d_arr, d_arr1, d_arr2);
    cudaDeviceSynchronize();
    auto t4 = std::chrono::high_resolution_clock::now();
    auto durationInWrap = std::chrono::duration_cast<std::chrono::milliseconds> (t4 - t3);

    printf("durationInWrap %ld ms\n", durationInWrap.count());
    printf("durationByWrap %ld ms\n", durationByWrap.count());

    cudaFree(d_arr1);
    cudaFree(d_arr2);
    free(arr1);
    free(arr2);
    cudaFree(d_arr);

    return 0;
}


