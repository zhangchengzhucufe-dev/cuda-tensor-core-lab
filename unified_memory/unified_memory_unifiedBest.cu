#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include "CPURandom.h"

__global__ void add(const float *arrC, float *arrG, const int n){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < n){
        arrG[tid] = arrC[tid] + arrC[tid] - arrC[tid];
        tid += blockDim.x * gridDim.x;
    }


}

int main(){
    const int n = 1 << 24;
    float *arrC = new float[n];
    float *arrCC = new float[n];
    for (int i = 0; i < n; i++) arrC[i] = dist_float(gen);
    for (int i = 0; i < n; i++) arrCC[i] = arrC[i];

    float *d_arrC, *arrGpuMan;
    cudaMalloc((void **)&d_arrC, n * sizeof(float));
    cudaMalloc((void **)&arrGpuMan, n * sizeof(float));

    float dummyMan;
    auto t1 = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 20; i++){
        cudaMemcpy(d_arrC, arrCC, n * sizeof(float), cudaMemcpyHostToDevice);
        add<<<120, 256>>>(d_arrC, arrGpuMan, n);
        cudaDeviceSynchronize();
        cudaMemcpy(arrCC, d_arrC, sizeof(float) * n, cudaMemcpyDeviceToHost);
        dummyMan = arrCC[1];
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (t2 - t1);
    std::cout << "manual management " <<duration.count() << "ms" << std::endl;



    float *arrGpuUni, *arrGpuUniRes;
    cudaMallocManaged((void **)&arrGpuUni, sizeof(float) * n);
    cudaMallocManaged((void **)&arrGpuUniRes, sizeof(float) * n);
    for (int i = 0; i < n; i++)  arrGpuUni[i] = arrC[i];

    float dummyUni;
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 20; i++){
        add<<<120, 256>>>(arrGpuUni, arrGpuUniRes, n);
        cudaDeviceSynchronize();
        dummyUni = arrGpuUni[1];
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    auto durationUni = std::chrono::duration_cast<std::chrono::milliseconds> (t4 - t3);
    std::cout << "cudaMallUnified " << durationUni.count() << "ms" << std::endl;
    




    float *arrGpuUniMan, *arrGpuUniManRes;
    float dummyUniMan;
    cudaMallocManaged((void **)&arrGpuUniMan, sizeof(float) * n);
    cudaMallocManaged((void **)&arrGpuUniManRes, sizeof(float) * n);

    cudaMemPrefetchAsync(arrGpuUniManRes, sizeof(float) * n, 0);
    cudaDeviceSynchronize();
    for (int i = 0; i < n; i++) arrGpuUniMan[i] = arrC[i];

    auto t5 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 20; i++){
        cudaMemPrefetchAsync(arrGpuUniMan, sizeof(float) * n, 0);
        cudaDeviceSynchronize();


        add<<<120, 256>>>(arrGpuUniMan, arrGpuUniManRes, n);
        cudaMemPrefetchAsync(arrGpuUniMan, sizeof(float) * n, cudaCpuDeviceId);
        cudaDeviceSynchronize();
        dummyUniMan = arrGpuUniMan[1];
    }

    auto t6 = std::chrono::high_resolution_clock::now();
    auto durationUniMan = std::chrono::duration_cast<std::chrono::milliseconds> (t6 - t5);
    std::cout << "memUniMan " << durationUniMan.count() << "ms" << std::endl;


    delete[] arrC;
    cudaFree(d_arrC);
    cudaFree(arrGpuMan);
    cudaFree(arrGpuUni);
    cudaFree(arrGpuUniRes);
    cudaFree(arrGpuUniMan);
    cudaFree(arrGpuUniManRes);
    delete[] arrCC;

}

