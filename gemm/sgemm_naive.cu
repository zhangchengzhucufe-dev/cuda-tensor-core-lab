#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "CPURandom.h"
#include "GPUTimer.h"
#define N 2048
#define REPEAT 10

//最朴素的矩阵乘法，一个线程负责 C 的一个元素，C = A x B
__global__ void sgemm_naive(const float *A, const float *B, float *C, const int n){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n){
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
            sum += A[row * n + k] * B[k * n + col];   //A 按行读，B 按列读，B 的访存不合并
        C[row * n + col] = sum;
    }
}

int main(){
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_ref = new float[N * N];
    for (int i = 0; i < N * N; i++) h_A[i] = dist_float(gen);
    for (int i = 0; i < N * N; i++) h_B[i] = dist_float(gen);

    //CPU 参考答案，i,k,j 的顺序对缓存友好
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++) h_ref[i * N + j] = 0.0f;
        for (int k = 0; k < N; k++){
            float a = h_A[i * N + k];
            for (int j = 0; j < N; j++) h_ref[i * N + j] += a * h_B[k * N + j];
        }
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * N * sizeof(float));
    cudaMalloc((void **)&d_C, N * N * sizeof(float));
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid(N / 32, N / 32);

    //先空跑一次预热，消除 JIT 开销
    sgemm_naive<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    double sumTime = 0.0;
    for (int r = 0; r < REPEAT; r++){
        GPUTimer timer;
        timer.Start();
        sgemm_naive<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        timer.End();
        sumTime += timer.Elapsed();
    }
    float avg = sumTime / REPEAT;
    printf("naive sgemm %dx%dx%d avg consumes %f ms, %f GFLOPS\n", N, N, N, avg, 2.0 * N * N * N / (avg * 1e6));

    float *h_C = new float[N * N];
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    float maxErr = 0.0f;
    for (int i = 0; i < N * N; i++){
        float err = fabsf(h_C[i] - h_ref[i]);
        if (err > maxErr) maxErr = err;
    }
    //CPU 和 GPU 的求和顺序不一样，误差不是 0，小量级就算对
    printf("max error vs cpu = %f\n", maxErr);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_ref;
    return 0;
}
