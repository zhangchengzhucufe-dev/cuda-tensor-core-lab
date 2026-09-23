#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "CPURandom.h"
#include "GPUTimer.h"
#define N 2048
#define TILE 32
#define REPEAT 10

//A、B 各分成 32x32 的块搬进 shared memory 再乘加，N 是 TILE 的倍数所以不判边界
__global__ void sgemm_tiled(const float *A, const float *B, float *C, const int n){
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    float sum = 0.0f;

    for (int t = 0; t < n / TILE; t++){
        As[threadIdx.y][threadIdx.x] = A[row * n + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * n + col];
        __syncthreads();
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();   //防止下一轮把还没用完的块覆盖掉
    }
    C[row * n + col] = sum;
}

int main(){
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_ref = new float[N * N];
    for (int i = 0; i < N * N; i++) h_A[i] = dist_float(gen);
    for (int i = 0; i < N * N; i++) h_B[i] = dist_float(gen);

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

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    //先空跑一次预热，消除 JIT 开销
    sgemm_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    double sumTime = 0.0;
    for (int r = 0; r < REPEAT; r++){
        GPUTimer timer;
        timer.Start();
        sgemm_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        timer.End();
        sumTime += timer.Elapsed();
    }
    float avg = sumTime / REPEAT;
    printf("tiled sgemm %dx%dx%d avg consumes %f ms, %f GFLOPS\n", N, N, N, avg, 2.0 * N * N * N / (avg * 1e6));

    float *h_C = new float[N * N];
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    float maxErr = 0.0f;
    for (int i = 0; i < N * N; i++){
        float err = fabsf(h_C[i] - h_ref[i]);
        if (err > maxErr) maxErr = err;
    }
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
