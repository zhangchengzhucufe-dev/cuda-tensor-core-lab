#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "CPURandom.h"
#include "GPUTimer.h"
#define N 4096
#define TILE 32
#define REPEAT 20

//朴素转置：读 A 连续，写 B 跨步 n，写的访存不合并
__global__ void transpose_naive(const float *A, float *B, const int n){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    B[x * n + y] = A[y * n + x];
}

//先搬进 shared memory，再按转置后的位置写，读写都合并；+1 错开 bank
__global__ void transpose_tiled(const float *A, float *B, const int n){
    __shared__ float tile[TILE][TILE + 1];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = A[y * n + x];
    __syncthreads();
    int x2 = blockIdx.y * TILE + threadIdx.x;   //行列块互换后再写
    int y2 = blockIdx.x * TILE + threadIdx.y;
    B[y2 * n + x2] = tile[threadIdx.x][threadIdx.y];
}

int main(){
    const int bytes = N * N * sizeof(float);
    float *d_A, *d_B1, *d_B2;
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B1, bytes);
    cudaMalloc((void **)&d_B2, bytes);

    float *h_A = new float[N * N];
    for (int i = 0; i < N * N; i++) h_A[i] = dist_float(gen);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    //先空跑一次预热
    transpose_naive<<<grid, block>>>(d_A, d_B1, N);
    cudaDeviceSynchronize();

    double sumTime = 0.0;
    for (int r = 0; r < REPEAT; r++){
        GPUTimer timer;
        timer.Start();
        transpose_naive<<<grid, block>>>(d_A, d_B1, N);
        cudaDeviceSynchronize();
        timer.End();
        sumTime += timer.Elapsed();
    }
    float avgNai = sumTime / REPEAT;
    printf("naive  transpose consumes %f ms, bandwidth %f GB/s\n", avgNai, 2.0 * bytes / (avgNai * 1e6));

    sumTime = 0.0;
    for (int r = 0; r < REPEAT; r++){
        GPUTimer timer;
        timer.Start();
        transpose_tiled<<<grid, block>>>(d_A, d_B2, N);
        cudaDeviceSynchronize();
        timer.End();
        sumTime += timer.Elapsed();
    }
    float avgTile = sumTime / REPEAT;
    printf("tiled  transpose consumes %f ms, bandwidth %f GB/s\n", avgTile, 2.0 * bytes / (avgTile * 1e6));

    float *h_B = new float[N * N];
    cudaMemcpy(h_B, d_B2, bytes, cudaMemcpyDeviceToHost);
    float maxErr = 0.0f;
    for (int i = 0; i < N * N; i++){
        float err = fabsf(h_B[i] - h_A[(i % N) * N + i / N]);   //和 h_A 的转置对
        if (err > maxErr) maxErr = err;
    }
    printf("max error vs cpu = %f\n", maxErr);

    cudaFree(d_A);
    cudaFree(d_B1);
    cudaFree(d_B2);
    delete[] h_A;
    delete[] h_B;
    return 0;
}
