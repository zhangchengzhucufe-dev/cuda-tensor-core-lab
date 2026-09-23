//编译: nvcc -O2 -arch=sm_86 -I random gemm/sgemm_wmma.cu -o gemm/sgemm_wmma
//wmma 要求 sm_70 以上，3060 是 sm_86
#include <stdio.h>
#include <math.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include "CPURandom.h"
#include "GPUTimer.h"

using namespace nvcuda;
#define N 2048
#define REPEAT 10

//tensor core 最小用法：一个 block 一个 warp，负责 C 里一个 16x16 的块
//输入是 fp16、累加是 fp32，这正是 tensor core 的标准姿势
__global__ void sgemm_wmma(const half *A, const half *B, float *C, const int n){
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag;
    wmma::fill_fragment(cFrag, 0.0f);

    int row = blockIdx.y * 16;
    int col = blockIdx.x * 16;
    for (int k = 0; k < n; k += 16){
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aFrag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bFrag;
        wmma::load_matrix_sync(aFrag, A + row * n + k, n);
        wmma::load_matrix_sync(bFrag, B + k * n + col, n);
        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
    }
    wmma::store_matrix_sync(C + row * n + col, cFrag, n, wmma::mem_row_major);
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

    //host 上把 float 转成 half 再传上去
    half *h_Ah = new half[N * N];
    half *h_Bh = new half[N * N];
    for (int i = 0; i < N * N; i++) h_Ah[i] = __float2half(h_A[i]);
    for (int i = 0; i < N * N; i++) h_Bh[i] = __float2half(h_B[i]);

    half *d_A, *d_B;
    float *d_C;
    cudaMalloc((void **)&d_A, N * N * sizeof(half));
    cudaMalloc((void **)&d_B, N * N * sizeof(half));
    cudaMalloc((void **)&d_C, N * N * sizeof(float));
    cudaMemcpy(d_A, h_Ah, N * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_Bh, N * N * sizeof(half), cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid(N / 16, N / 16);

    double sumTime = 0.0;
    for (int r = 0; r < REPEAT; r++){
        GPUTimer timer;
        timer.Start();
        sgemm_wmma<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        timer.End();
        sumTime += timer.Elapsed();
    }
    float avg = sumTime / REPEAT;
    printf("wmma sgemm %dx%dx%d avg consumes %f ms, %f TFLOPS\n", N, N, N, avg, 2.0 * N * N * N / (avg * 1e9));

    float *h_C = new float[N * N];
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    float maxErr = 0.0f;
    for (int i = 0; i < N * N; i++){
        float err = fabsf(h_C[i] - h_ref[i]);
        if (err > maxErr) maxErr = err;
    }
    //输入被砍成 fp16，误差比前两版大是正常的
    printf("max error vs cpu = %f\n", maxErr);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_Ah;
    delete[] h_Bh;
    delete[] h_C;
    delete[] h_ref;
    return 0;
}
