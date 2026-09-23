#include "GPURandom.h"
#include <stdio.h>
#include "GPUTimer.h"
#include <cuda_runtime.h>

__global__ void stride_copy(int stride, const int n, int *d_arr, int *d_out){
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    long idx = (long)gid * stride;
    int sum = 0;
    volatile int *p = d_arr;
    #pragma unroll
    for(int i = 0; i < 2048; i++){
        sum += p[idx];
    }
    d_out[gid] = sum;
}

int main(){
    int *d_arr;
    const int total_threads = 128 * 256;
    const int max_stride = 16;
    const int n = total_threads * max_stride;
    int bytes = n * sizeof(int);
    cudaMalloc((void **)&d_arr, bytes);
    GpuRandI<<<120, 256>>>(1ULL, n, d_arr);
    
    int stride[5] = {1, 2, 4, 8, 16};
    int *d_out;
    cudaMalloc((void **)&d_out, total_threads * sizeof(int));

    //预热kernel，消除JIT开销
    stride_copy<<<128, 256>>>(1, n, d_arr, d_out);
    cudaDeviceSynchronize();

    const int repeat = 10;
    for (int s : stride){
        double sum_time = 0.0;
        for(int r = 0; r < repeat; r++){
            GPUTimer timer;
            timer.Start();
            stride_copy<<<128, 256>>>(s, n, d_arr, d_out);
            cudaDeviceSynchronize();
            timer.End();
            sum_time += timer.Elapsed();
        }
        double avg = sum_time / repeat;
        printf("stride = %d avg consumes %f ms\n", s, avg);
    }
    cudaFree(d_arr);
    cudaFree(d_out);
    return 0;
}
