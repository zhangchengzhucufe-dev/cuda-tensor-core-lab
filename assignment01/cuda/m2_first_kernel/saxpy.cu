// 问题 2.9（压轴，FROM-SCRATCH）：SAXPY
// y = 2.0 * x + y，单精度。用法：./saxpy <n>
// 数据按固定公式生成：
//   x[i] = ((i % 2048) - 1024) * 0.5f
//   y[i] = (i % 1024) - 512
// 算完把 y 拷回 host，用 double 累加输出一行 SUM=<总和>，exit code 0。
// n = 0 时输出 SUM=0（0 个 block 的 launch 是非法的，直接特判跳过）。
// 本题不许 include common.h：错误检查宏和 cudaEvent 计时都自己写。

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 包住每个 CUDA API 调用，出错立刻报出文件、行号和原因。
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err_ = (call);                                \
        if (err_ != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",       \
                    cudaGetErrorName(err_), __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                    \
            exit(1);                                              \
        }                                                         \
    } while (0)

// kernel 启动本身没有返回值，要靠这两句查它的错误。
#define CUDA_CHECK_KERNEL()                   \
    do {                                      \
        CUDA_CHECK(cudaGetLastError());       \
        CUDA_CHECK(cudaDeviceSynchronize());  \
    } while (0)

__global__ void saxpy(int n, float a, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <n>\n", argv[0]);
        return 1;
    }
    long n = atol(argv[1]);
    if (n < 0) n = 0;

    // n = 0：没有元素可算，0 个 block 的 launch 是非法的，特判直接输出。
    if (n == 0) {
        printf("SUM=0\n");
        return 0;
    }

    size_t bytes = (size_t)n * sizeof(float);
    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    if (!h_x || !h_y) {
        fprintf(stderr, "host malloc failed\n");
        return 1;
    }
    for (long i = 0; i < n; i++) {
        h_x[i] = ((i % 2048) - 1024) * 0.5f;
        h_y[i] = (float)((i % 1024) - 512);
    }

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    // 向上取整：n 不是 256 整数倍时也要盖住全部元素。
    int blocks = (int)((n + threads - 1) / threads);

    // cudaEvent 计时：量的是 GPU 时间线上这段区间的耗时（毫秒）。
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    saxpy<<<blocks, threads>>>(/*n=*/(int)n, /*a=*/2.0f, d_x, d_y);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    double s = 0;
    for (long i = 0; i < n; i++) s += (double)h_y[i];
    printf("SUM=%.0f (n=%ld, %.3f ms)\n", s, n, (double)ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    free(h_x);
    free(h_y);
    return 0;
}
