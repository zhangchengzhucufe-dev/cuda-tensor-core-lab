// 对比三种数据管理方式跑同一个向量加法（结果 = 2 * 输入）：
//   1. manual   : cudaMalloc + cudaMemcpy 手动搬运
//   2. unified  : cudaMallocManaged，CPU 访问结果时靠缺页自动迁回
//   3. prefetch : cudaMallocManaged + cudaMemPrefetchAsync 显式预取
// 编译: nvcc -O2 -I random memory/unified_memory.cu -o memory/unified_memory
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include "CPURandom.h"

__global__ void add(const float *arrC, float *arrG, const int n){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < n){
        arrG[tid] = arrC[tid] + arrC[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main(){
    const int n = 1 << 24;
    const int bytes = n * sizeof(float);
    const int iter = 20;

    // 备份一份原始输入，三种方式每次迭代都喂同一份数据
    float *arrC = new float[n];
    float *arrBackup = new float[n];
    for (int i = 0; i < n; i++) arrC[i] = dist_float(gen);
    for (int i = 0; i < n; i++) arrBackup[i] = arrC[i];

    // CPU 上的标准答案，最后核对 GPU 算对没有
    double expected = 0.0;
    for (int i = 0; i < n; i++) expected += 2.0 * arrBackup[i];

    /* ---------- 1. 手动管理：cudaMalloc + cudaMemcpy ---------- */
    float *d_arrC, *d_arrG;
    cudaMalloc((void **)&d_arrC, bytes);
    cudaMalloc((void **)&d_arrG, bytes);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iter; i++){
        cudaMemcpy(d_arrC, arrBackup, bytes, cudaMemcpyHostToDevice);
        add<<<120, 256>>>(d_arrC, d_arrG, n);
        cudaMemcpy(arrC, d_arrG, bytes, cudaMemcpyDeviceToHost);
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    double sumMan = 0.0;
    for (int i = 0; i < n; i++) sumMan += arrC[i];
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "manual management  total " << duration.count() << " ms, avg "
              << duration.count() / (double)iter << " ms/iter" << std::endl;

    /* ---------- 2. 统一内存：CPU 读结果触发缺页，自动迁回 ---------- */
    float *arrGpuUni, *arrGpuUniRes;
    cudaMallocManaged((void **)&arrGpuUni, bytes);
    cudaMallocManaged((void **)&arrGpuUniRes, bytes);

    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iter; i++){
        // CPU 重新写输入，相当于手动版的 H2D 拷贝
        for (int j = 0; j < n; j++) arrGpuUni[j] = arrBackup[j];
        add<<<120, 256>>>(arrGpuUni, arrGpuUniRes, n);
        cudaDeviceSynchronize();    // 先等内核结束，CPU 再碰这些页（WSL2 等不支持并发缺页的平台必须这样）
        // CPU 读结果：缺页处理自动把数据从 GPU 迁回 CPU
        // volatile 读触发每一页的迁移，又没有浮点加法链的额外开销
        float volatile *p = arrGpuUniRes;
        float volatile sink = 0.0f;
        for (int j = 0; j < n; j++) sink = p[j];
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    double sumUni = 0.0;
    for (int i = 0; i < n; i++) sumUni += arrGpuUniRes[i];
    auto durationUni = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);
    std::cout << "unified (demand)   total " << durationUni.count() << " ms, avg "
              << durationUni.count() / (double)iter << " ms/iter" << std::endl;

    /* ---------- 3. 统一内存 + cudaMemPrefetchAsync 显式预取 ---------- */
    float *arrGpuUniMan, *arrGpuUniManRes;
    cudaMallocManaged((void **)&arrGpuUniMan, bytes);
    cudaMallocManaged((void **)&arrGpuUniManRes, bytes);

    auto t5 = std::chrono::high_resolution_clock::now();
    bool prefetchOk = true;
    for (int i = 0; i < iter; i++){
        for (int j = 0; j < n; j++) arrGpuUniMan[j] = arrBackup[j];
        cudaError_t pe = cudaMemPrefetchAsync(arrGpuUniMan, bytes, 0);   // 输入批量预取到 0 号 GPU
        if (pe != cudaSuccess && prefetchOk){
            prefetchOk = false;   // WSL2 的半虚拟化驱动不支持预取，只会报一次
            std::cout << ">> cudaMemPrefetchAsync failed (" << cudaGetErrorString(pe)
                      << "), section 3 falls back to demand paging <<" << std::endl;
        }
        add<<<120, 256>>>(arrGpuUniMan, arrGpuUniManRes, n);
        cudaMemPrefetchAsync(arrGpuUniManRes, bytes, cudaCpuDeviceId);  // 结果批量预取回 CPU
        cudaDeviceSynchronize();    // 预取失败时这里兜底等内核，再读数据
    }
    auto t6 = std::chrono::high_resolution_clock::now();

    double sumUniMan = 0.0;
    for (int i = 0; i < n; i++) sumUniMan += arrGpuUniManRes[i];
    auto durationUniMan = std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5);
    std::cout << "unified (prefetch) total " << durationUniMan.count() << " ms, avg "
              << durationUniMan.count() / (double)iter << " ms/iter" << std::endl;

    std::cout << "checksum expected=" << expected
              << " manual=" << sumMan
              << " demand=" << sumUni
              << " prefetch=" << sumUniMan << std::endl;

    cudaFree(d_arrC);
    cudaFree(d_arrG);
    cudaFree(arrGpuUni);
    cudaFree(arrGpuUniRes);
    cudaFree(arrGpuUniMan);
    cudaFree(arrGpuUniManRes);
    delete[] arrC;
    delete[] arrBackup;
    return 0;
}
