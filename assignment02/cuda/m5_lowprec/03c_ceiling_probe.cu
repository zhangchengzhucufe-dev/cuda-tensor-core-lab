// 问题 5.3(c):ceiling 探针。
//
// 目标:测出"5.3(b) 这种访存形状的上限带宽"。方法是写一个和你的
// quant kernel 访存完全同形(读同样的 bf16、写同样位置的 8 byte 数据
// 与 1 byte SF)、但不做任何数学的 kernel——读进来的位 xor 一下直接
// 写出去即可。它的耗时就是这个访存模式在这块卡上的地板。
//
// 跑完把三个数放在一起:探针 GB/s、你的 quant kernel GB/s(03b 的
// 输出)、两者比值。报告里回答:你的 kernel 离自己的上限还有多远,
// 差距是访存还是计算(ncu 的 SM% / DRAM% 可以佐证)。
//
// 在下面实现探针 kernel 和 launch;main 不需要修改。
#include <vector>
#include <random>
#include "../common.h"
#include "nvfp4_common.h"

template <int BLOCK>
__global__ void probe_kernel(const __nv_bfloat16* __restrict__ in,
                             uint8_t* __restrict__ dataOut,
                             uint8_t* __restrict__ sfOut, int M, int K) {
    // 与 quant kernel 完全相同的访存形状:一线程一组,读 16 个 bf16,
    // 写 8 byte 数据 + 1 byte SF;区别只是不量化,读进来的位 xor 后
    // 直通(防编译器把访存优化掉)。
    long g = (long)blockIdx.x * BLOCK + threadIdx.x;
    int groupsPerRow = K / NVFP4_GROUP;
    long total = (long)M * groupsPerRow;
    if (g >= total) return;
    int r = (int)(g / groupsPerRow);
    int kg = (int)(g % groupsPerRow);

    const uint16_t* row = reinterpret_cast<const uint16_t*>(
        in + (size_t)r * K + kg * NVFP4_GROUP);
    uint8_t out[NVFP4_GROUP / 2];
    uint8_t acc = 0;
#pragma unroll
    for (int i = 0; i < NVFP4_GROUP; i += 2) {
        uint16_t a = row[i], b = row[i + 1];
        uint8_t lo = (uint8_t)(a ^ b);
        uint8_t hi = (uint8_t)((a >> 8) ^ (b >> 8));
        out[i / 2] = (uint8_t)((hi << 4) | lo);
        acc ^= lo ^ hi;
    }
    uint8_t* dst =
        dataOut + ((size_t)r * K / 2 + kg * (NVFP4_GROUP / 2));
#pragma unroll
    for (int i = 0; i < NVFP4_GROUP / 2; i++) dst[i] = out[i];
    sfOut[sf_swizzled_offset(r, kg, nvfp4_num_ktiles(K))] = acc;
}

static void launch_probe(const __nv_bfloat16* in, uint8_t* dataOut,
                         uint8_t* sfOut, int M, int K, int sms) {
    // 启动配置与 launch_nvfp4_quant 完全一致,保证对比公平。
    constexpr int BLOCK = 256;
    long total = (long)M * (K / NVFP4_GROUP);
    int grid = (int)((total + BLOCK - 1) / BLOCK);
    probe_kernel<BLOCK><<<grid, BLOCK>>>(in, dataOut, sfOut, M, K);
}

int main() {
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    for (const auto& shape :
         {std::pair{4096, 7168}, {16384, 4096}, {16384, 8192}}) {
        int M = shape.first;
        int K = shape.second;
        size_t n = (size_t)M * K;
        __nv_bfloat16* dx;
        uint8_t *dd, *dsf;
        CUDA_CHECK(cudaMalloc(&dx, n * 2));
        CUDA_CHECK(cudaMalloc(&dd, n / 2));
        CUDA_CHECK(cudaMalloc(&dsf, nvfp4_sf_bytes(M, K)));
        CUDA_CHECK(cudaMemset(dx, 0x3c, n * 2));
        float ms = time_avg_ms(
            [&] { launch_probe(dx, dd, dsf, M, K, sms); }, 50);
        CUDA_CHECK_KERNEL();
        double bytes = n * 2.0 + n * 0.5 + n / 16.0;
        printf("M=%-6d K=%-5d  probe %8.2f us  %6.0f GB/s\n", M, K, ms * 1e3,
               effective_gbps(bytes, ms));
        cudaFree(dx); cudaFree(dd); cudaFree(dsf);
    }
    return 0;
}
