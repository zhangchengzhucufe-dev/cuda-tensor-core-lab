// 问题 1.3(FROM-SCRATCH):手写单 tile fp8 mma。
//
// 形状 mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32,
// 手工 fragment 装载(不用 ldmatrix),与 CPU 参考严格相等比较。
//
// fragment 映射直接用 1.1 判测过的公式(记 group = lane>>2,
// tig = lane&3,r = 寄存器序,j = 寄存器内 byte 序):
//   A(16x32,行主序存):
//     reg0: row = group,     col = tig*4 + j
//     reg1: row = group + 8, col = tig*4 + j
//     reg2: row = group,     col = tig*4 + 16 + j
//     reg3: row = group + 8, col = tig*4 + 16 + j
//   B(32x8,按 [k][n] 行主序存,消费按 col):
//     reg r 的 byte j: k = tig*4 + r*16 + j,n = group
//   D: d0,d1 在 row = group,col = tig*2 + {0,1};d2,d3 在 row = group+8
//
// 编译运行(需要 sm_89a 及以上的卡,fp8 mma 是 Ada/Hopper 起的指令):
//   ./judge_mma_fp8.sh 03_mma_fp8.cu          # 默认 ARCH=100f(B300)
//   ARCH=89a ./judge_mma_fp8.sh 03_mma_fp8.cu # 4090 等 Ada 卡
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <random>

constexpr int M = 16, N = 8, K = 32;

__global__ void mma_fp8(const uint8_t* A, const uint8_t* B, float* D) {
    int lane = threadIdx.x;
    int group = lane >> 2;
    int tig = lane & 3;

    // 手工装载 A:4 个 b32 寄存器,每个按公式收 4 个 fp8 字节。
    // 寄存器内 byte 序 = 低位在前(小端),即元素 i%4 放在第 i%4 个 byte。
    unsigned a[4];
#pragma unroll
    for (int r = 0; r < 4; r++) {
        unsigned reg = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int i = r * 4 + j;
            int row = group + ((r & 1) ? 8 : 0);
            int col = tig * 4 + (r >> 1) * 16 + j;
            reg |= (unsigned)A[row * K + col] << (8 * j);
        }
        a[r] = reg;
    }

    // 手工装载 B:2 个 b32 寄存器。
    unsigned b[2];
#pragma unroll
    for (int r = 0; r < 2; r++) {
        unsigned reg = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int k = tig * 4 + r * 16 + j;
            reg |= (unsigned)B[k * N + group] << (8 * j);
        }
        b[r] = reg;
    }

    float c[4] = {0.f, 0.f, 0.f, 0.f}, d[4];
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));

    D[group * N + tig * 2] = d[0];
    D[group * N + tig * 2 + 1] = d[1];
    D[(group + 8) * N + tig * 2] = d[2];
    D[(group + 8) * N + tig * 2 + 1] = d[3];
}

int main(int argc, char** argv) {
    unsigned seed = argc > 1 ? (unsigned)atoi(argv[1]) : 42;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-8, 7);  // e4m3 精确表示
    uint8_t hA[M * K], hB[K * N];
    float fA[M * K], fB[K * N], ref[M * N] = {};
    for (int i = 0; i < M * K; i++) {
        __nv_fp8_e4m3 v = __nv_fp8_e4m3((float)dist(rng));
        hA[i] = *(uint8_t*)&v;
        fA[i] = float(v);
    }
    for (int i = 0; i < K * N; i++) {
        __nv_fp8_e4m3 v = __nv_fp8_e4m3((float)dist(rng));
        hB[i] = *(uint8_t*)&v;
        fB[i] = float(v);
    }
    for (int r = 0; r < M; r++)
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
                ref[r * N + n] += fA[r * K + k] * fB[k * N + n];

    uint8_t *dA, *dB;
    float* dD;
    if (cudaMalloc(&dA, sizeof(hA)) != cudaSuccess ||
        cudaMalloc(&dB, sizeof(hB)) != cudaSuccess ||
        cudaMalloc(&dD, sizeof(float) * M * N) != cudaSuccess) {
        printf("FAIL: cudaMalloc\n");
        return 1;
    }
    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);
    mma_fp8<<<1, 32>>>(dA, dB, dD);
    if (cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        printf("FAIL: kernel launch\n");
        return 1;
    }
    float got[M * N];
    cudaMemcpy(got, dD, sizeof(got), cudaMemcpyDeviceToHost);

    long bad = 0;
    for (int i = 0; i < M * N; i++)
        if (got[i] != ref[i]) {
            if (bad < 5)
                printf("MISMATCH D[%d][%d]: got %.1f want %.1f\n", i / N,
                       i % N, got[i], ref[i]);
            bad++;
        }
    if (bad)
        printf("MISMATCH seed=%u: %ld / %d\n", seed, bad, M * N);
    else
        printf("PASS seed=%u\n", seed);
    return bad != 0;
}
