# CUDA Tensor Core Lab

A bunch of CUDA kernels I wrote while working through the standard material
(PMPP 3rd ed., the GPU-mode lectures, assorted GEMM blogs). Everything runs
and self-checks against a CPU reference, and every example prints real
timings so you can see the optimization actually pay off. Difficulty is
intermediate: shared memory, warp intrinsics, atomics, WMMA tensor cores,
graphs -- no hand-written PTX, no CUTLASS.

Hardware these were tested on: RTX 3060 Laptop (Ampere, `sm_86`), CUDA 13.x.
Numbers below are from this card; yours will differ.

## Build & run

```bash
make          # builds everything into build/
make run      # builds + runs all examples, each one verifies itself
make clean
```

Individual examples take size args if you want them:

```bash
./build/02_memory/transpose_tiled 8192
./build/03_gemm/sgemm_tiled 2048 2048 2048
```

## What's in here

| dir | file | the point |
|---|---|---|
| `common/` | `error_check.h` | `CUDA_CHECK` macro, event timer, result comparison -- shared by everything |
| `01_basics/` | `device_query.cu` | SM count, shared memory limits, etc. The numbers every launch config decision comes from |
| | `vector_add.cu` | smallest possible kernel, plus the bandwidth math: 1 FLOP per 12 bytes = memory bound |
| `02_memory/` | `transpose_tiled.cu` | coalescing: a transpose always loses on one side until you use shared memory. `[TILE][TILE+1]` kills bank conflicts |
| | `reduction.cu` | three versions of sum: divergent -> interleaved -> warp shuffles + coarsening. Same FLOPs, ~6x apart |
| | `unified_memory.cu` | `cudaMallocManaged`, page migration cost, `cudaMemPrefetchAsync` |
| `03_gemm/` | `sgemm_naive.cu` | zero data reuse, single-digit % of peak |
| | `sgemm_tiled.cu` | shared memory tiling, the foundation cuBLAS/CUTLASS build on |
| | `sgemm_wmma.cu` | tensor cores via WMMA: fragments, `mma_sync`, FP16 in / FP32 accumulate |
| `04_scan/` | `prefix_sum.cu` | work-efficient Blelloch scan + the three-pass trick for arrays bigger than a block |
| `05_histogram/` | `histogram.cu` | atomics: global `atomicAdd` contention -> shared memory privatization + coarsening (~17x) |
| `06_stencil/` | `conv2d_tiled.cu` | 2D conv with halo loading; the divide/mod pattern handles edges and corners with no special cases |
| `07_streams/` | `stream_overlap.cu` | pinned memory + `cudaMemcpyAsync` + multi-stream pipelining |
| `08_gemm_opt/` | `sgemm_register_tile.cu` | 2D register tiling (8x8 per thread) + `float4` loads: the biggest single step in the GEMM ladder |
| | `sgemm_vs_cublas.cu` | same data, me vs cuBLAS. Reaches ~71% of cuBLAS at 2048^3 on this card |
| `09_nn_ops/` | `softmax_rowwise.cu` | safe softmax, thread-per-row vs block-per-row; reusable block reduction helper |
| | `layernorm.cu` | two block reductions, `E[x^2] - mean^2` variance, and why that's okay here |
| | `bias_gelu_fused.cu` | kernel fusion 101: skip a full read+write round trip of the intermediate |
| `10_attention/` | `attention_forward.cu` | online softmax (the Flash Attention trick): m/l/acc streaming update, no S x S score matrix in memory |
| `11_sparse/` | `spmv_csr.cu` | CSR SpMV, thread-per-row vs warp-per-row -- pick granularity to match row length |
| `12_sort/` | `bitonic_sort.cu` | branch-free compare-exchange network, O(n log^2 n), why GPUs tolerate that |
| `13_graphs/` | `cuda_graphs.cu` | stream capture + graph replay; kills most of the launch overhead for tiny kernels |

## Numbers worth knowing (RTX 3060 Laptop)

- reduction: divergent 35.7 GB/s -> warp shuffles 227.7 GB/s
- transpose: naive 73 GB/s -> tiled 190 GB/s
- GEMM at 2048^3: naive 536 GFLOPS -> register-tile ~4.7 TFLOPS -> cuBLAS ~6.6 TFLOPS
- histogram: naive 6.8 ms -> privatized 0.4 ms
- bias+GELU fusion: 2.0 ms -> 0.64 ms (memory bound, so fusion ≈ 2x)
- CUDA graphs: ~40-70% total time saved on a 3-tiny-kernel pipeline

## Two quirks of this test machine (native Linux won't have these)

- WSL2 doesn't support unified memory page migration: `cudaMemPrefetchAsync`
  returns `invalid device ordinal`. `unified_memory.cu` detects it and moves on.
- Consumer GPUs have a single copy engine, so the multi-stream example only
  gets ~1.03x. That's a hardware limit, not a bug.

## Bugs I actually hit while writing these

- block reductions: a `__shfl_sync` broadcast only crosses lanes, not warps.
  Warps 1-7 silently read garbage. Broadcast through shared memory instead.
- scan: the second pass needs an *exclusive* scan of block sums. I wrote an
  inclusive one and block 0 was off by exactly its own sum.
- non-deterministic float accumulation (`sums[bid] += out[i]` from 256
  racing threads) made my graphs example un-verifiable. Tree-reduce in
  shared memory instead.

## Ideas if you want to go further

- double-buffer the GEMM K loop with `cp.async`, measure again
- multi-head + multi-query-per-block attention, then flash style tiling
- try `ncu --set full` on the register-tile GEMM and chase the occupancy /
  register pressure numbers
