"""问题 7.7（压轴）：softmax in TileLang（FROM-SCRATCH）。

contract：
- softmax(x) 接收形状 (M, N) 的 float32 CUDA tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 用 TileLang 自己写，一个 block 处理一行（或一小批行）；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意，可以假设 N <= 4096。TileLang 的 kernel 按形状编译，
  用 make_xxx(M, N) 针对形状生成、在 wrapper 里按形状缓存编译结果
  是常见做法（结构可以参考 7.3、7.4）；
- 归约用 T.reduce_max / T.reduce_sum，逐元素部分用 T.Parallel 加 T.exp；
- fragment 的宽度建议取不小于 N 的 2 的幂（类比 Triton 的
  next_power_of_2），不足的位置补 -inf（T.if_then_else 加 T.infinity），
  否则布局推断可能报 no available layout；
- 通过 pytest tests/test_tilelang_softmax.py 即为完成。

(Optional) 将你的实现和 torch.softmax 比较一下性能（行宽取 256/1024/4096），
Tip: elementwise + 行内归约的 kernel 大概率是带宽瓶颈，可以想想理论上限是多少。
"""

import torch
import tilelang
import tilelang.language as T

_kernel_cache = {}


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def make_softmax(M, N, dtype="float32", threads=128):
    # fragment 宽度取不小于 N 的 2 的幂；不足的位置补 -inf。
    BLOCK_N = _next_pow2(max(N, 1))

    @T.prim_func
    def softmax_kernel(
        X: T.Buffer((M, N), dtype),
        Y: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(M, threads=threads) as (m):
            row = T.alloc_fragment((BLOCK_N,), dtype)
            row_max = T.alloc_fragment((1,), dtype)
            row_sum = T.alloc_fragment((1,), dtype)

            # 装载：j >= N 的位置补 -inf（读 global 时把下标夹在 N-1，
            # 避免越界，值反正会被 -inf 盖掉）。
            for j in T.Parallel(BLOCK_N):
                row[j] = T.if_then_else(
                    j < N, X[m, T.min(j, N - 1)], -T.infinity(dtype))

            # 数值稳定的关键：先减掉行内最大值，exp 的参数最大是 0，不会溢出。
            T.reduce_max(row, row_max, dim=0, clear=True)
            for j in T.Parallel(BLOCK_N):
                row[j] = T.exp(row[j] - row_max[0])
            T.reduce_sum(row, row_sum, dim=0, clear=True)

            # 写回：只写前 N 个位置。
            for j in T.Parallel(N):
                Y[m, j] = row[j] / row_sum[0]

    return softmax_kernel


def softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    key = (M, N)
    if key not in _kernel_cache:
        _kernel_cache[key] = tilelang.compile(make_softmax(M, N))
    y = torch.empty_like(x)
    _kernel_cache[key](x, y)
    return y
