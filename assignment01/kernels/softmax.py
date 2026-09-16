"""问题 7.8（选做）：softmax in Triton（FROM-SCRATCH）。

注：此题可以不用GPU (conftest.py 会自动切到 interpreter 模式)。

contract：
- softmax(x) 接收形状 (M, N) 的 2D tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 自己写，一个 program 处理一行；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意（用 mask 处理），可以假设 N <= 4096，BLOCK_SIZE 用
  triton.next_power_of_2(N) 是常见做法；
- 通过 pytest tests/test_softmax.py 即为完成。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, y_ptr, stride_row, n, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    # 越界位置填 -inf，这样后面 max 不会被它们干扰，exp(-inf) 也恰好是 0。
    x = tl.load(x_ptr + row * stride_row + offs, mask=mask,
                other=-float("inf"))

    # 数值稳定的关键：先减掉行内最大值，exp 的参数最大就是 0，不会溢出。
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(y_ptr + row * stride_row + offs, y, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    softmax_kernel[(M,)](x, y, x.stride(0), N, BLOCK_SIZE=BLOCK_SIZE)
    return y
