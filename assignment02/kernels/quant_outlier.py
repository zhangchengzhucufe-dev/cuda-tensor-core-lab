"""问题 5.1:per-tensor scale 与 outlier。

构造一个张量:一万个元素均匀分布在 [-1, 1],外加一个 3000 的
outlier。按 per-tensor 方式量化到 E4M3(scale = amax / 448,cast 用
torch.float8_e4m3fn),反量化后测逐点相对误差,填题面的表并回答三问。

需要动手的是下面两个 TODO;跑法:
    uv run python kernels/quant_outlier.py
输出直接用于报告,没有自动判测。
"""

import torch

E4M3_MAX = 448.0


def build_tensor(n: int = 10000, outlier: float = 3000.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n, generator=g) * 2 - 1
    return torch.cat([x, torch.tensor([outlier])])


def quant_dequant_per_tensor(x: torch.Tensor) -> torch.Tensor:
    """per-tensor E4M3 量化再反量化。"""
    amax = x.abs().max()
    scale = amax / E4M3_MAX
    q = (x / scale).to(torch.float8_e4m3fn)
    return q.float() * scale


def rel_err_at(x: torch.Tensor, y: torch.Tensor, value: float) -> float:
    """取 x 中最接近 value 的元素,返回该点的相对误差。"""
    i = (x - value).abs().argmin()
    return ((y[i] - x[i]) / x[i]).abs().item()


def main() -> None:
    x = build_tensor()
    y = quant_dequant_per_tensor(x)
    print("含 outlier:")
    for v in (0.5, 0.1, 0.01, 0.005, 3000.0):
        print(f"  x≈{v:<8} rel_err={rel_err_at(x, y, v):.3e}")

    # (a) 去掉 outlier 重新量化,对比 0.5 处的误差
    x_no = build_tensor()[:-1]
    y_no = quant_dequant_per_tensor(x_no)
    e_with = rel_err_at(x, y, 0.5)
    e_without = rel_err_at(x_no, y_no, 0.5)
    print(f"\n(a) 0.5 处误差: 含 outlier {e_with:.3e}, "
          f"不含 {e_without:.3e}, 变化 {e_with / e_without:.1f} 倍")

    # (b) 被量化成 0 的阈值:E4M3 的最小正规数是 2^-6,数值 / scale 不足
    #     最小表示的一半会舍入到 0。
    scale = x.abs().max() / E4M3_MAX
    small = torch.tensor([1e-3, 2e-3, 3e-3, 5e-3, 8e-3, 1e-2])
    for v in small:
        q = (v / scale).to(torch.float8_e4m3fn)
        print(f"  (b) x={v:.4f} -> 量化值 {q.float().item() * scale.item():.3e}"
              f" ({'0' if q.float().item() == 0 else '非零'})")

    # (c) 1x128 per-block scale:含 outlier 的 block scale 被 3000 抬高,
    #     其余 block 不受影响。
    xu = x[:-1]                    # 一万个均匀元素
    base = xu[: (len(xu) // 128) * 128]  # 截成 128 的整数倍,78 个 block
    blocks = base.view(-1, 128)
    s = blocks.abs().amax(dim=1, keepdim=True) / E4M3_MAX
    qb = (blocks / s).to(torch.float8_e4m3fn).float() * s
    for v in (0.5, 0.01):
        i = int((base - v).abs().argmin())
        b = i // 128
        err_blk = ((qb[b, i % 128] - base[i]) / base[i]).abs().item()
        print(f"  (c) 无 outlier 的 block, x≈{v}: "
              f"per-tensor 误差 {rel_err_at(x, y, v):.3e}, "
              f"per-block 误差 {err_blk:.3e}")
    # 含 outlier 的 block:把 3000 塞进一个 128 元素的 block 再量化,
    # 它的同 block 邻居退化成 per-tensor 的水平。
    blk = torch.cat([torch.tensor([3000.0]), xu[:127]])
    s2 = blk.abs().max() / E4M3_MAX
    q2 = (blk / s2).to(torch.float8_e4m3fn).float() * s2
    for off in (1, 60):
        err2 = ((q2[off] - blk[off]) / blk[off]).abs().item()
        print(f"  (c) 含 outlier 的 block, x≈{blk[off].item():.3f}: "
              f"误差 {err2:.3e}")


if __name__ == "__main__":
    main()
