import tilelang
import tilelang.language as T


def make_mm(M=128, N=128, K=128, BM=64, BN=64, BK=32, dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            A_s = T.alloc_shared((BM, BK), dtype)
            B_s = T.alloc_shared((BK, BN), dtype)
            C_l = T.alloc_fragment((BM, BN), "float32")
            T.clear(C_l)
            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
                T.copy(A[by * BM, k * BK], A_s)
                T.copy(B[k * BK, bx * BN], B_s)
                T.gemm(A_s, B_s, C_l)
            T.copy(C_l, C[by * BM, bx * BN])
    return main


if __name__ == "__main__":
    import sys
    outdir = "/home/zcz/xinwork/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/m6_lowering"
    k = make_mm()
    for name, tgt in [("sm_90a", {"kind": "cuda", "arch": "sm_90a"}),
                      ("sm_100a", {"kind": "cuda", "arch": "sm_100a"})]:
        try:
            kern = tilelang.compile(k, target=tgt, verbose=False)
            src = kern.get_kernel_source()
            open(f"{outdir}/{name}_gemm.cu", "w").write(src)
            print(name, "compiled,", len(src), "bytes")
        except Exception as e:
            print(name, "FAILED:", str(e)[:120].replace("\n", " "))
