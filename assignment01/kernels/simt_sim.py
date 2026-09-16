"""问题 1.6（选做）：SIMT Simulator —— 一个 warp 的执行模拟器。

不需要 GPU

contract: 实现 run(program) -> (regs, cycles)
- warp 固定 32 个 lane，lane i 的寄存器初值为 i（int）；
- program 是指令列表，指令是元组，共三种：
    ("add", k)   active lanes 的 reg += k，1 cycle
    ("mul", k)   active lanes 的 reg *= k，1 cycle
    ("if_lt", t, then_prog, else_prog)
        reg < t 的 lane 走 then_prog，其余走 else_prog。
        模拟器先带 mask 执行 then_prog，再带 mask 的补集执行
        else_prog，然后汇合。某一支没有 active lane 时整支跳过、
        不计拍。嵌套指令照常计拍（divergence 的代价就在这里）。
        if_lt 这条指令本身不计拍，拍数只来自实际执行到的 add / mul。
- 返回值 regs 是 32 个 lane 的最终寄存器值（list），cycles 是总拍数。

通过 pytest tests/test_simt_sim.py 即为完成。
"""


def run(program):
    regs = list(range(32))

    # exec_block 返回这段程序消耗的拍数，只改 active 的 lane。
    def exec_block(prog, active):
        cycles = 0
        for inst in prog:
            if inst[0] == "add":
                k = inst[1]
                for i in range(32):
                    if active[i]:
                        regs[i] += k
                cycles += 1
            elif inst[0] == "mul":
                k = inst[1]
                for i in range(32):
                    if active[i]:
                        regs[i] *= k
                cycles += 1
            elif inst[0] == "if_lt":
                t, then_prog, else_prog = inst[1], inst[2], inst[3]
                # 按 reg < t 把 active 的 lane 再切成两半；
                # 两个分支各自在"父 mask ∩ 分支条件"上执行。
                then_active = [active[i] and regs[i] < t for i in range(32)]
                else_active = [active[i] and regs[i] >= t for i in range(32)]
                if any(then_active):
                    cycles += exec_block(then_prog, then_active)
                if any(else_active):
                    cycles += exec_block(else_prog, else_active)
                # 汇合：if_lt 本身不计拍。
            else:
                raise ValueError(f"unknown instruction: {inst!r}")
        return cycles

    cycles = exec_block(program, [True] * 32)
    return regs, cycles
