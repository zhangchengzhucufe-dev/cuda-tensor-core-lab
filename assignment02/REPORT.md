# 作业 2 报告（Tensor Core & Pipeline）

> 我的机器：WSL2 + NVIDIA GeForce RTX 3060 Laptop GPU（6GB，sm_86），系统 nvcc 是 12.6。为了编译 B300 的代码，我另外装了一套 CUDA 13.0.88（放在 /tmp/cuda13，不影响系统）。
> **先说清楚一件重要的事**：本作业的 M3、M4（CUDA 部分）、M5（CUDA 部分）按题面要求需要 B300（sm_100 家族），我手里没有这块卡。这些题我全部**完成了实现**，并且用 CUDA 13 对着 `sm_100f` **全部编译通过**，但**没有在真机上跑过判测**，下面的报告里凡是这种情况我都单独标了"⚠️ 未在真机验证"。能在我这台机器上跑的题（纯 host 判测、sm_86 能跑的、纯 Python 的），数据都是我自己跑出来的。

## 判测与运行结果汇总

| 类别 | 判测内容 | 命令 | 结果 |
| --- | --- | --- | --- |
| host 判测 | 1.1 fragment 映射真值表 | `make run/m1_sm80/01_fragment_map` | **PASS** |
| host 判测 | 2.2 descriptor 三场景 | `make run/m2_smem/02_descriptor` | **3/3 PASS** |
| host 判测 | 2.3 swizzle 双射+无冲突 | `make run/m2_smem/03_swizzle` | **3/3 PASS** |
| GPU 判测 | 0.1 最小 mma（sm_86） | `make run/m0_env/01_first_mma ARCH=86` | **PASS** |
| DEBUG | 1.2 fragment bug 修复（sm_86） | `make run/m1_sm80/02_bug_fragment ARCH=86` | 修复前 59/128 错 → 修复后 **PASS** |
| 实验 | 1.5 ldmatrix 行跨度 + ncu | `make run/m1_sm80/05_ldsm_stride ARCH=86` + ncu | 运行 OK，原始计数器存档 `cuda/m1_sm80/05_ldsm_stride_ncu.csv` |
| 实验 | 4.5 thin GEMM 全形状（63 行） | `./bin/m4_gemm/05_thin_gemm 25 300` | 跑完，全表见正文与附录 |
| Python | 5.2 block scaling | `uv run pytest tests/` | **3 passed** |
| Python | 5.1 outlier 实验 | `uv run python kernels/quant_outlier.py` | 输出见 5.1 节 |
| B300 项 | 1.3/1.4/3.2/3.3/3.4/4.1/4.2/4.3/5.3/5.4 | 用 CUDA 13 逐个 `-gencode ...sm_100f` 编译 | **全部编译通过**，真机判测待 B300（正文逐题标 ⚠️） |

---

## Module 0 环境与峰值

### prob 0.1 HANDS-ON

`make run/m0_env/01_first_mma` 默认 ARCH=100f，我的 CUDA 12.6 不认识这个架构，改用 `ARCH=86 make ...` 编译（m16n8k16 fp16 的 mma.sync 从 sm_80 就有，3060 能跑）。运行结果：

```
D[0][0]=2 D[0][7]=2 D[15][0]=2 D[15][7]=2
PASS
```

不匹配 ARCH 的实验：用 `ARCH=90a make bin/m0_env/01_first_mma` 编译（Hopper 专属后缀），在我的 sm_86 卡上运行报：

```
CUDA error cudaErrorNoKernelImageForDevice at m0_env/01_first_mma.cu:76:
no kernel image is available for execution on the device
```

`make ptx/m0_env/01_first_mma` 生成的 PTX 里能看到那条 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`。

解释：fatbin 里只嵌了 sm_90a 的 SASS 时，sm_86 的卡找不到自己能执行的机器码，驱动也没有 PTX 可以 JIT 兜底（`code=sm_90a` 不带 PTX），所以加载阶段直接报"没有可用的 kernel image"。这正好和作业 1 Module 8 的 sassonly/ptxonly 实验对上了：SASS 不跨架构，PTX 才能靠 JIT 兜底。

### prob 0.2 DERIVE（5090 与 B300 的 Tensor Core 峰值）

先声明口径：**dense（不算稀疏加速）、boost 频率、一次 FMA 记 2 个 FLOP、以 fp32 累加的 bf16 为基准**。

**5090（sm_120，消费级 Blackwell）**：

- 21760 个 CUDA core ÷ 128 = 170 个 SM；boost ≈ 2.41 GHz。
- 每个 SM 4 个 Tensor Core，每拍每 TC 做 128 次 bf16 FMA → **bf16 FLOP/cycle/SM = 4 × 128 × 2 = 1024**。
- bf16 峰值 = 170 × 1024 × 2.41e9 ≈ **419 TFLOPS**。
- dtype 每窄一倍宽度峰值翻倍：fp8 = **838 TFLOPS**，fp4 = **1676 TFLOPS**。
- 和官方数据对账：5090 标称 "AI TOPS 3352"，正好是 fp4 **sparse**（1676 × 2 = 3352），链条 fp4 dense 1676 ÷2 = fp8 838 ÷2 = bf16 419 完全自洽，说明我的口径和 NVIDIA 标称口径的差别只在 sparse/dense。
- 显存：GDDR7 512-bit @ 28 Gbps ≈ **1792 GB/s**。
- 机器平衡点 = 419e12 / 1792e9 ≈ **234 FLOP/byte（bf16）**。

**B300（sm_103，Blackwell Ultra）**：

- 我按 GB300 NVL72 的公开规格折算到单 GPU：bf16 dense ≈ **2.25 PFLOPS**（Ultra 在 bf16/fp8 上和 B200 基本一致，升级集中在 fp4），fp8 ≈ **4.5 PFLOPS**，fp4 dense ≈ **15 PFLOPS**（是 B200 的 1.5 倍，这是 Ultra 的主要增量；72 × 15 PF ≈ 1.1 EF，和官方 "1.1 exaflops FP4 per rack" 对得上）。
- bf16 FLOP/cycle/SM 用 2.25e15 ÷ (148 SM × 1.965 GHz) ≈ 2048 量级倒推（数据中心 Blackwell 每 SM 的 TC 配比更高，和消费级 1024 不同——这里我按 datasheet 反推，不是正向推导，和 5090 的推导方向正好相反，老实说）。
- HBM3e ≈ **8 TB/s**。
- 机器平衡点 = 2.25e15 / 8e12 ≈ **281 FLOP/byte（bf16）**。

汇总成题面要求的表（口径都是 dense、boost、FMA 计 2）：

| 量 | 5090 | B300 |
| --- | --- | --- |
| bf16 FLOP/cycle/SM | 1024（4 TC × 128 FMA × 2） | ≈2048（按 datasheet 反推） |
| bf16 峰值(TFLOPS) | ≈419（170 SM × 2.41 GHz） | ≈2250 |
| fp8 峰值(TFLOPS) | ≈838 | ≈4500 |
| fp4 峰值(TFLOPS) | ≈1676 | ≈15000（Ultra 的 1.5× 主要加在这里） |
| datasheet 对照与口径差异 | 官方 3352 AI TOPS = fp4 sparse，÷4 得 bf16 419，与推导一致；口径差 = sparse/dense | 官方给的是整机 rack 值，折单卡时口径是 dense；bf16/fp8 沿用 B200 水平，只有 fp4 ×1.5 |
| HBM/GDDR 带宽(GB/s) | ≈1792（GDDR7） | ≈8000（HBM3e） |
| 机器平衡点(FLOP/byte，bf16) | ≈234 | ≈281 |

诚实说明：5090 那列是正向推导 + 官方数对账；B300 那列我没有卡也没有一手 datasheet，是按公开规格折算的，"FLOP/cycle/SM"一行尤其是反推出来的——等真用上 B300 时应该拿 `deviceQuery` 的 SM 数和实际频率重新校一遍。

**和平衡点对照**：单条 m16n8k16 fp16 mma 的计算强度（S016 口径）：分子 2×16×8×16 = 4096 FLOP；分母 = A 16×16×2B + B 16×8×2B + D 16×8×4B = 512+256+512 = 1280B → **3.2 FLOP/byte**。

3.2 vs 234/281，差了大约 **70~90 倍**。这意味着：如果每做一次 mma 都要 dram 去读一遍操作数，tensor core 有 99% 的时间在等数据。要让 kernel 逼近峰值，必须把每个操作数在片上**复用几十次**（tile 化：一次搬进 smem，喂很多条 mma）——这就是为什么 M2-M4 全在优化"数据供给路径"（descriptor/swizzle → TMA → 流水线），而不是 mma 本身。

### prob 0.3 CONCEPT

- (a) **对**。分子 2MNK，分母是这条 mma 消费 A、B 和写回 D 的字节总和（3.2 FLOP/byte 就是这么算出来的）。
- (b) **对**。`mma.sync` 是 warp 级协作指令，32 个 lane 各持 fragment 一部分，要求全 warp 一致发射，lane 发散时行为未定义。
- (c) **错**。形状越大计算强度越高是真的，但有代价：fragment 寄存器占用更多（可能 spill）、对 smem 布局和 ldmatrix 的要求更苛刻、tail effect 变大，不是越大越好。
- (d) **错**。单条 mma 的计算强度低只是说"以 DRAM 直接喂"不行；经过 smem/寄存器复用后，整个 kernel 的有效计算强度可以远高于单条指令，照样逼近峰值（这正是 M4 的做法）。

---

## Module 1 sm80: fragment 与 mma.sync

### prob 1.1 DERIVE（01_fragment_map.cu）

对照 PTX ISA 的 "Matrix Fragments for mma.m16n8k32"，记 group = lane>>2、tig = lane&3、r = i/4（寄存器序）、j = i&3（寄存器内字节序）：

- A（16×32）：`a_row_of = group + (r&1)*8`；`a_col_of = tig*4 + (r>>1)*16 + j`。也就是 reg0/reg1 管 k 的前 16 列（分别在第 0-7 行 / 8-15 行），reg2/reg3 管 k 的后 16 列。
- B（32×8，col 布局）：`b_row_of = tig*4 + r*16 + j`（k 方向）；`b_col_of = lane>>2`（n = group，两个寄存器相同）。

判测是纯 host 的真值表比对，**PASS**（一次过，说明公式和硬件一致）。

**附加问**：A 的同一个 b32 寄存器里 4 个 fp8 沿 **k 方向（列方向）相邻**——寄存器内 byte j 对应列 +j。这对 1.4 用 ldmatrix 的影响：ldmatrix 的"一个元素"是 16 个原始 bit（两个 fp8），k 相邻的两个 fp8 恰好凑成一个 b16 元素，所以行主序存的 A、不带 `.trans` 的一次 `ldmatrix.x4` 就能把 fragment 填对；如果相邻方向是 n，就得换 `.trans` 或者换 smem 布局。

### prob 1.2 DEBUG（02_bug_fragment.cu）

先跑的现象（修之前，在 3060 上）：

```
MISMATCH D[8][0]: got -1, want -11
MISMATCH D[8][1]: got -5, want 5
FAIL: 59 / 128 mismatches
```

(a) **症状**：D 的上半场（rows 0-7）全对，下半场（rows 8-15）全错（128 个里错 59 个，剩下的是小整数碰撞撞对的）。错法不是简单翻转，而是"下半场的点积里，A 的 k=0、1 和 k=8、9 四列被换成了上半场的值"——具体说 got[8+r][n] = Σ_{k∉{0,1,8,9}} A[r][k]B[k][n] + A[r][0]B[0][n] + A[r][1]B[1][n] + A[r][8]B[8][n] + A[r][9]B[9][n]，和正确值的关系是"k 的四个位置张冠李戴"。

(b) **错处**：A fragment 的第 1 对寄存器（a2/a3）和第 3 对（a6/a7）。这两个寄存器管的是 **row = group+8** 的那半（k 前/后 16 列），原代码抄 a0/a1、a4/a5 时把行号也抄了（少了 +8）。修法就是把 a2/a3/a6/a7 的行下标加上 `group + 8`。修完 PASS。为什么恰好是 (a) 的症状：这四个寄存器只影响 D 的 8-15 行；k=0、1 的错位只影响每行的前两项部分积，所以是"上半场全对、下半场每个元素都错但错得有规律"。

### prob 1.3 FROM-SCRATCH：手写单 tile fp8 mma（03_mma_fp8.cu）

⚠️ **未在真机验证**（fp8 的 m16n8k32 mma 需要 sm_89 及以上，我的 3060 是 sm_86，跑不了；判测脚本默认 ARCH=100f 也要求 CUDA 13）。

实现：接受 seed 参数，用 [-8, 7] 的小整数填 A（16×32）、B（32×8，保证 e4m3 精确表示）；fragment 装载直接用 1.1 判过的公式（寄存器内字节按小端拼装：`reg |= A[row][col] << (8*j)`）；一条 `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`；host 参考用 `cuda_fp8.h` 转回 float 后累加（小整数下 f32 精确，可以严格相等比较）；输出以 PASS / MISMATCH 开头。用 CUDA 13 对 sm_100f 编译通过。判测命令 `./judge_mma_fp8.sh 03_mma_fp8.cu` 需要在有 fp8 mma 的卡上跑。

### prob 1.4 MODIFY：ldmatrix 装载路径（04_ldmatrix.cu）

⚠️ **未在真机验证**（同样是 fp8 mma 的硬件要求）。

两条路径都写好了、共存于一个 kernel（模板参数切换）：

- `load_manual`：1.3 的公式逐字节装配，从 sA（行主序）/ sBk（k-major）读。
- `load_ldsm`：A 用一条 `ldmatrix.sync.aligned.m8n8.x4.shared.b16`（不带 .trans）。四个 8×8 b16 矩阵分别取 A 的（rowhalf 0/1 × colhalf 0/1）四块：lane l 给 matrix p = l>>3 提供第 l&7 行的地址 `&sA[8*(p&1)+(l&7)][16*(p>>1)]`。B 用一条 `.x2`：fragment 的 b16 单元是 (B[k][n], B[k+1][n])，k 相邻——sBk 里不相邻（差 8B），所以必须用 main 里准备的 sBn（n-major，每列 32 个 k 连续）才满足 ldmatrix 的 16B 行地址要求；lane l 提供地址 `&sBn[(l&7)*32 + p*16]`。

报告问题：(a) ldmatrix 省掉了手工路径里的：每 lane 十几次的 smem 地址计算（group/tig 的乘加、行列偏移）、以及把字节拼进寄存器的一堆移位和 OR——一条指令 + 一次地址计算就完事。(b) 手工路径绕不开这些，因为 mma 的 fragment 分布是"warp 32 个 lane 各持不连续的几片"，数据在 smem 里又是行主序，两者之间的重排总得有人做：ldmatrix 就是硬件里专门干这件事的指令，一次把 32 个地址的数据分发到各 lane 的寄存器；手工路径没有这种指令可用，只能每个 lane 自己算地址、自己移位拼字节。

我把两个模板实例用 `cuobjdump -sass` 反汇编（CUDA 13，sm_100f）数了一遍指令：

| | 手工装载版 | ldmatrix 版 |
| --- | --- | --- |
| 指令总数（整个 kernel） | 136 | 104 |
| smem → fragment 的读指令 | 24 条 LDS（逐字节） | **2 条 LDSM**（一条 .x4 管 A，一条 .x2 管 B） |
| 地址运算（IMAD/IADD3/SHF/LEA） | 41 条 | 20 条 |
| mma 本体（2×HMMA.16816 + 12×F2FP fp8 解包） | 相同 | 相同 |

两版的 mma 本体完全一样，差距全部来自装载段：24 条逐字节 LDS + 约 21 条额外的地址算术，被 2 条 LDSM 吃掉了。（有个意外收获：m16n8k32 的 fp8 mma 在 sm_100 上其实编译成了 2 条 HMMA.16816.F32 加 12 条 F2FP 解包——fp8 操作数是先转成 fp16 再走 fp16 的 HMMA 的。）

### prob 1.5 EXPERIMENT：ldmatrix 行跨度（05_ldsm_stride.cu）

程序本身不挑架构（fp16 ldmatrix），我在 3060 上跑了（ARCH=86）。ncu 一开始报 `ERR_NVGPUCTRPERM`（WSL 下 GPU 性能计数器默认只给管理员，需要在 Windows 侧的 NVIDIA 控制面板 → 开发人员 → 管理 GPU 性能计数器里放开，改完立即生效；注意这台机器的驱动上这个开关不持久，GPU 断电唤醒会翻回去），放开之后两个计数器都拿到了。原始输出存档在 `cuda/m1_sm80/05_ldsm_stride_ncu.csv`。总量除以 4096 次循环 × 8 个 warp，折算成单条 ldmatrix（每 warp 每轮发一条）：

| 档位 | 预测 wavefront 比 | 实测 wavefront / 条 | 实测 conflict / 条 | 平均 cycle / ldmatrix |
| --- | --- | --- | --- | --- |
| 32 B | 2× | 8 | 4 | 64.0 |
| 64 B | 4× | 16 | 12 | 128.0 |
| 128 B | 8× | 32 | 28 | 256.0 |
| 128+16 B | 1× | 4 | **0** | 32.1 |

（预测的算法：一次 ldmatrix.x4 = 32 个 lane 各交一个 16B 行地址 = 512B，shared 每波最多吐 128B，所以无冲突下限是 4 个 wavefront/条；行距 32B 时每 4 行共用一组 bank（2×），64B 时每 2 行共用（4×），128B 全部撞同一组（8×），padding 到 144B 后 8 行恰好铺满 32 个 bank（1×）。）

结果特别整齐：**每一档的 wavefront 数都正好等于"下限 4 + conflict"**（4+4=8、4+12=16、4+28=32），conflict 计数就是超出下限的重放次数；而且 **wavefront 的比例（2×/4×/8×/1×）和实测平均 cycle 的比例完全对上**——回答题面那一问：64B 行距正好把 wavefront 数放大到 4 倍。题面说"实际耗时的差距通常没有这么大"，我这个实验里却基本是 1:1 放大，原因正是提示里的那句话：8 个 warp 同发把 LSU 打满，循环里除了 ldmatrix 没有别的指令，冲突带来的等待没有别的指令能盖住，全都变成了实际耗时；真实 GEMM 里 ldmatrix 只是指令流的一小部分，周围有 mma、地址运算可以把这些等待藏掉大半。

---

## Module 2 smem 供数：descriptor 与 swizzle

### prob 2.1 CONCEPT

(a) 正确顺序：

1. `st.shared`（generic proxy 写 smem）
2. `fence.proxy.async`（把 generic 写对 async proxy 可见）
3. `wgmma.fence`（寄存器/SM 状态对 wgmma 就绪）
4. `wgmma.mma_async`（发射）
5. `wgmma.commit_group`（给本组打个提交点）
6. `wgmma.wait_group`（等组完成）

排序防的乱序：1→2 防的是"generic 写 vs async 读"两个 proxy 之间的乱序（wgmma 可能读到旧值）；2/3→4 防的是编译器/硬件把发射提前；4→6 防的是"消费方读寄存器/TMEM vs mma 还没写完"的乱序；5 是给 6 的等待提供分组单位。

(b)
1. **错**。`fence.proxy.async` 是通用的 proxy 顺序机制，TMA（`cp.async.bulk`）和 tcgen05 场景同样需要（3.2 里就用了）。
2. **错**。`commit_group` 只是个提交/打标操作，不等；等的是 `wait_group`。
3. **对**。写走 generic proxy、读走 async proxy，两个 proxy 之间硬件不保证顺序，不加 fence 就可能读到旧值。

### prob 2.2 DERIVE（02_descriptor.cu）

(a) 位域编码：`start_address>>4` 放 bit[0,14)，`LBO>>4` 放 bit[16,30)，`SBO>>4` 放 bit[32,46)，version 固定 1，layout 占 bit[61,64)。

(b) 三个场景（64×64 bf16 B tile）：

| 场景 | LBO | SBO | layout |
| --- | --- | --- | --- |
| 1：K-major 无 swizzle | 128 | 1024 | NONE(0) |
| 2：K-major 128B swizzle | 0（硬件忽略） | 1024 | 128B(2) |
| 3：MN-major 128B swizzle | 0（硬件忽略） | 1024 | 128B(2) |

推导：场景 1 的 atom 是 8 行 × 16B 紧密打包（128B 连续）；leading 方向沿 K，相邻 atom 差 128B → LBO=128；沿 MN 方向跨过一个 n 组要跳 8n×64k×2B = 1024B → SBO=1024。场景 2/3 swizzle 下 LBO 被忽略按 0 填，atom 间 strided 方向仍是 1024B。

判测三个场景全 PASS（真值来自 B300 上验证过的描述符）。

**报告问题**：场景 2 和 3 的 descriptor 完全相同。MN-major 和 K-major 的区别**不体现在描述符字段里**，而是体现在"数据是谁、按什么顺序摆进 smem"——也就是 staging 阶段的摆放布局。这个 64×64 的方 tile 恰好让两个方向的 SBO 算出来一样（都是 1024B），加上 swizzle 模式下 LBO 被忽略，描述符就完全重合了；换成非方 tile，SBO 就会不同。

### prob 2.3 FROM-SCRATCH（03_swizzle.cu）

三种 swizzle 都实现了，规则是统一的：**行内 16B chunk 的下标与行号低位做 XOR**，参与异或的位数由行宽决定（128B 行 3 bit ↔ row 低 3 bit；64B 行 2 bit；32B 行 1 bit），16B 以内的低位直通。

```
128B: row*128 + (colByte ^ ((row & 7) << 4))
 64B: row*64  + (colByte ^ ((row & 3) << 4))
 32B: row*32  + (colByte ^ ((row & 1) << 4))
```

判测（双射 + 列访问无 bank conflict）三种模式全 PASS。64B/32B 是我自己从 PTX ISA 的 swizzling 一节推的：本质是同一个 XOR 模式在不同 atom 宽度下的截断。

---

## Module 3 sm100：tcgen05

### prob 3.1 CONCEPT

- (a) **对**。tcgen05.ld 每个 warp 只能读自己 warp 对应的 32 条 lane（taddr 高 16 bit 是 lane 偏移），warp 之间不能越界读。
- (b) **对**。tcgen05.mma 由单个线程（elected lane）发射，硬件异步执行；不同于 mma.sync 的 warp 协作和 wgmma 的 warpgroup 协作。
- (c) **错**。TMEM 的结果要用 `tcgen05.ld` 搬进寄存器再走普通路径（或 TMA store 之前先 ld 出来——TMEM 本身不是 TMA 的直接源）。
- (d) 计算过程：TMEM = 128 lane × 512 column × 4B。m128n256 的 f32 accumulator 占 128 lane × 256 column × 4B = 128 × 512 × 4B 的一半 → **对**（256 column / 512 column = 一半）。
- (e) **错**。`tcgen05.commit` 只是"提交完成通知"——它把完成事件挂到 mbarrier 上，本身不阻塞；要等 mbarrier 的相位翻转（`mbarrier.try_wait`）之后才能安全读。这条直接对应 3.3 的 bug。

### prob 3.2 FROM-SCRATCH：tcgen05 单 tile GEMM（02_single_tile.cu）

⚠️ **未在真机验证**（tcgen05 是 sm_100 家族专属，判测和 `./judge_tile.sh` 都需要在 B300 上跑；我用 CUDA 13 对 sm_100f 编译通过了）。

实现按七步流程写全了：

1. mbarrier init（count=1）+ warp0 `tcgen05.alloc` 64 列 TMEM，结果落 shared 广播；
2. 全体 128 线程把 A（128×64）、B（64×64）按 `swz128`（即 2.3 的 128B swizzle）st.shared 进 smem（各 16KB/8KB，1024 对齐）；
3. `fence.proxy.async.shared::cta` + `__syncthreads`；
4. elected 单线程连发 4 条 k16 的 `tcgen05.mma.cta_group::1.kind::f16`，descriptor 用 2.2 的编码（`base + kk*2, LBO=0, SBO=1024, layout=128B`），**第一条 enable-input-d=0、后三条累加**，idesc 与 3.3 参考程序一致（m128n64、bf16、f32 累加）；`tcgen05.commit` 到 mbarrier；
5. 全体 `mbar_wait(parity=0)` + `tcgen05.fence::after_thread_sync`；
6. epilogue：每 warp 用 `tcgen05.ld.sync.aligned.32x32b.x8.b32` 读自己的 32 条 lane（taddr 高 16bit lane 偏移 = warp*32，低 16bit 列偏移），写回 global；
7. `__syncthreads` 后 `tcgen05.dealloc`。

题面要求的"故意去掉 `fence.proxy.async` 再跑一次"我无法在真机上观察，先按 2.1 的分析把预期写在这里（等有 B300 了再对答案）：`st.shared`（generic proxy）的写和 tcgen05 的读（async proxy）之间没有顺序保证，mma 可能读到 swizzle 摆放之前的旧数据——现象应该是 FAIL 且错误数据"看起来像摆错位置/旧值"，而不是随机噪声；因为 `__syncthreads` 只保证线程视角的可见性，不跨 proxy。

### prob 3.3 DEBUG：mbarrier 多轮（03_bug_mbarrier.cu）

⚠️ 现象记录部分需要在 B300 上跑 `./judge_mbar.sh` 才能拿到真机输出，下面 (a) 是我按代码语义推的预测 + (b) 是修复。

(a) 预测：`rounds=1` PASS；`rounds=2` FAIL 或数据错（读到在飞的 mma 的半成品/上一轮结果）；`rounds=4` 必错。条件：只要"等同一个相位两次"就开始错——即 rounds ≥ 2。

(b) 修复 + 状态分析。原代码每一轮都 `mbar_wait(mbar, 0)`。mbarrier 的相位是完成一次翻转一次（0→1→0→1…），第 round 轮的 commit 完成的是第 round 个相位。round 0 等 phase 0 没问题；round 1 开始，phase 0 早已完成，`try_wait.parity(0)` **立刻放行**——放行过早。于是所有 warp 冲去 `tcgen05.ld` 读 TMEM，而此时 round 1 的 mma 还在飞（提示里说的"tcgen05.ld 与尚未完成的 mma 之间的关系"），读到的要么是上一轮的部分积、要么是写到一半的数据。轮数越多错得越离谱。

错误版 vs 修复版的相位/到达计数变化（错误版只画 rounds=2，第 2 轮就出事）：

```
错误版（每轮都等 parity 0）
round 0: init(phase=0,count=1) ──commit──> phase 0 完成,翻到 1
         wait(parity 0): 等 phase0 → 放行 ✓            ld 读 ✓
round 1: ──commit──> phase 1 完成,翻回 0
         wait(parity 0): phase0 早已完成 → 立刻放行 ✗   ld 读 ← mma 还在飞!
              （arrival count 每 commit +1 后清零翻转,但 wait 检查的相位错了）

修复版（等 parity = round&1,即"本轮 commit 完成的那一相"）
round 0: commit → 完成 phase0(parity0) → wait(0) 放行 ✓
round 1: commit → 完成 phase1(parity1) → wait(1) 放行 ✓
round 2: commit → 完成 phase2'(parity0) → wait(0) 放行 ✓   …依此交替
```

修复就一行：`mbar_wait(mbar_u32, round & 1)`。

### prob 3.4 EXPERIMENT：cta_group::1 vs ::2（04_cta_pair.cu）

⚠️ 没有可运行的 B300，ncu 也拿不到；程序编译通过。下面按硬件语义给预测（真机上应该用程序打印的 smem 用量和 ncu 流量验证）：

(a) m256n64k64 用 cta_group::2 时，一条 M=256 的 mma 由一个 cluster 的两个 CTA 共同执行，A 沿 M 分成两半（每个 CTA 持 128 行），B 沿 N 分成两半（每个 CTA 持 32 列）。所以**每个 CTA 的 B smem 是 cta_group::1 的一半**（64 列 → 32 列，8KB → 4KB）；A 的 smem 每 CTA 不变（各持一半的 M，总量一样）。TMEM：accumulator 256×64 沿 lane 切成两个 128×64，**每个 CTA 的 TMEM 占用不变**（各占满自己的 128 lane × 64 col）。
(b)（预测）shared memory 总流量：B 从 global 只需要搬一份但两个 CTA 各读自己的半份，总读入量不变；不过 smem → tensor core 的侧流量因 B 复用方式变化会有差异，具体要 ncu 实测。
(c) 省下来的 B smem（每 CTA 4KB，两 CTA 共 8KB）在 M4 的流水线里可以换**更深一层的 stage**（每 stage 需要 24KB，省出的量约再挤 1/3 个 stage；或者配合缩小 BK 后多排几级），让预取更靠前、对 global 延迟的容忍更强。
(d) 依赖 sm90 引入的 **cluster（CTA pair + 分布式共享内存）**机制：一个 cluster 里两个 CTA 的 shared memory 可以互相访问，mbarrier 也能跨 CTA 到达。5090（sm120）不支持 2-CTA MMA。这类机制更常出现在数据中心 GPU 上，我猜是因为：CTA 之间的互联要占不少晶体管和功耗，数据中心卡跑的基本都是大 GEMM，用这份开销换数据复用（两个 CTA 共享一份 B、等效 tile 加倍）是划算的；消费级卡主要跑游戏，晶体管花在别的地方，这类功能就被砍掉了。

---

## Module 4 完整 GEMM

### prob 4.1 / 4.2 / 4.3

⚠️ **均未在真机验证**（需要 B300；三个程序用 CUDA 13 对 sm_100f 全部编译通过，梯子表的实测行需要占卡后运行 `make run/m4_gemm/01_tiled`、`02_tma`、`03_pipeline` 和 `./sweep_stages.sh` 补上）。实现要点：

- **4.1（01_tiled.cu）**：3.2 扩成 grid = (M/BM, N/BN)，每 block 按 blockIdx 认领 tile；K 循环里每轮把 A/B 的 tile 段用 `swz128` 布局 st.shared，`fence.proxy.async` + syncthreads 后 elected 单线程发 4 条 k16 mma；**整个 K 循环只有第一条 mma 不累加**（`it==0 && kk==0`）；commit 后全体 `mbar_wait(parity = it&1)` 才允许下一轮覆写 smem（parity 随轮次翻转，等错会出现"小 K 侥幸、大 K 必炸"）。
- **4.2（02_tma.cu）**：host 用 `cuTensorMapEncodeTiled` 建 A/B 的 tensor map（dim0=K、box={BK,BM}/{BK,BN}、SWIZZLE_128B——TMA 落进 smem 的布局和手工 swz128 完全一致，descriptor 一字不改）；kernel 里 staging 换成"elected 线程 `mbarrier.arrive.expect_tx` 报满 (BM+BN)*BK*2 字节 + 两条 `cp.async.bulk.tensor.2d`"，单缓冲：`it>0` 先等 empty（上一轮 mma commit 完成）、再发 TMA、等 full、fence、mma、commit 到 empty；TMA 和 tcgen05 都走 async proxy，`fence.proxy.async` 不再需要。
- **4.3（03_pipeline.cu）**：NSTAGE 级循环缓冲，每 stage 一对 mbarrier（full/empty）。预热发 min(NSTAGE, iters) 轮；主循环里**本轮要消费的 TMA 用阻塞等 empty 保证发出**（hazard：用机会式 try_wait 代替强制发射，一旦某轮检查时 stage 未空被跳过，wait full 等的就是一条永远不来的拷贝 → 死锁，1024³ 侥幸、4096³ 必挂），随后**机会式深预取**后续 stage（try_wait 成功就发、失败立刻停）；empty 的 parity 按"该 stage 自己第几次复用"算（`((it/NSTAGE)-1)&1`），full 的 parity 按 `(it/NSTAGE)&1` 算——这两个下标我一开始写混了（用了全局轮次 it 去算 empty 的 parity），后来推状态图时发现 stage 的相位翻转频率是 1/NSTAGE，改成了按 barrier 自己的完成次数计数。
- 4.4（Optional）没做，时间花在了保证 4.1-4.3 的实现完整上。

性能表（4.1-4.3 的实测行、stages 扫描表、流水时空图）留待 B300 补测。概念部分的回答可以先写：

- **4.1 问（此时瓶颈在哪）**：4.1 的结构是"全体线程 st.shared 搬 24KB → fence → 单线程发 mma → 全体等"完全串行的循环，128 个线程的搬运期里 tensor core 闲置，mma 期间搬运又停着。按 0.2 的机器平衡点（234~281 FLOP/byte）衡量，这个 kernel 的实际计算强度远低于平衡点——每个 K tile 只被消费一次、A/B tile 也只在 smem 里复用了一次 mma 组，瓶颈预计在 **staging（数据搬运占用 SM）** 而不是带宽或算力，达成率估计只有 cuBLAS 的百分之几到十几。具体数字要等真机跑出来填表验证。
- 4.2 问（4.1 的 staging 开销构成）：128 线程用普通 `st.shared` 搬 24KB——每线程 96 次读 global + 地址计算 + store smem，期间占用 SM 的 LSU 和线程槽位，且 K 循环里"搬 → 等 → 算"完全串行，mma 干等。换 TMA 后：寻址、打包、swizzle 全部由 TMA 硬件做，SM 只发三条指令，且搬运是 async 的，为 4.3 的重叠创造了条件。
- 4.3 问 (a)：瓶颈从 4.1 的"staging 占用 SM"→ 4.2 的"单缓冲下搬运与计算串行"→ 4.3 的"global 访存带宽/TMA 吞吐"；(b) 梯子逐级：tiling 换来数据复用（算力上得了台面）、TMA 消掉 staging 的指令开销、pipeline 消掉搬运与计算的串行空闲；(c) 继续扩 tile/stage，**shared memory 先顶住**（每 stage (BM+BN)*BK*2，BK/STAGES 一大就超 228KB 上限；TMEM 每 SM 固定 128 lane×512 col×4B，m128n64 的 accumulator 只用 64 列，还有很大余量；除非 tile 的 N 涨到 256 以上 TMEM 才会成为约束）。3.4(c) 的结论在这里接上：cta_group::2 省的 B smem 正好可以换 stage。
- 4.3 第 3 项（流水时空图，以 S=3、K 循环稳态为例，横轴时间、纵轴 stage）：

```
stage0: [TMA it=0][等待][mma it=0][等待]...........[TMA it=3].......
stage1: .......[TMA it=1]....[mma it=1][等待]......[TMA it=4].......
stage2: ............[TMA it=2]....[等待][mma it=2].....[TMA it=5]...
时间 →  t0        t1       t2        t3            t4
```

稳态下每个 stage 的节奏是"TMA 预取(提前 2 拍) → 等 full → mma 消费 → commit 到 empty"，TMA 与 mma 在不同 stage 上错开重叠：当 mma 在消费 stage i 时，stage i+1、i+2 的 TMA 正在飞行中。稳态的吞吐由 TMA 和 mma 中较慢的一方决定（流水填满后两者应接近背靠背）；若把 S 从 3 加到 4，容忍的 TMA 延迟更长，但每 stage 的 smem 代价是 24KB/级。

### prob 4.5 EXPERIMENT：thin GEMM

程序用 cuBLAS bf16，不挑架构——我先算了下显存：满配（M 到 65536）约 3.6 GB，这台 3060 的空闲显存刚好塞得下，于是**全表 63 行都在本机跑完了**（不是 B300 的数字，但 M 轴的规律是通用的）。峰值参数用 25 TFLOPS / 300 GB/s（本机 cuBLAS bf16 的上限量级；个别行 %TC 超过 100% 说明这个参考值略保守，实际峰值 ≈26.5 TFLOPS）。

各层对 Tensor Core 峰值的达成率（%TC，按 M 展开；完整原始输出 63 行见文末附录）：

| layer \ M | 1 | 8 | 16 | 64 | 256 | 1024 | 4096 | 16384 | 65536 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f_b_proj (1536,128) | 0.0 | 0.3 | 1.6 | 6.8 | 22.1 | 60.4 | 83.2 | 86.7 | 85.0 |
| q_b_proj (2304,1536) | 0.9 | 7.6 | 15.5 | 53.3 | 79.0 | 91.9 | 101 | 104 | 97.1 |
| o_proj (7168,1536) | 1.0 | 7.8 | 15.8 | 51.6 | 99.3 | 106 | 106 | 104 | 104 |
| fused_qkv_a (2112,7168) | 1.2 | 7.5 | 15.0 | 59.7 | 89.8 | 96.4 | 100 | 105 | 103 |
| in_proj_qkvgfab (6288,7168) | 1.2 | 8.6 | 16.3 | 48.3 | 93.3 | 96.0 | 107 | 104 | 104 |
| dense_down (7168,8448) | 1.2 | 9.0 | 18.6 | 51.0 | 96.8 | 97.4 | 102 | 101 | 98.1 |
| dense_gate_up (16896,7168) | 1.1 | 9.1 | 16.9 | 67.9 | 85.5 | 93.1 | 98.9 | 94.7 | 94.0 |

M=1 时各层的带宽达成率（%BW）对照：f_b_proj 1.6%、q_b_proj 78.7%、o_proj 85.4%、fused_qkv 96.2%、in_proj 103.8%、dense_down 103.4%、dense_gate_up 95.3%。

(a) 趋势：M=1 时 TC 达成率只有 0~1.2%，而除 f_b_proj 外各层的 %BW 都在 78~104%——纯粹的权重搬运，算力闲着。M=8/16 时 TC 爬到 8~19%、BW 仍在 78~97%（还是带宽吃满）。**从 M=64 开始 TC 明显起飞**（48~68%），M=256 时大 K 的层已到 85~99%；M≥1024 进入平台：除 f_b_proj 外各层稳定在 93~107%（≈cuBLAS 自己的上限）。塌陷区和平台之间过渡很陡，就在 M=16→256 这一段。

(b) M ≤ 16 的形状全是 %BW ≫ %TC，**主要受显存带宽限制**：这时 activation 只有 M×K×2B（几十 KB），权重 N×K×2B（几 MB）是绝对大头，每个权重字节只参与 M 次 FMA，AI ≈ M，远低于机器平衡点。

(c) f_b_proj（K=128）两头都不达标：M=1 时 GB/s 只有 4.8（%BW 1.6%）——权重才 384KB，整个 GEMM 的工作量小到 kernel 启动 + tile 调度的固定开销就把时间吃光了，既摸不到带宽也摸不到算力；大 M 端平台也只到 ~85%，因为 K=128 太浅（只够 2 个 BK=64 的 K 步），每个输出 tile 的 mma 链条太短，prologue/epilogue 占比下不来。它的限制是**形状本身（K 过小）+ 固定开销**，不是两个 roof 里的任何一个。

(d) vLLM 在 M≤16 时弃用 cuBLAS/TC 路径的原因，就是 (a)(b) 的数据：这个区间 TC 达成率不到 20%，时间全花在搬权重上，而 TC 路径还背着 TMA/tile 的 setup；直接用 CUDA Core FMA 的 skinny kernel 不碰这套流水，小矩阵上延迟反而低（上游微基准提升 8%-100%）。

---

## Module 5 低精度与 block scaling

### prob 5.1 EXPERIMENT（quant_outlier.py）

```
含 outlier:
  x≈0.5      rel_err=4.611e-02
  x≈0.1      rel_err=4.634e-02
  x≈0.01     rel_err=3.085e-01
  x≈0.005    rel_err=1.000e+00   （被量化成了 0）
  x≈3000.0   rel_err=0.000e+00
```

(a) 去掉 outlier 重新量化：0.5 处误差 4.611e-2 → 3.086e-4，**好了 149.4 倍**。一个 outlier 把全张量的 scale 抬高了 6.7 倍（3000/448），所有人的有效精度都跟着掉了。
(b) 被量化成 0 的阈值：scale = 3000/448 = 6.696；e4m3 的最小非零幅度是 subnormal 2^-9。当 |x|/scale < 2^-10（最小码点的一半，RN 时舍到 0）即 |x| < scale × 2^-10 ≈ **0.00654** 时量化成 0。关系式：**x_q = 0 ⟺ |x| < scale × min_subnormal(e4m3) / 2**（实测：0.005 → 0，0.008 → 1.3e-2，和推导吻合）。
(c) 1×128 per-block：不含 outlier 的 block 误差 0.5 处 4.6e-2 → 1.4e-3（约 34 倍改善），0.01 处 0.31 → 1.2e-3；**含 outlier 的那个 block 里的邻居们还是遭殃**（0.007 的元素误差 0.75，0.517 的误差 1.3e-2）——outlier 只毒害自己和同 block 的 127 个元素，不再祸害全张量。这就是 block scaling 的意义：把 outlier 的污染半径从"全张量"缩到"一个组"。

### prob 5.2 DERIVE（block_scale_sim.py）

两个函数补全后，`pytest tests/test_block_scale.py` **3 个全过**（fp64 容差内与直接 GEMM 等价；反例函数确实和参考差出一个大数）。

(a) 两行代数式：

- row/col scale（整个点积内常数）：
  `Σ_k (a_k/sA)(b_k/sB) × sA·sB = (sA·sB) × Σ_k (a_k b_k / sA sB) = Σ_k a_k b_k` —— 常数因子可以从和式里提出，最后乘回一次。
- K-block scale（随 k 变）：
  `Σ_j c_j·P_j ≠ c_?·Σ_j P_j`（c_j 是第 j 段的 scale 乘积，P_j 是该段部分和）——不同段的因子不同，提不出来，只能**每段先乘回再累加**：`Σ_j (c_j × P_j)`。

(b) CUTLASS 把 scale 布局写成 M×⌈K/SV⌉（每 scale 覆盖 16/32 个连续 K）：GEMM 内循环沿 K 连续推进、Tensor Core 的 mma 一次消费一段连续 K（k16/k32），硬件沿 K 分段乘 scale 正好和"mma 指令的 K 覆盖 + 连续供数"对齐——scale factor 可以随 K tile 一起流进 tensor core，不需要额外的索引/散布逻辑；如果 scale 沿 M/N 分段，同一段 K 归约里因子会变，硬件就得把归约拆碎，代价大得多。

(c) DeepSeek-V3 的 128×128：scale 在输出通道方向共享、每 128 个 K 一组；NVFP4 每 16 个 K 一组。粒度 16 的优势（结合 5.1(c)）：outlier 的污染半径从 128 个元素缩到 16 个，组内 amax 更接近元素本身，有效精度更高（outlier 密集的 activation 上尤其明显）。代价：scale metadata 数量 ×8（每 16 个元素一个 e4m3，相对每 128），供数路径要同步喂 scale（NVFP4 里 SF 走专用通道/TMEM），带宽和调度复杂度都上升。

### prob 5.3 FROM-SCRATCH：NVFP4 量化通路

(a) **e2m1_encode.h**（host/device 通用）：阈值链实现 RN-even。每个中点的归属我先手推了一遍：0.25→0、0.75→2、1.25→2、1.75→4、2.5→4、3.5→6、5.0→6（"取偶码点"方向是上下交替的），所以边界是 `≤ / <` 交替。写完我在本机用独立参考（float64 按格点对枚举 + 平局取偶）交叉校验了 402,864 个候选值（含全部中点 ±1e-6、饱和区、-0.0），**0 mismatch**——中间还抓到一个真 bug：`v < 0` 判符号会漏掉 **-0.0**（硬件保留符号，应输出 0x8），改成 `signbit(v)` 后完全一致。⚠️ 03a 的硬件判测（`make run/m5_lowprec/03a_encode_check`）需要 sm_100 的 fp4 cvt 指令，编译已通过，等 B300 跑。

(b) **nvfp4_quant_kernel.h**：一线程一组（16 元素），amax → e4m3 SF → swizzled 布局写入 → 设备侧 `__nv_fp4x2_e2m1` 硬件转换打包。`03b_nvfp4_quant`、`test_fp4_gemm`（cuBLASLt 消费验证）都编译通过，⚠️ 判测需 B300（maxrel ≈ 4e-3 的验收口径要真机确认）。

(c) **03c_ceiling_probe.cu**：探针 kernel 与 quant kernel 完全同形（一线程一组、读 16 个 bf16、写 8B 数据 + 1B SF，xor 直通），启动配置也一致。GB/s 对比表 ⚠️ 需真机跑。

(d) Optional 没做（需要 B300 + tcgen05 kind::mxf4nvf4）。

### prob 5.4 FROM-SCRATCH：融合 rms_norm + NVFP4

⚠️ 未在真机验证（编译已通过）。融合 kernel 写法：一个 block 一行，阶段 1 用 float4 向量化读 + shuffle 树形归约求 sumsq → rnorm；阶段 2 一线程一组，逐组"重算 v = x·rnorm·w → amax → SF → 打包"，**不落 bf16 中间结果**。公平基线：两步版各自给独立配置（rms_norm 的 grid/block 在注释里标了要在真机上扫 {256,512,1024}×{M,sms,2sms,4sms} 取最优，quant kernel 用 5.3(b) 的配置），呼应题面"基线吃亏的对比没有意义"的教训。

逐形状加速比表 ⚠️ 待真机补测。预期的分析框架先写好：2.56× 是纯字节账的上限；M 小（decode）时整个 fusced kernel 也只有几微秒，launch/调度固定开销占比大，实测加速比会明显低于 2.56；M 大（prefill）时两步版的第一步写 + 第二步读的 4B/elem 往返才是主要浪费，融合版应接近 ceiling probe（≈2.5 B/elem）的水平，加速比向 2.5× 靠拢但受 SF/数据写的随机化限制。

### prob 5.5 CONCEPT：W4A16+Marlin vs NVFP4 GEMM

(a) W4A16+Marlin 是**存储量化**（权重 int4 只为省显存/带宽，计算时反量化回 fp16 走 fp16 tensor core）；NVFP4 是**计算量化**（数据以 fp4 直接进 tensor core 参与运算）。
(b) W4A16 省显存容量和（decode 时）权重带宽，算力不变；NVFP4 两者都省之外还提升了计算吞吐（fp4 的 TC 峰值是 bf16 的 4 倍）。
(c) 小 batch decode 时 batch×seq 的 activation 很小、权重搬运是绝对大头，所以**存储/带宽量化（W4A16）的收益更直接**——这正是 4.5 里 M=1 时带宽达成率 90%+、算力达成率 1% 的那类形状。

---

## Module 6 TileLang 对照

用 `tilelang_matmul` 的 `T.gemm` kernel（64×64×32、3 stages）分别编译到两个 target，lowering 输出保存在 `m6_lowering/`（`sm_90a_gemm.cu`、`sm_100a_gemm.cu` 和编译脚本 `m6_compile.py`）。编译时踩了个环境坑：tilelang 会调 PATH 里的 nvcc，系统的 12.6 不认识 sm_100a，把 CUDA 13 放进 PATH 就好了。

| | sm_90a | sm_100a |
| --- | --- | --- |
| 选中的 Tensor Core 指令 | **wgmma**（`instruction/wgmma.h`、wgmma_ss、warpgroup 路径） | **mma.sync**（`instruction/mma.h` + `mma_sync` + ldsm）——0.1.13 还没生成 tcgen05，走 sm80 风格回退 |
| descriptor 在哪里、由谁生成 | host 侧生成 CUtensorMap（TMA 用）+ wgmma 的 smem descriptor 由 TileLang 的模板在 kernel 里构造 | 无 wgmma descriptor；ldmatrix 地址由模板生成（mma.sync 不需要 descriptor） |
| smem swizzle 布局在哪一步确定 | layout 推断阶段（tilelang 的 layout pass），落成 128B swizzle 的 TMA/descriptor 参数 | 同样在 layout 推断，落成 ldmatrix 需要的行布局 |
| 数据由谁搬入 smem | **TMA**（`tma_load` + mbarrier，host 建 tensor map） | 普通 ld.global → st.shared（拷贝模板） |

对照 M2-M4 的手写实现回答：

(a) DSL 自动完成的硬件决策：选哪代 Tensor Core 指令、smem 的 swizzle 布局、descriptor/tensor map 的生成、搬运走 TMA 还是普通 load、mbarrier 的插入——这些在 M2-M4 里全是我手推手写的。
(b) 仍然要程序员决定的：tile 尺寸（BM/BN/BK）、num_stages、threads——必须靠实测调的旋钮。
(c) 在 assignment01 7.5 的表里补一行：**"Tensor Core 指令选择与供数布局"：CUDA SIMT = 用户，cuTile = 编译器，Triton = 编译器，TileLang = 编译器**。

另一个观察（和讲义的"实现成熟度要分开评估"呼应）：手里的 tilelang 0.1.12/0.1.13 编到 sm_100a 时还只会退回 mma.sync 这条老路，说明它还没跟上新卡的指令——想用好 sm_100 的新特性，M2-M4 那种手写路径暂时还绕不开。

---

## 团队选做（C1/C2）

没有做。原因：团队题要求 2-4 人组队 + 10 分钟答辩，而且测量基本都依赖 B300/H20 一类的卡（FlashKDA 的 SM100 迁移评估、MSA decode 的 vLLM 复现），我一个人一台 6GB 的 3060 满足不了实验条件。If 有集群账号的话这两题倒是很好的练手材料（TASK.md 和 harness 都给得很全）。

---

## 提交清单

- **代码**：assignment02/cuda 下所有题目文件已实现/修复（host 题判测全 PASS；B300-only 题编译通过、待真机判测）。
- **Python**：`pytest tests/` 3 passed（5.2）；`kernels/quant_outlier.py` 输出见 5.1。
- **真机数据**：本报告所有未标 ⚠️ 的表格都来自我自己的 RTX 3060 Laptop（WSL2，CUDA 12.6 / 13.0.88）实测。
- **待补**（需要 B300）：1.3/1.4 判测、3.2/3.3/3.4 判测与 ncu、4.1-4.3 梯子表与 stages 扫描、5.3(a)(b)(c) 硬件判测、5.4 逐形状计时。（4.5 的全形状表已在 3060 上完整跑完，见正文与附录；若要与 B300 的绝对值对标，再在 B300 上重跑同一程序即可。）

---

## 附录：4.5 thin GEMM 完整原始输出（RTX 3060 Laptop，峰值参考 25 TFLOPS / 300 GB/s）

```
f_b_proj                 1   1536    128      82.9       0.0       4.8     1.0     0.0%     1.6%
f_b_proj                 8   1536    128      40.8       0.1      10.3     7.5     0.3%     3.4%
f_b_proj                16   1536    128      15.7       0.4      28.4    14.1     1.6%     9.5%
f_b_proj                64   1536    128      14.8       1.7      41.1    41.5     6.8%    13.7%
f_b_proj               256   1536    128      18.2       5.5      68.4    80.8    22.1%    22.8%
f_b_proj              1024   1536    128      26.7      15.1     142.5   105.9    60.4%    47.5%
f_b_proj              4096   1536    128      77.4      20.8     181.2   114.8    83.2%    60.4%
f_b_proj             16384   1536    128     297.4      21.7     184.7   117.3    86.7%    61.6%
f_b_proj             65536   1536    128    1212.2      21.3     180.2   117.9    85.0%    60.1%
q_b_proj                 1   2304   1536      30.0       0.2     236.2     1.0     0.9%    78.7%
q_b_proj                 8   2304   1536      29.6       1.9     241.1     7.9     7.6%    80.4%
q_b_proj                16   2304   1536      29.2       3.9     246.6    15.7    15.5%    82.2%
q_b_proj                64   2304   1536      34.0      13.3     222.8    59.8    53.3%    74.3%
q_b_proj               256   2304   1536      91.8      19.7      98.5   200.3    79.0%    32.8%
q_b_proj              1024   2304   1536     315.5      23.0      47.4   485.1    91.9%    15.8%
q_b_proj              4096   2304   1536    1148.0      25.3      33.6   752.3   101.0%    11.2%
q_b_proj             16384   2304   1536    4457.4      26.0      29.8   872.5   104.1%     9.9%
q_b_proj             65536   2304   1536   19111.8      24.3      26.7   908.8    97.1%     8.9%
o_proj                   1   7168   1536      86.0       0.3     256.1     1.0     1.0%    85.4%
o_proj                   8   7168   1536      90.7       1.9     244.3     7.9     7.8%    81.4%
o_proj                  16   7168   1536      89.3       3.9     249.8    15.8    15.8%    83.3%
o_proj                  64   7168   1536     109.3      12.9     211.7    60.9    51.6%    70.6%
o_proj                 256   7168   1536     227.1      24.8     116.6   212.9    99.3%    38.9%
o_proj                1024   7168   1536     852.7      26.4      46.7   565.9   105.8%    15.6%
o_proj                4096   7168   1536    3420.2      26.4      27.3   966.5   105.5%     9.1%
o_proj               16384   7168   1536   13838.0      26.1      22.2  1174.3   104.3%     7.4%
o_proj               65536   7168   1536   55553.0      26.0      20.9  1241.0   103.9%     7.0%
fused_qkv_a_proj         1   2112   7168     105.0       0.3     288.5     1.0     1.2%    96.2%
fused_qkv_a_proj         8   2112   7168     128.5       1.9     236.8     8.0     7.5%    78.9%
fused_qkv_a_proj        16   2112   7168     129.1       3.8     236.8    15.8    15.0%    78.9%
fused_qkv_a_proj        64   2112   7168     129.8      14.9     242.5    61.6    59.7%    80.8%
fused_qkv_a_proj       256   2112   7168     345.2      22.5     101.5   221.3    89.8%    33.8%
fused_qkv_a_proj      1024   2112   7168    1286.1      24.1      38.3   629.1    96.4%    12.8%
fused_qkv_a_proj      4096   2112   7168    4960.8      25.0      21.4  1166.7   100.0%     7.1%
fused_qkv_a_proj     16384   2112   7168   18851.6      26.3      17.7  1483.6   105.3%     5.9%
fused_qkv_a_proj     65536   2112   7168   76801.1      25.8      16.2  1591.7   103.3%     5.4%
in_proj_qkvgfab          1   6288   7168     289.6       0.3     311.3     1.0     1.2%   103.8%
in_proj_qkvgfab          8   6288   7168     337.1       2.1     268.1     8.0     8.6%    89.4%
in_proj_qkvgfab         16   6288   7168     353.5       4.1     256.2    15.9    16.3%    85.4%
in_proj_qkvgfab         64   6288   7168     478.1      12.1     192.2    62.8    48.3%    64.1%
in_proj_qkvgfab        256   6288   7168     989.7      23.3      98.0   237.8    93.3%    32.7%
in_proj_qkvgfab       1024   6288   7168    3846.7      24.0      30.6   784.2    96.0%    10.2%
in_proj_qkvgfab       4096   6288   7168   13844.0      26.7      14.5  1842.7   106.7%     4.8%
in_proj_qkvgfab      16384   6288   7168   56692.1      26.1       9.4  2781.0   104.2%     3.1%
in_proj_qkvgfab      65536   6288   7168  227949.8      25.9       8.1  3186.7   103.7%     2.7%
dense_down_proj          1   7168   8448     390.7       0.3     310.1     1.0     1.2%   103.4%
dense_down_proj          8   7168   8448     428.7       2.3     283.1     8.0     9.0%    94.4%
dense_down_proj         16   7168   8448     417.4       4.6     291.4    15.9    18.6%    97.1%
dense_down_proj         64   7168   8448     608.3      12.7     202.4    63.0    51.0%    67.5%
dense_down_proj        256   7168   8448    1281.3      24.2     100.8   240.1    96.8%    33.6%
dense_down_proj       1024   7168   8448    5095.6      24.3      30.0   810.1    97.4%    10.0%
dense_down_proj       4096   7168   8448   19487.6      25.5      12.8  1991.9   101.8%     4.3%
dense_down_proj      16384   7168   8448   78945.0      25.1       8.0  3135.6   100.5%     2.7%
dense_down_proj      65536   7168   8448  323512.5      24.5       6.7  3661.1    98.1%     2.2%
dense_gate_up_proj       1  16896   7168     847.2       0.3     286.0     1.0     1.1%    95.3%
dense_gate_up_proj       8  16896   7168     853.8       2.3     284.1     8.0     9.1%    94.7%
dense_gate_up_proj      16  16896   7168     916.6       4.2     265.1    15.9    16.9%    88.4%
dense_gate_up_proj      64  16896   7168     912.8      17.0     268.7    63.2    67.9%    89.6%
dense_gate_up_proj     256  16896   7168    2899.4      21.4      87.8   243.6    85.5%    29.3%
dense_gate_up_proj    1024  16896   7168   10657.0      23.3      27.4   850.9    93.1%     9.1%
dense_gate_up_proj    4096  16896   7168   40136.3      24.7      10.9  2258.2    98.9%     3.6%
dense_gate_up_proj   16384  16896   7168  167608.4      23.7       6.1  3850.2    94.7%     2.0%
dense_gate_up_proj   65536  16896   7168  675442.6      23.5       5.0  4673.9    94.0%     1.7%
```
