# 作业 1 报告（GPU & GPU Programming）

> 我的机器：WSL2 里的 NVIDIA GeForce RTX 3060 Laptop GPU（6GB），CUDA Toolkit 12.6，编译用默认 `ARCH=native`（sm_86）。
> 说明一下：这份作业的代码题我都是对着 PTX/CUDA 文档和讲义一点点写的，数据全是自己机器上跑出来的，所以我只敢对这块 3060 的数字负责，别的卡上数字会不一样。

## 判测与运行结果汇总

| 类别 | 判测内容 | 命令 | 结果 |
| --- | --- | --- | --- |
| CUDA | 模块 0-5 共 16 个练习 | `make run/...`（README 的方式逐个跑） | 全部 **PASS**（详见各节） |
| CUDA 压轴 | 2.9 SAXPY 对拍（7 个 n） | `./judge_saxpy.sh saxpy.cu` | **7/7 全部通过** |
| CUDA 压轴 | 3.5 归约两版 + 选做 shuffle 版 | `make run/m3_simt/03_reduce` | 全部 PASS，比值 1.56x |
| Python | 模块 1/7 全部测试 | `uv run pytest tests/` | **24 passed** |
| Bonus | naive matmul 三档 BS | `./bin/bonus/matmul_bs{8,16,32}` | 三档都 PASS，GFLOPS 见 Bonus 节 |

---

## Module 0 环境准备

### prob 0.1 HANDS-ON

`make run/m0_env/01_hello` 编译运行成功。4 个 block、每个 8 个线程都打印了。我连跑 5 次记录 block 顺序，5 次全部是 `1 2 0 3`——不是 launch 的编号顺序（说明顺序确实由硬件调度决定，CUDA 不承诺按 0,1,2,3 来），但在这台机器/驱动状态下恰好是稳定的。对比 2.8 的 whoami（每 block 只打一行）次次乱序，可见顺序既无保证也无规律可依赖。

### prob 0.2 FILL-IN

五个空都是 `cudaDeviceProp` 的字段，我查了 Runtime API 文档填的：`multiProcessorCount`、`warpSize`、`sharedMemPerBlock`、`maxThreadsPerMultiProcessor`、`totalGlobalMem`。

运行结果，对照 Guide 附录 Compute Capabilities 里 Ampere（8.6）那栏核对过：

| 项目 | 我的卡（实测） |
| --- | --- |
| 型号 / compute capability | NVIDIA GeForce RTX 3060 Laptop GPU / 8.6 |
| SM 数量 | 30 |
| warp 大小 | 32 |
| shared memory / block | 49152 B（48 KB） |
| 最大常驻线程 / SM | 1536 |
| 显存总量 | 6441926656 B（约 6 GB） |

核对：8.6 每 SM 最大线程数 1536 ✓、warp 32 ✓。这张表后面 Module 3/4 反复用到，我确实一直翻回来查。

---

## Module 1 为什么要用 GPU

### prob 1.1 CONCEPT

- (a) **错**。100 TFLOPS 是吞吐（每秒能干多少活），不是单条指令的延迟。GPU 单条指令延迟其实和 CPU 差不多甚至更长（几百个 cycle），它快是靠同时几千个线程把延迟"藏"起来。
- (b) **对**。HBM 的标称带宽是连续大块访问的理想值，随机零散访问打不满，因为每次访问的有效字节占比低、还吃 cache miss。
- (c) **对**。每步依赖上一步，就是一条串行依赖链，GPU 再多核也并行不起来（这就是 1.2 要展开的事）。
- (d) **错**。1000 TFLOPS 是"每秒 10^15 次运算"的吞吐，前提是有足够多的运算并行在飞。单次运算的延迟远大于 10^-15 秒，两者是吞吐和延迟两个概念。

### prob 1.2 CONCEPT

因为"总计算量 10^12 FLOP"完全可以并行摊给几万个核心，所以是毫秒级；但严格在线的串行算法每一步都要等上一步的结果，任意时刻只有一个运算真正在推进，能利用的就只有一个核心的计算能力（连 CPU 单核都打不满流水线）。吞吐再高，绕不过依赖链——能并行的是依赖"图"的宽度，不是总量。

### prob 1.3 CONCEPT（补全表）

| 执行层次 | 软件含义 | 对应硬件 | 直接可用的存储 | 同步与通信手段 |
| --- | --- | --- | --- | --- |
| thread | kernel 的最小执行单位 | 计算单元上的一个 lane | 自己的寄存器 | （自身天然有序） |
| warp | 32 个 thread 的执行单位，一起取指发射 | 一个 SM 子单元（32 lane 共用一套取指/发射） | 寄存器（不够用会溢出到 local） | shuffle 指令（warp 内直接换数据） |
| block / CTA | 逻辑上的一组线程，可一维/二维/三维 | 驻留在同一个 SM 上的一批 warp | 寄存器 + shared memory | `__syncthreads()`、shared memory、原子操作 |
| grid | 一次 kernel 启动的所有 block | 整颗 GPU（block 被分发到各 SM） | global memory 对所有 block 可见 | 原子操作；没有原生的全 grid 同步，一般靠拆成两次 kernel |

### prob 1.4 CONCEPT

SIMD：一条指令对所有 lane 操作，控制流单一，分支要靠掩码。SIMT：每个线程有自己的寄存器、能写"单线程风格"的程序，硬件按 warp 成组执行，分支时 warp 内拆成两条路径分别执行（带掩码）。

判断题：**错**。Volta 之后每线程有独立 PC 解决的是"正确性"（不再要求程序员手工保证 warp 内对齐），但同一个 warp 走不同分支时还是要把两条路径各执行一遍，divergence 的性能代价还在，只是形式变了。

### prob 1.5 EXPERIMENT（01_scaling.cu 实测，RTX 3060 Laptop）

| 配置 | 耗时 (ms) | ns / 元素 |
| --- | --- | --- |
| CPU 单线程 | 14.548 | 3.47 |
| GPU `<<<1,1>>>` | 321.422 | 76.63 |
| GPU `<<<1,256>>>` | 6.124 | 1.46 |
| GPU 铺满 grid | 0.348 | 0.08（16384 blocks × 256 threads） |

(a) GPU 单线程比 CPU 还慢 20 多倍：因为这条路径上 GPU 只有一个核心在干活，还叠加了显存访问延迟没有别的东西来遮掩；CPU 单线程有 cache、预取、乱序这些为延迟优化的机制。这直接验证了"GPU 快不是核心快，是多"。

(b) 从单 block 到铺满 grid 的 900 倍提速说明：GPU 的加速来自**海量线程并行**，把访存延迟用别的 warp 的计算盖住（延迟隐藏）。并行度不够时，GPU 就是一颗很慢的小 CPU。

### prob 1.6 FROM-SCRATCH（选做）：simt_sim.py

按 docstring 的 contract 写了递归模拟器：`exec_block(prog, active)` 按 mask 执行，`if_lt` 把当前 active 按条件切成两半分别递归、空支跳过不计拍、`if_lt` 本身不计拍。`pytest tests/test_simt_sim.py` 5 个用例全过（嵌套分支、单边跳过、汇合后全体执行都覆盖了）。写完最大的感受是：divergence 的代价真的就是"两支各算一遍"，模型很朴素。

---

## Module 2 第一个 CUDA 程序

### prob 2.1 FILL-IN（01_vector_add.cu）

六个空：`__global__`、`blockIdx.x * blockDim.x + threadIdx.x`、`idx < n`、`cudaMemcpyHostToDevice`（两处）、`(n + threadsPerBlock - 1) / threadsPerBlock`、`<<<blocksPerGrid, threadsPerBlock>>>`。运行输出 PASS。

### prob 2.2 CONCEPT

- (a) GPU 上执行、CPU 启动 → `__global__`
- (b) 只被 kernel 调用的辅助函数 → `__device__`
- (c) host 和 device 都要调 → `__host__ __device__`
- (d) 全 kernel 不变、所有线程都读的系数表 → `__constant__`（constant memory）
- (e) block 内共享的暂存数组 → `__shared__`

### prob 2.3 MODIFY（02_vector_add_um.cu）

按原样先跑了一次基准：**显式管理版 153.3 ms**。然后改成 `cudaMallocManaged`，删掉所有 `cudaMemcpy`，在 CPU 读结果前加了 `cudaDeviceSynchronize()`。改完 PASS，**48.4 ms**。

(a) kernel 启动是异步的，host 不等它做完就往下走；下一行 CPU 要读 `c` 的内容，必须等 GPU 写完，所以要显式同步。原版里这次同步藏在 `cudaMemcpy`（D2H）里——它本身是个同步调用，会顺带等前面的 kernel。

(b) 对比：153.3 ms → 48.4 ms，unified memory 反而快了 3 倍多。我一开始觉得反常识，后来想了个解释（不确定对不对）：显式版要在计时窗口里做 4 次大块拷贝（h2d 两次 + d2h 一次 + kernel 读写），每次都要把 64MB 数据在 host 内存和显存之间整个搬一遍；UM 版没有显式拷贝，kernel 直接在托管页上算，缺哪页搬哪页，可能省掉了一些整块的来回。题面说"谁快谁慢都有可能，与使用的卡有关"，我这块笔记本卡的实际情况就是 UM 明显快。

### prob 2.4 CONCEPT

- (a) **错**。`vectorAdd<<<...>>>(...)` 只是提交，返回时 kernel 一般还没跑完（异步）。
- (b) **对**。同一个 stream 内操作按序执行，D2H 的 `cudaMemcpy` 会等前面的 kernel 完成才开始。
- (c) **错**。非法访存是异步报的，错误会挂在**之后**任何一个同步点（比如下一次 `cudaMemcpy`/`cudaDeviceSynchronize`）上冒出来，不会在启动语句处同步报。

### prob 2.5 DEBUG（03_bug_launch.cu）

按提示在 launch 后面补了 `CUDA_CHECK_KERNEL()`，用带 bug 的原版（threads=2048 + 补上的检查）复跑，抓到了现场：

```
CUDA error cudaErrorInvalidConfiguration at 03_bug_launch.cu:36: invalid configuration argument
```

原因：`threads = 2048` 超过了每 block 最大线程数（prob 0.2 查过的 1024），整个 launch 直接无效。为什么不加检查就一声不吭？因为 kernel launch 语句没有返回值，错误只记在 CUDA 的错误槽里，没人去查它就一直躺着。修法：threads 改成 1024，并把错误检查留在原地。修完 PASS。

### prob 2.6 FILL-IN（04_matrix_add.cu）

四个空：`blockIdx.y * blockDim.y + threadIdx.y`、`blockIdx.x * blockDim.x + threadIdx.x`、`row < M && col < N`、`dim3((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y)`。PASS。

### prob 2.7 MODIFY（05_grid_stride.cu）

把 kernel 改成 grid-stride loop（起点 = 全局线程号，步长 = `blockDim.x * gridDim.x`），launch 不动，PASS。

价值：kernel 和问题规模解耦了——launch 给多给少都能算对，还留了调优空间（线程数是独立旋钮）。代价：16384 个线程要轮流处理 16M 个元素，每个线程循环 1024 次，线程数远低于能铺满 GPU 的量，延迟隐藏和访存并行度都不足，性能比铺满 grid 的版本差很多（1.5 里铺满是 0.348 ms 量级）。

### prob 2.8 EXPERIMENT（06_whoami.cu）

跑了三次，16 个 block 的报到顺序每次都不一样，比如一次是 `2, 11, 1, 14, 10, 8, 13, 5, 7, 4, 0, 9, 15, 3, 12, 6`，另一次是 `1, 10, 13, 2, 7, ...`，大致有按编号上升的趋势但完全是乱序穿插的。

(a) 顺序由硬件调度器决定（block 分发到空闲 SM 的时机），CUDA 不承诺任何顺序。
(b) 不可以依赖。Guide 1.1 说的 scalable programming model 就是这个意思：程序的正确性不能假设"有多少个 SM、谁先跑"，这样同一个程序才能在从最小的卡到最大的卡上都正确——block 之间互相独立、不依赖执行顺序，才可扩展。

### prob 2.9 FROM-SCRATCH：SAXPY（saxpy.cu）

自己写的完整程序：错误检查宏（`CUDA_CHECK` / `CUDA_CHECK_KERNEL`）和 `cudaEvent` 计时都照着 common.h 的思路手写了一遍（没 include 它），数据按公式生成，n=0 特判（0 个 block 的 launch 非法）。

判测结果（`./judge_saxpy.sh saxpy.cu`）：

```
n=0  PASS  (SUM=0)
n=1  PASS  (SUM=-1536)
n=31  PASS  (SUM=-46686)
n=1024  PASS  (SUM=-525312)
n=1025  PASS  (SUM=-525824)
n=1048576  PASS  (SUM=-1048576)
n=1048579  PASS  (SUM=-1053178)
全部通过
```

踩的坑：一开始没注意数据都是精确的半整数/整数，担心浮点误差，后来想明白了 float 下这些值全都精确表示，double 累加没有舍入问题。

---

## Module 3 SIMT 执行

### prob 3.1 CONCEPT

blockDim = (8, 8, 1)：
- (a) `threadIdx=(3,5,0)` 线性编号 = 5×8+3 = **43**。warp = 43/32 = **第 1 个 warp**，lane = 43%32 = **11**。
- (b) 64 个线程 = **2 个 warp**。
- (c) blockDim=(33,1,1) 占 **2 个 warp**。浪费：第二个 warp 只有 1 个活跃线程，另外 31 个 lane 白占调度槽位。

### prob 3.2 EXPERIMENT（01_divergence.cu 实测）

先写的预测：按奇偶分的更慢，因为一个 warp 里一半走一支；我猜大概慢 1 倍左右。实测：

```
warp 内分支 (tid % 2)    :  2.814 ms
按 warp 分支 (tid/32 % 2):  1.191 ms
比值: 2.36
```

比预测略大。解释：奇偶版每个 warp 都要串行执行两条分支路径，总指令数×2；按 warp 分的版本每个 warp 只走一条路。两分支计算量相同时，理论上限就是 2 倍差，实测 2.36（还有点额外开销）。

若两个分支计算量一大一小：奇偶分的版本，每个 warp 都要既走大支又走小支，时间由"每 warp 两支之和"决定 = (大+小)×warp 数；按 warp 分的版本，一半 warp 只走大支、一半只走小支，总时间由**大支**的那批 warp 决定（小的那批先干完闲着）。所以前者由两支之和决定，后者由较大那支决定。

### prob 3.3 EXPERIMENT（02_sync_matters.cu）

1. 原样跑：PASS。
2. 注释掉 `__syncthreads()` 后连跑三次，全是 MISMATCH，而且每次第一个错的位置都不一样：一次在 index 0，一次在 64，一次在 32（都是 32 的倍数，正好是 warp 边界）。恢复后 PASS。

(a) 没有同步，读 `buf[BLOCK-1-t]` 的线程不等写它的线程（t'=255-t，在另一个 warp），可能读到 shared memory 里还没被写入的旧值，倒序结果就错了。

(b)（选做）有些位置一直是对的：如果写者所在 warp 恰好在读者之前执行完了。我算了一下：t 在 warp w，255-t 所在的 warp 恰好是 7-w（t∈[32w, 32w+31] ⇒ 255-t∈[224-32w, 255-32w]，整除落进 warp 7-w），两者永远不会相等；w=3 时是 warp 3 和 warp 4，相邻但也不相同。所以任何位置的正确性都取决于 8 个 warp 的实际发射顺序——硬件大致按 0..7 顺序发 warp 时，warp 4-7 读的值由 warp 3-0 写（已写完，对），warp 0-3 读的值由 warp 7-4 写（还没写，错）。这就解释了为什么错的总是低编号 warp 的位置、看起来随机但又不是全错。

### prob 3.4 CONCEPT

标准做法：**把 kernel 拆成两次 launch**（kernel 边界就是隐式的全 grid 同步），或者用 cooperative groups 的 grid sync（需要 cooperative launch 支持，block 数受限制）。最常用还是拆 kernel。

### prob 3.5 FROM-SCRATCH：block 内归约（03_reduce.cu）

按 contract 实现了两个版本（交错配对：`tid % (2s) == 0` 干活；连续配对：`tid < s` 干活），都用了 `__shared__ float buf[BLOCK]`，循环边界用 `blockDim.x`。选做的 shuffle 版也写了（warp 内 `__shfl_down_sync` 归约 → 8 个 warp 部分和落 shared → 第 0 个 warp 再归约一次）。

实测：

```
interleaved: PASS  平均 0.0865 ms
contiguous : PASS  平均 0.0554 ms
interleaved / contiguous = 1.56x
shuffle(warp) : PASS  平均 0.0327 ms
```

解释：两版加法次数完全一样，差别只在活跃线程在 warp 里的分布。交错版活跃线程隔一个一个，每轮每个 warp 都有两支路径要串行发（divergence）；连续版活跃线程挤在低编号一头，大部分轮次整个 warp 同进同出或者干脆整 warp 空闲，分支代价小。shuffle 版直接绕开了 shared memory 往返和 `__syncthreads`，最快。这题给我的震撼：加法次数一样，耗时差 1.5 倍以上——算法课的"复杂度"在 GPU 上远远不够描述性能。

---

## Module 4 存储空间

### prob 4.1 CONCEPT（补全表）

| 空间 | 谁可见 | 生命周期 | 片上/片外 | 谁管理 |
| --- | --- | --- | --- | --- |
| register | 单个线程 | 线程 | 片上 | 编译器 |
| local | 单个线程（"local"指私有） | 线程 | 片外（显存里，走 L1/L2 缓存） | 编译器（spill / 动态索引时放这） |
| shared | block 内所有线程 | block | 片上 | 程序员（声明、`__syncthreads` 配合） |
| global | 所有线程 + host | 程序/显式控制 | 片外 | 程序员（cudaMalloc/Free） |
| constant | 所有线程（只读） | 程序 | 片外存 + 片上专用 cache | 程序员（cudaMemcpyToSymbol） |
| L1 / L2 cache | L1: SM 内；L2: 全卡 | 自动（硬件） | 片上 | 硬件自动 |

### prob 4.2 FILL-IN（01_stencil.cu）

五个空：`BLOCK + 2 * RADIUS`、`__syncthreads();`、`(tile[l-1] + tile[l] + tile[l+1]) / 3.f`、`extern __shared__ float tile[];`、launch 第三参数 `(BLOCK + 2 * RADIUS) * sizeof(float)`。static/dynamic 两版都 PASS。

### prob 4.3 MODIFY（02_constant_coeff.cu）

声明 `__constant__ float COEF[8]`，`cudaMemcpyToSymbol` 拷入，kernel 改读 `COEF`（参数表保留但不用）。两版 PASS，实测：

```
global  : PASS  平均 0.5480 ms
constant: PASS  平均 0.5433 ms
global / constant = 1.01x
```

差距小到基本测不出来。我猜原因：8 个系数太小了，早就被 cache 装下了，global 版的读也几乎全部命中缓存，所以这点差别根本影响不了总时间（这是个刻意安排的"负结果"，说明优化要用在真正的瓶颈上）。constant cache 真正的优势在：一个 warp 里 32 个线程读**同一个地址**时，硬件一次广播给全 warp，不用重复取；如果 warp 内各线程读的是不同地址（比如每线程查自己下标对应的表项），constant cache 反而帮不上忙。

### prob 4.4 CONCEPT

- (a) **对**。local 是"线程私有"的意思，物理位置在片外显存（带 caching）。
- (b) **对**。编译期无法把动态下标展开成寄存器号，数组只能放进 local memory。

### prob 4.5 FILL-IN（03_histogram.cu）

空：`atomicAdd(&hist[v], 1u);`。PASS，实测平均 5.9025 ms（2.84 GB/s）。

### prob 4.6 MODIFY（04_histogram_priv.cu）

shared 私有化版：每 block 声明 `__shared__ unsigned int local[BINS]` 清零 → 数据先 atomicAdd 进自己的 private 副本 → 同步后再 256 次 atomicAdd 汇入全局。两版 PASS，实测：

```
naive: PASS  平均 5.9565 ms  (2.82 GB/s)
priv : PASS  平均 0.1414 ms  (118.62 GB/s)
naive / priv = 42.12x
```

提速来源：16M 次全局原子冲突被压缩到 block 内片上原子（无 DRAM 往返、冲突域从全卡缩到 256 个线程），最后每 block 只做 256 次全局原子（1024 个 block 总共 26 万次，比 16M 次少 60 倍）。瓶颈从"全局计数器串行化"变成了纯数据搬运。

### prob 4.7 EXPERIMENT（05_bandwidth.cu 实测）

| stride | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| GB/s | 251.4 | 167.7 | 100.6 | 55.8 | 56.3 | 56.5 |

趋势：stride 翻倍带宽大致减半，到 stride≥8 后跌到 56 GB/s 左右的平台就不再掉了。原因：stride=1 时一个 warp 的 32 次读合并成 8 条 128B 的访存事务（全利用）；stride 变大后，相邻线程的地址落进不同的事务，每次事务里有效的 4 字节只占 4B/32B，有效带宽按比例掉；stride≥8 之后每个线程读的地址都落在不同的 128B 段（我的 16MB 数组、32 位下标转一圈），事务数饱和，就停在这个平台上了。

### prob 4.8 EXPERIMENT（06_occupancy.cu 实测）

程序开头打印：`shared memory 100 KB / SM，最大常驻 1536 线程 / SM`。

| shared memory / block (KB) | 0.0 | 13.2 | 15.0 | 18.0 | 29.0 | 55.0 |
| --- | --- | --- | --- | --- | --- | --- |
| 理论驻留 block / SM | 6 | 6 | 6 | 5 | 3 | 1 |
| occupancy | 100.0% | 100.0% | 100.0% | 83.3% | 50.0% | 16.7% |
| 实测带宽 (GB/s) | 255.5 | 250.7 | 249.2 | 253.3 | 247.3 | 142.2 |

（`cudaOccupancyMaxPotentialBlockSize` 建议 blockSize = 768。）

(a) 手算：100 KB/SM ÷ 29 KB/block = 3.44 → 取 3 个 block（还受每 block 上限约束），3×256/1536 = 50% ✓，和 API 报的一致。55 KB 档：100/55 = 1.8 → 1 个 block → 16.7% ✓。

(b) 带宽靠足够多的常驻 warp 把访存延迟"泡"在并发行列里（Little 定律：在途请求数 = 吞吐 × 延迟）。occupancy 掉到 16.7% 时每 SM 只有 8 个 warp，能同时在途的请求不够覆盖延迟，带宽就塌了。

(c) 100%→83.3%（少 1 个 block）带宽几乎没掉（255.5→253.3），83.3%→50% 也没掉多少，但 50%→16.7% 直接掉了约 43%。说明带宽和 occupancy 不是正比关系：只要常驻 warp 数够"泡满"内存的并发度，再少一点也无妨；一旦低于某个阈值（这道题里在 8 个 warp 附近），在途请求严重不足，带宽才崩。换句话说 occupancy 是余量，不是线性资源。

---

## Module 5 计时与异步初步

### prob 5.1 EXPERIMENT（01_timing_trap.cu 实测）

```
host 计时、不等 GPU :  0.0438 ms
host 计时、等 GPU   :  1.0616 ms
cudaEvent 计时      :  0.5878 ms
```

(a) 能写进报告的是 **cudaEvent 计时**（0.5878 ms）——它在 GPU 时间线上打点，量的是 kernel 真正的执行时长。
(b) 0.0438 ms 只是 kernel 启动调用（提交）的 host 侧开销，GPU 根本还没干完活；1.0616 ms 是"提交 + 等 GPU 干完 + host 被唤醒回来"的全程，比 kernel 本身多了同步和唤醒延迟（所以比 event 的数还大）。

### prob 5.2 CONCEPT

- (a) **对**。stream 内按提交顺序执行。
- (b) **对**。这就是异步语义：提交完立刻返回。
- (c) **对**。UM 的按页迁移机制：CPU 碰到正在被 GPU 使用的页会触发 page fault，驱动把页迁过来。

---

## Module 6 Tile 视角

### prob 6.1 CONCEPT

- (a) **错**。tile 是编程模型里的逻辑分块，不是显存里某块可变区域，也不是用指针直接改写。
- (b) **对**。编译器把"对 tile 的一个操作"映射成 block 内多线程（这正是 tile 模型的核心）。
- (c) **错**。两者可以在同一个程序里混用（Guide 也把它们并列写成两种模型）。

### prob 6.2 CONCEPT（补全表）

|  | CUDA SIMT | cuTile | Triton |
| --- | --- | --- | --- |
| 并行单位 | block 里的 thread | block | program（一个 block） |
| 编号 | `blockIdx` / `threadIdx` | `ct.bid(0)` | `tl.program_id(0)` |
| 数据分工 | 线程用全局下标来划分数据 | tiled_view 按 tile 划分，block 领一个 tile | `pid * BLOCK_SIZE + tl.arange(...)` 划分一段，block 内"整体"处理 |
| 边界处理 | `if` 判断 | tiled_view 负责不越界 | mask（`offsets < n` 的掩码） |

（Triton 一列是我做完 Module 7 回来填的，`vector_add.py` 就是照这个思路写的。）

### prob 6.3 CONCEPT

(a) 由**编译器**决定——用户只说"对这个 tile 做 +"，tile 到线程的映射是编译器推的。
(b) CUDA SIMT 版里一定会出现、这里没有的：`threadIdx`/`blockIdx` 的索引计算、`if (idx < n)` 边界判断、`__global__` 修饰符、launch 配置（`<<<blocks, threads>>>`）里"每 block 多少线程"的选择、以及 kernel 内的显式循环/掩码。

---

## Module 7 TileLang 与 Triton

### prob 7.1 FILL-IN（vector_add.py）

四个空：`tl.program_id(0)`、`pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`、`offsets < n`、`tl.store(z_ptr + offsets, x + y, mask=mask)`。测试过（conftest 在没有 GPU 的机器上会自动切到 interpreter 模式；我的机器有卡，所以是真正的 GPU 在跑）。

### prob 7.2 MODIFY（fused_op.py）

把计算行改成 `z = tl.maximum(a * x + b, 0.0)`，给 kernel 和 wrapper 加了 `a, b` 两个参数。`pytest tests/test_fused_op.py` 过。

回答：改动全部集中在 kernel 里"算数"的那一行 + 参数表，索引、mask、load/store 主体一行没动。因为 Tile/Triton 把"数据怎么划分、怎么搬"固化在骨架里，换个逐元素公式只是换操作数表达式——这正是 6.2 表格说的"数据分工和边界处理都交给模型"的好处。

### prob 7.3 FILL-IN（tilelang_scale_add.py）

两个空：`T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128)`、`T.Parallel(block_M, block_N)`。测试 PASS。

### prob 7.4 FILL-IN（tilelang_copy2d.py）

三个空：grid 同 7.3；`T.copy(X[by * block_M, bx * block_N], X_shared)`；`T.copy(X_shared, Y[by * block_M, bx * block_N])`。测试 PASS。

对照 2.6：行列号还有对应（tile 起点 `by*block_M, bx*block_N` 自己算）；**grid 尺寸**还有对应（`T.ceildiv` 自己填）；**边界保护没有对应了**——被 `T.copy` 吃掉了，越界的部分它自己处理。

### prob 7.5 CONCEPT（补全表）

| 谁负责 | CUDA SIMT | cuTile | Triton | TileLang |
| --- | --- | --- | --- | --- |
| 线程到数据的映射 | 用户 | 编译器 | 编译器 | 编译器 |
| 边界处理 | 用户 | 编译器 | 编译器 | 编译器（T.copy 一类） |
| tile / block 尺寸的选择 | 用户 | 用户 | 用户 | 用户 |
| block 内同步 | 用户 | 编译器 | 编译器 | 编译器 |

（6.2 的表里也能看到同一个结论：模型接管映射和边界，尺寸留给用户。）

### prob 7.6 FILL-IN（tilelang_matmul.py）

五个空：`A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)`、`B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)`、`C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)`、`T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages)`、`T.copy(A[by*BLOCK_M, k*BLOCK_K], A_shared)` / `T.copy(B[k*BLOCK_K, bx*BLOCK_N], B_shared)` / `T.gemm(A_shared, B_shared, C_local)`。测试 PASS。

回答：shared/fragment/pipeline 这些在 CUDA SIMT 里要手写的东西，Triton 版完全没出现，因为 Triton 的编译器自动决定中间值的存储层级（放寄存器还是 shared）和软件流水——用户只给 `tl.dot` 和 tile 大小。TileLang 刻意把这些暴露成显式的 `alloc_shared/alloc_fragment/T.Pipelined`，控制粒度更细，代价就是用户要自己摆。

### prob 7.7 FROM-SCRATCH：softmax in TileLang（tilelang_softmax.py）

自己写的：`make_softmax(M, N)` 按形状编译 + wrapper 缓存；fragment 宽度取不小于 N 的 2 的幂，尾部补 `-T.infinity(dtype)`；`T.reduce_max` → 减最大值 → `T.exp` → `T.reduce_sum` → 除掉写回。一个 block 管一行。

测试 4 个用例全过，包括"数值巨大"那行（×1000 的输入，先减 max 所以没溢出）。

（选做的性能对比，M=4096，RTX 3060 Laptop 实测）：

| N | 我的实现 | torch.softmax | 我的 GB/s | torch GB/s |
| --- | --- | --- | --- | --- |
| 256 | 45.8 µs | 52.9 µs | 183.3 | 158.7 |
| 1024 | 115.3 µs | 115.3 µs | 291.1 | 291.1 |
| 4096 | 563.4 µs | 574.7 µs | 238.2 | 233.6 |

分析：这种"读一遍 + 行内归约 + 写一遍"的 kernel 就是带宽瓶颈。理论上限是读+写各 4 字节 = 8 B/元素，按 4.7 题量到的连续访问带宽（~250 GB/s）算，N=4096 时我做到了 238 GB/s，约等于把显存跑满了；N=256 时只有 183 GB/s，因为每行只有 1KB，一个 block 一行、4096 个 block 的调度开销占比变大。和 torch.softmax 基本打平（N=256 还快了一点，可能是它对任意行宽更保守）。

### prob 7.8 FROM-SCRATCH（选做）：softmax in Triton（softmax.py）

一个 program 一行，`BLOCK_SIZE = triton.next_power_of_2(N)`，mask 掉越界位置（load 时 `other=-inf`），先减行 max 再 exp 再归一。4 个用例全过。

对比 7.7：Triton 里归约（`tl.max/tl.sum`）、边界（mask）、按形状编译（`triton.jit` + constexpr 特化）全是编译器隐式处理；TileLang 版归约要显式调 `T.reduce_max/reduce_sum`、边界要自己补 -inf、按形状编译要自己写 `make_xxx + cache`。所以 Again 是 7.5 那张表的结论。

---

## Module 8 平台与编译

### prob 8.1 CONCEPT

- (a) **错**。PTX 是给驱动 JIT 的中间表示（虚拟机汇编），不是机器码；GPU 直接执行的是 SASS。
- (b) **错**。sm_70 的 SASS 只能跑在 7.x 上，9.0 的卡不认（没有二进制兼容，除非有 PTX 可以 JIT）。
- (c) **对**。fatbin 就是把多个架构的 SASS + PTX 打包在一起的容器。
- (d) **对**。JIT 由驱动在运行时（首次加载时）完成。

### prob 8.2 EXPERIMENT（选做）

(a) `make sassonly/m0_env/01_hello`（sm_90 SASS）在我的 8.6 卡上运行报错：

```
CUDA error cudaErrorNoKernelImageForDevice at m0_env/01_hello.cu:11:
no kernel image is available for execution on the device
```

(b) `make ptxonly/m0_env/01_hello`（compute_75 PTX）**能正常运行**。PTX 是在程序加载（首次 launch 这个 kernel）时，由**驱动**在运行时 JIT 编译成这块卡（sm_86）的 SASS 的。代价是首次启动多一段 JIT 时间，而且如果驱动太老不认识高版本 PTX 才会真跑不了。

### prob 8.3 CONCEPT（选做）

Runtime API 是高层封装（隐式初始化、方便），Driver API 是底层直接接口（能控制 context、有更细的功能）。`cudaMalloc` 属于 Runtime API。

---

## Bonus（选做）：matmul 调参记录

(a) naive CUDA（1024³ fp32，RTX 3060 Laptop；每个 BS 连测 3 轮取中位）：

| BS | GFLOPS（3 轮范围） | 结论 |
| --- | --- | --- |
| 8 | 469~528（中位 ≈522） | 稳定最差：一个 block 只有 64 线程，访存请求太少 |
| 16 | 492~729（中位 ≈597） | 与 32 相当 |
| 32 | 492~651（中位 ≈625） | 与 16 相当 |

注：笔记本 GPU 的功耗/热状态让单轮数字漂动很大（同一二进制两轮能差 30%），16 和 32 的名次每轮互翻，在我这台机器上分不出明显赢家；能下的结论是 BS=8 明显吃亏、16/32 进入同一档。规范测量应该锁频并跑更多轮取统计量，这里如实记录波动。

(b) Tiled Triton（2048³ fp16）：

| 配置 | TFLOPS |
| --- | --- |
| 32×32×32 | 15.0 |
| 64×64×32 | 17.6 |
| 128×64×32 | **25.7**（最快） |
| 128×128×32 | 20.7 |
| 128×128×64 | 23.1 |
| 64×64×64 | 18.3 |
| torch (cuBLAS) | 26.9 |

(c) TileLang（2048³ fp16，调 `bench()` 里配置）：

| 配置 (BM,BN,BK,threads,stages) | TFLOPS |
| --- | --- |
| 64,64,32,128,1 | 7.9 |
| 64,64,32,128,3 | 15.6 |
| 128,128,32,128,3 | 17.6 |
| 128,128,64,256,3 | **19.1**（最快） |
| 128,256,64,256,3 | （shared memory 超限，编译出的 kernel 申请 144KB，这块卡每 SM 只有 100KB，直接报错） |

分析：
- naive 和 cuBLAS 差了约 50 倍，隔着 tensor core、数据复用、流水这一整条链。
- Triton 最好的 25.7 TFLOPS 已经有 cuBLAS 的 96%，说明 tile 化 + tl.dot（走 tensor core）就是主要收益来源。
- TileLang 这组里最快 19.1，反而比 Triton 略慢。我想到两个原因：一是它把 shared memory / 寄存器 tile / 流水级数都交给我自己指定，要一块卡一块卡地调，我没有 A100 那种大卡去慢慢扫配置；二是我这卡只有 6GB、shared memory 也小，再大一点的配置直接编译不过（128×256 那组要 144KB，超了每 SM 100KB 的上限），所以我的调优空间本来就窄。Triton 把这些都交给编译器自动安排，反而省心。这和讲义里 A100 的观察（TileLang 118 vs Triton 125）方向一致：能控制得更细 ≠ 默认就更快。

---

## 收尾

三次视角切换走完了：单线程 → warp（M1-3）、计算 → 数据搬运（M4-5）、线程 → tile（M6-7）。最让我记住的三个瞬间：1.5 里单线程 GPU 比 CPU 还慢；3.5 里加法次数一样、纯靠活跃线程分布差出 1.5 倍；Bonus 里 naive 和 cuBLAS 差 50 倍。写 kernel 保证正确只是及格线，"知道数据此刻在哪、以什么模式流动"才决定快不快。
