# Tiling in CUTLASS

## What Problem Does Tiling Solve?

A GEMM (General Matrix Multiply) computes `D = alpha * A * B + beta * C` where A is M×K, B is K×N, and C/D are M×N. For real workloads, M, N, and K can be in the thousands or millions. The full matrices don't fit in fast on-chip memory (shared memory, registers), so we can't just load everything and compute.

Tiling solves this by breaking the big matrix multiply into many small matrix multiplies that **do** fit in fast memory. Each small piece is called a **tile**.

## The Three Levels of Tiling

CUTLASS tiles hierarchically, matching the three levels of the GPU execution model:

```
GPU
├── Threadblock (CTA)     ← uses shared memory (tens of KB)
│   ├── Warp              ← uses registers
│   │   └── Instruction   ← Tensor Core MMA operation
│   ├── Warp
│   └── ...
├── Threadblock
└── ...
```

Each level takes a chunk of work from the level above, loads data into its level of the memory hierarchy, and computes.

### Level 1: Threadblock Tile

The full M×N output is divided into tiles. Each threadblock (CTA) is responsible for computing one tile of the output.

```
Output matrix (M × N)
┌────────┬────────┬────────┬───┐
│ TB(0,0)│ TB(0,1)│ TB(0,2)│...│   Each box = one threadblock's work
├────────┼────────┼────────┼───┤   Size = CtaTileM × CtaTileN
│ TB(1,0)│ TB(1,1)│ TB(1,2)│...│
├────────┼────────┼────────┼───┤
│  ...   │  ...   │  ...   │...│
└────────┴────────┴────────┴───┘
```

A threadblock **iterates** along the K dimension in chunks of CtaTileK. Each iteration:
1. Loads a CtaTileM × CtaTileK slice of A from global memory into shared memory
2. Loads a CtaTileK × CtaTileN slice of B from global memory into shared memory
3. Computes the partial matrix product from shared memory
4. Accumulates the result

```
A (M×K)                    B (K×N)
┌──────────────────┐       ┌──────────────────────┐
│  ┌─────┐         │       │  ┌────────────────── │
│  │load │  CtaTileK       │  │  load             │ CtaTileK
│  │ A   │         │       │  └────────────────── │
│  │tile │CtaTileM │       │                      │
│  └─────┘         │       │       CtaTileN       │
│                  │       └──────────────────────┘
└──────────────────┘

iteration 0: load K[0..CtaTileK], compute, accumulate
iteration 1: load K[CtaTileK..2*CtaTileK], compute, accumulate
...
iteration T: load K[...GemmK], compute, accumulate → done
```

The number of iterations along K is:
```
gemm_k_iterations = ceil(GemmK / CtaTileK)
```

In CUTLASS 2.x, the threadblock tile shape (including `CtaTileK`) is defined as:
```cpp
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 64>;
//                                                    M    N    K
// CtaTileK = 64 in this example
```

Here, **128×256** is the output tile this threadblock computes (CtaTileM × CtaTileN), and **64** is how much of the K dimension it processes per mainloop iteration (CtaTileK). The threadblock loops over the full K dimension in steps of 64.

The iteration count itself is computed at runtime in the kernel, since it depends on the actual problem size:
```cpp
// from include/cutlass/gemm/kernel/gemm_universal.h
int gemm_k_iterations = (problem_size_k - offset_k + Mma::Shape::kK - 1) / Mma::Shape::kK;
//                        └── runtime ──┘              └── 64 (from GemmShape) ──┘
```

### Level 2: Warp Tile

Within a threadblock, the CtaTileM × CtaTileN work is divided among multiple warps. Each warp computes a WarpTileM × WarpTileN piece of the threadblock's output.

```
One threadblock tile (CtaTileM × CtaTileN)
┌───────────┬───────────┐
│  Warp(0,0)│  Warp(0,1)│    WarpCount_M = CtaTileM / WarpTileM
│           │           │    WarpCount_N = CtaTileN / WarpTileN
├───────────┼───────────┤
│  Warp(1,0)│  Warp(1,1)│    Total warps = WarpCount_M × WarpCount_N
│           │           │
└───────────┴───────────┘
```

Each warp reads its data from **shared memory** into **registers** and computes using Tensor Core instructions.

In CUTLASS 2.x:
```cpp
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;
//                                              M   N   K
```

The warp count is computed automatically:
```cpp
// from include/cutlass/gemm/threadblock/mma_base.h
using WarpCount = GemmShape<
    Shape::kM / WarpGemm::kM,    // = 128/64 = 2 warps in M
    Shape::kN / WarpGemm::kN,    // = 256/64 = 4 warps in N
    Shape::kK / WarpGemm::kK     // = 64/64  = 1 warp  in K
>;
// Total warps per threadblock = 2 × 4 × 1 = 8 warps = 256 threads
```

Each warp computes a 64×64 output tile. Within that tile, the warp issues multiple Tensor Core MMA instructions (e.g., `mma.sync.m16n8k16`), each of which is a cooperative operation across all 32 threads. For a 16×8 MMA on Ampere, each thread holds 4 output elements per instruction. The warp issues `(64/16) × (64/8) = 32` instructions to cover its tile, so each thread accumulates `4 × 32 = 128` output elements in its registers.

### Level 3: Instruction Tile

Each warp's work is further broken into individual Tensor Core MMA (matrix multiply-accumulate) instructions. The instruction tile size is fixed by the hardware.

```
One warp tile (WarpTileM × WarpTileN)
┌─────┬─────┬─────┬─────┐
│ mma │ mma │ mma │ mma │   Each mma = one Tensor Core instruction
├─────┼─────┼─────┼─────┤   Size = MmaM × MmaN (e.g. 16×8)
│ mma │ mma │ mma │ mma │
├─────┼─────┼─────┼─────┤   MMA count along M = WarpTileM / MmaM
│ mma │ mma │ mma │ mma │   MMA count along N = WarpTileN / MmaN
├─────┼─────┼─────┼─────┤
│ mma │ mma │ mma │ mma │
└─────┴─────┴─────┴─────┘
```

In CUTLASS 2.x:
```cpp
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;
//                                            M  N  K
```

These correspond to actual PTX instructions like `mma.sync.aligned.m16n8k16`.

The number of MMA instructions a warp issues per K-group is:
```cpp
// from include/cutlass/gemm/threadblock/mma_base.h
static int const kWarpGemmIterations = WarpGemm::kK / MmaShape::kK;
// e.g. 64 / 16 = 4 MMA instructions along K per warp-level K-step
```

## Putting It All Together

Here is the complete tiling hierarchy expressed as a loop nest, from `media/docs/cpp/efficient_gemm.md`:

```cpp
// Threadblock-level: grid of threadblocks covers M × N
for (int cta_n = 0; cta_n < GemmN; cta_n += CtaTileN) {           // threadblock concurrency
  for (int cta_m = 0; cta_m < GemmM; cta_m += CtaTileM) {

    // Mainloop: iterate along K dimension (NOT unrolled)
    for (int cta_k = 0; cta_k < GemmK; cta_k += CtaTileK) {

      // Warp-level: warps within a threadblock
      for (int warp_n = 0; warp_n < CtaTileN; warp_n += WarpTileN) {   // warp parallelism
        for (int warp_m = 0; warp_m < CtaTileM; warp_m += WarpTileM) {

          // Instruction-level: MMA instructions within a warp (fully unrolled)
          for (int warp_k = 0; warp_k < CtaTileK; warp_k += WarpTileK) {
            for (int mma_k = 0; mma_k < WarpTileK; mma_k += MmaK) {
              for (int mma_n = 0; mma_n < WarpTileN; mma_n += MmaN) {
                for (int mma_m = 0; mma_m < WarpTileM; mma_m += MmaM) {

                  mma_instruction(d, a, b, c);  // one Tensor Core operation

                }
              }
            }
          }

        }
      }
    }

  }
}
```

The outer two loops map to the GPU grid (each iteration = one threadblock). The K loop is the **mainloop** that runs on each threadblock. The warp loops map to warps within the threadblock. The inner MMA loops are fully unrolled at compile time.

## How Are Tile Sizes Determined?

Tile sizes are **compile-time template parameters**. They are chosen to balance several constraints:

### Threadblock Tile Size

| Factor | Larger tile | Smaller tile |
|--------|------------|--------------|
| Data reuse | More reuse from shared memory, fewer global loads | Less reuse |
| Shared memory | Needs more shared memory per threadblock | Needs less |
| Occupancy | Fewer threadblocks can run concurrently | More threadblocks can run |
| Tail effects | Worse if M or N isn't divisible by tile size | Better for small problems |

Typical threadblock tiles: 64×64, 128×128, 128×256, 256×128.

The K dimension of the threadblock tile (CtaTileK) determines how much data is loaded into shared memory per mainloop iteration. This must fit in shared memory alongside any double-buffering stages.

### Warp Tile Size

The warp tile must evenly divide the threadblock tile:
```
CtaTileM % WarpTileM == 0
CtaTileN % WarpTileN == 0
```

The number of warps this produces determines the thread count of the threadblock:
```
threads = (CtaTileM/WarpTileM) × (CtaTileN/WarpTileN) × 32
```

Typical warp tiles: 32×32, 64×64, 64×32.

### Instruction Tile Size

This is fixed by the GPU architecture. Examples:

| Architecture | Data type | Instruction shape (M×N×K) |
|-------------|-----------|--------------------------|
| Turing (SM75) | INT8 | 8×8×16 |
| Ampere (SM80) | FP16 | 16×8×16 |
| Ampere (SM80) | BF16 | 16×8×16 |
| Ampere (SM80) | TF32 | 16×8×4 |

## Memory and Pipelining

### Shared Memory Layout

The threadblock allocates shared memory to hold tiles of A and B. With multi-stage pipelining, multiple K-tiles are buffered:

```cpp
// from include/cutlass/gemm/threadblock/mma_base.h
using ShapeA = MatrixShape<
    Shape::kM + padding,           // CtaTileM + padding (avoid bank conflicts)
    Shape::kK * kStages + padding  // CtaTileK × num_stages
>;

using ShapeB = MatrixShape<
    Shape::kK * kStages + padding,  // CtaTileK × num_stages
    Shape::kN + padding             // CtaTileN + padding
>;
```

### Software Pipelining

CUTLASS overlaps memory loads with computation using double (or multi-stage) buffering:

- **Threadblock level**: While computing from one shared memory tile, the next tile is being loaded from global memory.
- **Warp level**: While computing from one register fragment, the next fragment is being loaded from shared memory.

From the actual mainloop code (`include/cutlass/gemm/threadblock/mma_pipelined.h`):

```cpp
// Double-buffered fragments for overlapping loads and compute
WarpFragmentA warp_frag_A[2];
WarpFragmentB warp_frag_B[2];

for (; gemm_k_iterations > 0; --gemm_k_iterations) {       // iterate over K tiles
  for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; ++warp_mma_k) {

    // Load NEXT warp fragment from shared memory (into the other buffer)
    warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
    warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);

    // Meanwhile, load next tile from global → shared (only on first inner iteration)
    if (warp_mma_k == 0) {
      iterator_A.load(tb_frag_A);
      iterator_B.load(tb_frag_B);
    }

    // Compute MMA using CURRENT fragment
    warp_mma(accum, warp_frag_A[warp_mma_k % 2], warp_frag_B[warp_mma_k % 2], accum);
  }
}
```

This ensures the GPU is never idle waiting for memory.

## Compile Time vs. Runtime

| What | When | Example |
|------|------|---------|
| Tile sizes | Compile time (template parameters) | `GemmShape<128, 256, 64>` |
| Problem sizes (M, N, K) | Runtime (function arguments) | `{M, N, K}` passed to `gemm_op()` |
| Number of threadblocks | Runtime (computed from both) | `ceil(M/128) × ceil(N/256)` |
| Number of K iterations | Runtime (computed from both) | `ceil(K/64)` |
| Loop body (unrolled MMA ops) | Compile time (fully unrolled) | Fixed sequence of `mma.sync` |

The tile sizes are baked into the kernel as template parameters. This lets the compiler fully unroll inner loops, allocate exact shared memory sizes, and generate specialized Tensor Core instructions. The problem sizes are runtime values that only affect how many tiles to launch and how many K iterations to run.

## CUTLASS 2.x vs 3.x

In CUTLASS 2.x, tiles are specified with three explicit `GemmShape` parameters:

```cpp
using Gemm = cutlass::gemm::device::Gemm<
    ...,
    cutlass::gemm::GemmShape<128, 256, 64>,   // threadblock tile
    cutlass::gemm::GemmShape<64, 64, 64>,     // warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,      // instruction tile
    ...>;
```

In CUTLASS 3.x, the `CollectiveBuilder` abstracts some of this. You specify the threadblock tile and cluster shape, and the builder selects the warp and instruction tiling automatically:

```cpp
using TileShape = Shape<_128, _128, _64>;       // threadblock tile
using ClusterShape = Shape<_2, _2, _1>;         // SM90+: groups of threadblocks

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,  // auto-select pipeline stages
    KernelScheduleAuto                          // auto-select schedule
>::CollectiveOp;
```

CUTLASS 3.x also introduces **cluster tiling** (SM90+), where multiple threadblocks in a cluster can cooperate via distributed shared memory, adding another level to the hierarchy above the threadblock.
