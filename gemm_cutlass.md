---
layout: page
title: GEMM implementation in CUTLASS
---

# The CUTLASS 2.x GEMM Device Template

CUTLASS 3.x and 4.x use cute c++ to write GEMM. will explore it later. 

## Overview

`cutlass::gemm::device::Gemm` is the main entry point for launching a GEMM kernel in CUTLASS 2.x. It computes:

```
D = alpha * A * B + beta * C
```

where A is M×K, B is K×N, and C/D are M×N.

The class is defined in `include/cutlass/gemm/device/gemm.h`. It is a C++ template that takes compile-time parameters describing the data types, layouts, tile sizes, and target architecture, and provides a host-callable `operator()` that launches the kernel on the GPU.

## Gemm Template Parameters

OperatorClass_ should be OpClassSimt for slow scalar FMA (general for all arch), and can be OpClassTensorOp for >SM70 arch (Volta). 

ThreadblockShape_ can be auto determined by using DefaultGemmConfiguration(...), basically a lookup table implemented via template specialization- you give it (OperatorClass, ArchTag, ElementA dtype, ElementB, ElementC, ElementAccumulator) and it returns the default tile sizes, stages, alignment, etc.

WarpShape_ and InstructionShape_ is simiar to ThreadblockShape_. They are 3-level tiling config. 

Default tile sizes for `OpClassTensorOp` with `half_t`:

| Architecture | ThreadblockShape | WarpShape | InstructionShape | Stages |
|-------------|-----------------|-----------|-----------------|--------|
| Sm75 | 128×256×32 | 64×64×32 | 16×8×8 | 2 |
| Sm80 | 128×256×64 | 64×64×64 | 16×8×16 | 3 |

You can override any of these:

```cpp
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, LayoutA,
    cutlass::half_t, LayoutB,
    cutlass::half_t, LayoutC,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 64>,   // custom threadblock tile
    cutlass::gemm::GemmShape<64, 64, 64>,     // custom warp tile
    cutlass::gemm::GemmShape<16, 8, 16>       // instruction tile (fixed by hardware)
>;
```

#### SplitKSerial (default: false)

Enables split-K mode, where the K dimension is partitioned across multiple threadblocks to improve GPU utilization when M and N are small (see `docs/tiling.md`, "Split-K: Parallelizing Along K").

## Runtime Arguments

Arguments (user-facing host)                                                                                                                   
      │                                       
      │  initialize()                     
      ▼                                       
  GemmKernel::Params (internal, copied to GPU)                                                                                                    
      │                                       
      │  run() — kernel launch                                                                                                                    
      ▼           
  GPU kernel reads params_ to know problem size, pointers, strides, grid shape    

The `Arguments` struct holds the runtime parameters:

```cpp
struct Arguments {
    GemmCoord problem_size;                    // {M, N, K}
    TensorRef<ElementA const, LayoutA> ref_A;  // pointer + stride for A
    TensorRef<ElementB const, LayoutB> ref_B;  // pointer + stride for B
    TensorRef<ElementC const, LayoutC> ref_C;  // pointer + stride for C (source)
    TensorRef<ElementC, LayoutC> ref_D;        // pointer + stride for D (destination)
    typename EpilogueOutputOp::Params epilogue; // {alpha, beta}
    int split_k_slices;                        // number of K partitions (default: 1)
};
```

C and D can point to the same memory (in-place update).


## The Chain of Template Instantiation

`device::Gemm` doesn't contain the mainloop or epilogue directly. It assembles the kernel through a chain of template metaprogramming:

```
device::Gemm                                          (include/cutlass/gemm/device/gemm.h)
  │
  ├── DefaultGemmConfiguration                        (device/default_gemm_configuration.h)
  │     fills in default tile sizes, stages, alignment, etc.
  │
  └── kernel::DefaultGemm                             (kernel/default_gemm.h)
        │   one partial specialization per architecture (Sm70, Sm75, Sm80, Sm89, Sm90)
        │
        ├── threadblock::DefaultMma::ThreadblockMma    (threadblock/default_mma.h)
        │     selects the mainloop based on Stages:
        │       Stages == 2 (via template param matching) →  MmaPipelined           (threadblock/mma_pipelined.h)
        │       Stages > 2   →  MmaMultistage          (threadblock/mma_multistage.h)
        │
        ├── Epilogue   (apply alpha beta)                                (epilogue/threadblock/default_epilogue_*.h)
        │
        └── kernel::Gemm<Mma, Epilogue, ...>           (kernel/gemm.h — the GPU kernel entry point)
```

### MmaPipelined

```cpp
template<...>
class MmaPipelined {
  ...
  void operator()(
    int gemm_k_iterations,                            ///< number of iterations of the mainloop
    FragmentC &accum,                                 ///< destination accumulator tile
    IteratorA iterator_A,                             ///< iterator over A operand in global memory
    IteratorB iterator_B,                             ///< iterator over B operand in global memory
    FragmentC const &src_accum)                       ///< source accumulator tile
  {
    // Prologue, iterator_A is in unit of CtaTileM x CtaTileK . iterator_B in unit of CtaTileK x CtaTileN
    // like 128*64
    prologue(iterator_A, iterator_B, gemm_k_iterations);
    // Each thread loads its portion of the first CtaTile (A and B) from global memory
    // into registers (FragmentA / FragmentB), then stores them to shared memory buffer 0.
    // Advances the shared memory write pointer to buffer 1 (wraps to 0 if past the end).
    // smem_iterator_A_ stored will be read by warp_tile_iterator_A_

    // Wait until we have at least one completed global fetch stage
    gmem_wait();

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    // Perform the MAC-iterations, mainloop. shared memory has first CtaTile and iterator_* has other tiles in GM 
    gemm_iters(gemm_k_iterations, accum, iterator_A, iterator_B);
  }
```

`MmaPipelined` (`include/cutlass/gemm/threadblock/mma_pipelined.h`) is the double-buffered mainloop. It is the default for Sm70 and Sm75 (where `DefaultGemmConfiguration` sets `kStages = 2`), but it is not architecture-restricted — any architecture can use it if `Stages` is set to 2. It has a compile-time assertion enforcing exactly 2 stages:

```cpp
static_assert((Base::kStages==2), "MmaPipelined requires kStages set to value 2");
```

#### Mainloop gemm_iters

```cpp
// from include/cutlass/gemm/threadblock/mma_pipelined.h

// Double-buffered warp fragments for overlapping shared memory loads and compute
WarpFragmentA warp_frag_A[2];
WarpFragmentB warp_frag_B[2];
// Load A fragment from shared A
// warp tile K dimension will be split into multi K-group, each K-group is for one MMA instruction
// i.e. warp tile K = 64, MMA K = 16, there will be 4 K-group
this->warp_tile_iterator_A_.set_kgroup_index(0); // start from 0 K-group
this->warp_tile_iterator_A_.load(warp_frag_A[0]);
++this->warp_tile_iterator_A_;
... // load warp_frag_B[0]
// Pair of fragments used to overlap global memory loads and math instructions;
FragmentA tb_frag_A;
FragmentB tb_frag_B; 
  
for (; gemm_k_iterations > 0; --gemm_k_iterations) {        // outer: K-tiles
    // gemm_k_iterations = ceil(K / CtaTileK)
    for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; ++warp_mma_k) { // inner: MMAs
        // kWarpGemmIterations = WarpTileK / MmaK, WarpTileK == CtaTileK in most time, so no need for 
        // cta_k_iteration
        // At the LAST inner iteration: store the globally-loaded tile to shared memory,
        // __syncthreads(), and swap buffers, because ctatile in one shared mem buffer has been consumed.
        if (warp_mma_k == kWarpGemmIterations - 1) {
            smem_iterator_A_.store(tb_frag_A);      // write NEXT tile to shared
            smem_iterator_B_.store(tb_frag_B);
            __syncthreads();                         // barrier
            advance_smem_stages();                   // swap read/write buffers (XOR toggle)
        }

        // Load NEXT warp fragment from shared memory (into the other register buffer)
        warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
        warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);

        // At the FIRST inner iteration: start loading the NEXT K-tile from global memory
        if (warp_mma_k == 0) { 
            iterator_A.load(tb_frag_A);  // global → register, in last mma_k, register-->share mem
            iterator_B.load(tb_frag_B);
            ...
            // some boundary protection to avoid load ctatile using iterator clear_mask
        }

        // Compute MMA using CURRENT register buffer
        warp_mma(accum, warp_frag_A[warp_mma_k % 2], warp_frag_B[warp_mma_k % 2], accum);
    }
}
```
some bad naming in the code: 
 ┌────────────────────────────────┬───────────────────────────────┬─────────┐
  │          Name in code          │          What it is           │ Example │
  ├────────────────────────────────┼───────────────────────────────┼─────────┤
  │ Mma::Shape::kK                 │ Threadblock tile K (CtaTileK) │ 64      │
  ├────────────────────────────────┼───────────────────────────────┼─────────┤
  │ WarpGemm::kK                   │ Warp tile K (WarpTileK)       │ 64      │
  ├────────────────────────────────┼───────────────────────────────┼─────────┤
  │ Operator::Policy::MmaShape::kK │ MMA instruction K (MmaK)      │ 16      │
  └────────────────────────────────┴───────────────────────────────┴─────────┘

### MmaMultistage
TBD
