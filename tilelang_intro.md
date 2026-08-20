---
layout: page
title: TileLang Introduction
---

# TileLang Introduction

## Why TileLang

Writing a fast GPU GEMM has always meant choosing a bad trade-off:

- **Raw CUDA** — you control everything (thread indexing, shared-memory staging, `mma` instructions, barriers), but you write hundreds of lines of boilerplate and hand-tune every layout to avoid bank conflicts and hit the tensor cores. Getting it right is expert work.
- **CUTLASS / CuTe** — template libraries that package the expert knowledge. They reach state-of-the-art performance, but the abstraction (layouts, atoms, `TiledMma`, TMA partitioning) is dense and the C++ template machinery is hard to read and slow to iterate on. See the [CuTe DSL notes](cute_dsl) for a taste.
- **Triton** — Pythonic and productive: you write blocked code and the compiler auto-schedules memory movement. The cost is control — the memory hierarchy and layout choices are mostly hidden, so when the autoscheduler leaves performance on the table there is little you can do.

TileLang aims at the **middle ground**: as approachable as Triton, but with the explicit control over the memory hierarchy and pipelining that CUTLASS gives you — without the C++ template pain.

Its design rests on three ideas:

1. **Pythonic** — a kernel is plain Python, JIT-compiled through a compiler built on Apache TVM. It drops straight into a PyTorch workflow.
2. **Dataflow-centric** — you describe *what data moves where* (global → shared → registers → compute), and the compiler handles the low-level scheduling.
3. **Composable** — kernels, tile primitives, and scheduling hints combine cleanly, so FlashAttention, dequant-GEMM, and linear attention are built from the same building blocks.

The key move is **separating dataflow from scheduling and layout**. You write the tile-level algorithm once; the compiler *infers* the register/shared-memory layouts and (optionally) *autotunes* the block sizes and pipeline depth. In CuTe you spell those layouts out by hand.

## The programming model

TileLang works at the **tile** level: the unit of reasoning is a block of data (`block_M × block_K`), not a scalar element. You explicitly place tiles in **memory scopes** that mirror the GPU hierarchy:

| Primitive | Scope | Maps to |
|-----------|-------|---------|
| `T.Tensor` | global | HBM / DRAM input-output |
| `T.alloc_shared` | shared | on-chip shared memory (SMEM) |
| `T.alloc_fragment` | local | registers (a per-thread fragment) |

Data movement between scopes is a single primitive, `T.copy`, and the compiler decides *how* to move it — vectorized loads, `cp.async`, or TMA on Hopper/Blackwell — based on the layout it infers.

## A minimal GEMM

Here is a complete tiled `C = A × B` in TileLang:

```python
import tilelang
import tilelang.language as T

def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum="float"):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # 1. One CTA (thread block) per output tile; grid = (N/bN, M/bM)
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                      threads=128) as (bx, by):
            # 2. Declare the tiles and where they live
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum)  # registers

            # 3. Zero the accumulator
            T.clear(C_local)

            # 4. Main loop over K, software-pipelined across 3 stages
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)  # gmem -> smem
                T.copy(B[ko * block_K, bx * block_N], B_shared)  # gmem -> smem
                T.gemm(A_shared, B_shared, C_local)              # tensor-core MMA

            # 5. Write the result tile back to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])
    return main

# Compile and run
kernel = tilelang.compile(matmul(1024, 1024, 1024, 128, 128, 32))
import torch
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = kernel(a, b)
```

That is the whole kernel. A few things to notice:

- `T.Kernel(...) as (bx, by)` declares the grid and hands you the block indices — the equivalent of `blockIdx.x/y`. `threads=128` sets the block size; you never index individual threads.
- `T.clear`, `T.copy`, and `T.gemm` are **tile-level** — one call moves or multiplies an entire tile. There is no per-thread loop, no manual `__syncthreads()`, and no explicit `mma` instruction.
- The three scopes (`Tensor`, `alloc_shared`, `alloc_fragment`) make the memory hierarchy explicit — you decide A and B stage through shared memory and the accumulator lives in registers, exactly as you would in CUTLASS, but in three lines.
- **Nothing here mentions a layout.** How `A_shared` is swizzled to avoid bank conflicts, and which registers of which threads hold `C_local`, are *inferred* by the compiler from how the tiles feed `T.gemm`.

### Epilogue and fusion

Because the accumulator is just a tile in registers, fusing an epilogue is trivial — you write element-wise math over it with `T.Parallel` before the final copy:

```python
    for i, j in T.Parallel(block_M, block_N):
        C_local[i, j] = T.max(C_local[i, j], 0)  # fused ReLU
    T.copy(C_local, C[by * block_M, bx * block_N])
```

`T.Parallel` marks a loop nest as parallel over the block's threads; the compiler infers the thread→element mapping. This is how FlashAttention (softmax fused between two GEMMs) is expressed without leaving the kernel.

## Key primitives

| Primitive | Role |
|-----------|------|
| `T.Kernel(grid…, threads=)` | declare grid + block size, yields block indices |
| `T.alloc_shared / alloc_fragment` | allocate a tile in shared memory / registers |
| `T.copy` | move a tile between scopes (compiler picks the mechanism) |
| `T.gemm` | tile matrix-multiply, mapped to tensor cores |
| `T.Pipelined(n, num_stages=)` | software-pipelined loop (multi-stage ring buffer) |
| `T.Parallel` | parallel element-wise loop over the block (epilogues) |
| `T.clear / T.reduce_*` | init and reductions on tiles |
| `T.annotate_layout` | *optionally* override the inferred layout |

## Layout inference and autotuning — the differentiator

The reason the GEMM above is so short is **layout inference**. In CUTLASS/CuTe you must construct, by hand:

- the swizzled shared-memory layout of A and B (to kill bank conflicts),
- the thread-value (TV) layout that says which thread owns which register of the accumulator,
- the TMA partitioning that cuts a gmem tile into transfer units.

TileLang derives all of these from the dataflow and the target architecture. When the defaults are not optimal, you can pin a specific layout with `T.annotate_layout` — you drop to manual control *only where you need it*, instead of specifying everything up front.

On top of that, block sizes (`block_M/N/K`) and `num_stages` are ordinary Python arguments, so they can be swept by the built-in **autotuner** to find the best configuration per shape and per GPU — the same source compiles to CUDA, HIP (AMD), and CPU backends.

## Software pipelining

`T.Pipelined(..., num_stages=3)` is the same idea as the multi-stage ring buffer in the [CuTe GEMM](cute_dsl): while the tensor core consumes stage *s*, the copy engine prefetches stage *s+1* into the next shared-memory buffer, hiding global-memory latency. In CuTe you build this with explicit `PipelineTmaUmma` producer/consumer objects, barriers, and stage bookkeeping. In TileLang it is one argument — the compiler allocates the extra shared-memory stages and inserts the barriers.

## TileLang vs CuTe DSL

Both let you write GPU kernels in Python and both compile down to the same tensor-core instructions. The difference is **how much you specify** and **how much control you keep**.

| | TileLang | CuTe DSL |
|--|----------|----------|
| Level | tile-level dataflow | layout algebra + atoms |
| Layouts | **inferred** (override optional) | **explicit** (`ComposedLayout`, TV layouts) |
| Pipelining | `num_stages=` argument | manual producer/consumer + barriers |
| Data movement | `T.copy` (mechanism auto-chosen) | explicit `tma_partition` / `CopyAtom` |
| Thread mapping | hidden (`threads=`, `T.Parallel`) | explicit (`thr_mma`, `get_slice`) |
| Backends | CUDA, HIP, CPU | NVIDIA only |
| Autotuning | built-in | you write the sweep |
| Ceiling | high, compiler-mediated | maximal, you control every layout |
| Lines for a GEMM | ~20 | ~150 (see [CuTe notes](cute_dsl)) |

**How to read this:** CuTe hands you the raw layout algebra — `TiledMma`, `ComposedLayout`, `tma_partition`, TV layouts. Nothing is hidden, so nothing is out of reach: you can hand-place every element and squeeze the last few percent, at the cost of writing (and understanding) all of it. TileLang inverts the default — the compiler infers the layouts and pipelines from your dataflow, and you reach for `T.annotate_layout` only when its choice is not good enough. You trade a sliver of peak control for an order of magnitude less code and instant portability across backends.

Notably the two are converging: TileLang recently added a **CuTe DSL backend**, so a TileLang kernel can lower *through* CuTe — using CuTe as the code generator while you keep TileLang's productivity at the source level.

## Can TileLang be adapted to ASICS NPUs?

Yes — and it is already happening. Because TileLang describes a kernel as **tiles moving through explicit memory scopes**, not as SIMT threads, the front-end is largely architecture-agnostic. Porting to a new accelerator is mostly a matter of teaching the compiler a new backend plus the target's memory-scope and layout rules. Two adapters already exist: [`tilelang-ascend`](https://github.com/tile-ai/tilelang-ascend) for Huawei Ascend NPUs and [`tilelang-metax`](https://github.com/tile-ai/tilelang-metax) for MetaX GPUs.

The interesting case is Ascend, because an NPU is **not** just a GPU with different names. Two things differ:

**1. Asymmetric compute units.** A GPU has symmetric SIMT cores that do both math and data movement. An Ascend NPU splits the work across specialized units:

- **Cube cores** — the matmul engine (a large systolic array), analogous to tensor cores but standalone.
- **Vector cores** — element-wise / reduction / data-movement engine.

These two exchange data through global memory / L2, so a fused operator has to be explicitly *partitioned* across cube and vector units. TileLang models this by extending its scope idea from *memory* scopes to *compute* scopes: `T.Scope("C")` marks cube work and `T.Scope("V")` marks vector work. In **developer mode** the compiler splits the kernel across the two automatically and inserts the cross-unit synchronization; **expert mode** lets you place the scopes and manage the flags by hand.

**2. A deeper, named memory hierarchy.** The tile scopes remap cleanly:

| TileLang scope | GPU | Ascend NPU |
|----------------|-----|------------|
| global (`T.Tensor`) | HBM | global memory |
| shared (`T.alloc_shared`) | SMEM | L1 buffer (cube) / Unified Buffer (vector) |
| local (`T.alloc_fragment`) | registers | L0A / L0B (MMA inputs), L0C (accumulator) |

The `T.copy` / `T.gemm` / `T.Pipelined` primitives keep the same meaning — `T.gemm` targets the cube array, `T.Parallel` vectorizes onto the vector cores, and `T.Pipelined` still hides the global-memory latency of staging tiles into L1. What changes underneath is the codegen: instead of emitting CUDA/PTX, the Ascend backend lowers to **Ascend C** (via a Program-Transformation-Optimizer path) or to **AscendNPU IR**, running on Huawei's CANN stack.

This is the payoff of the dataflow-centric design: the same tile-level source is portable, and the hard, architecture-specific parts (the cube/vector split, the L0/L1 layout rules) are absorbed by the compiler and layout inference rather than being rewritten by hand — exactly the work you *would* rewrite by hand in CUTLASS/CuTe, which is NVIDIA-only.

The honest caveats: NPU support is younger and less battle-tested than the CUDA backend, the cube/vector partitioning can still need expert-mode hints to reach peak, and the layout-inference rules for a systolic array differ from a GPU's, so a kernel that is optimal on CUDA is not automatically optimal on Ascend.

## When to use which

- **Reach for TileLang** when you want CUTLASS-class performance quickly, are iterating on a fused operator (FlashAttention, dequant-GEMM), or need one source to run on both NVIDIA and AMD.
- **Reach for CuTe** when you are chasing the absolute peak on a specific NVIDIA architecture and need to control every layout and instruction by hand — or when you are writing the backend that TileLang itself lowers to.

For most kernel work, TileLang gets you 90%+ of the performance for a fraction of the effort, and leaves the escape hatch open when you need it.
