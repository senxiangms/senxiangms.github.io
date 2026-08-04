---
layout: page
title: Cute DSL
---

# Cute DSL
## Metaprogramming

In CUTLASS C++, metaprogramming is done via C++ templates — tile sizes, layouts, and data types are all compile-time template parameters that the compiler resolves and optimizes away. This is powerful but results in complex, hard-to-read template code.

CuTe DSL replaces C++ templates with **Python as the metaprogramming language**. A CuTe DSL kernel is a Python function decorated with `@cute.kernel`. When you "compile" a kernel, the Python interpreter runs first:

1. **Python executes at compile time** — all Python-level expressions (shapes, layouts, allocations, tiling arithmetic) are evaluated by the Python interpreter. These produce concrete values.
2. **Concrete values become IR constants** — the results are lowered into compiler IR (MLIR/LLVM) as hard-coded constants, just like C++ template parameters become compile-time constants.
3. **The GPU kernel is generated** — the compiler produces optimized GPU machine code with all tile sizes, memory layouts, and loop bounds baked in.

For example, `smem = cutlass.utils.SmemAllocator()` is a normal Python object — the Python interpreter computes shared memory addresses at compile time, and these addresses appear as literal constants in the generated kernel. Similarly, `mma_tiler_mnk = (128, 256, 64)` is a Python tuple that gets folded into the IR, not a runtime variable.

This means Python code in a CuTe DSL kernel serves two roles:
- **Lines that touch `cute.*` primitives** generate GPU instructions (loads, MMAs, barriers)
- **Plain Python expressions** (arithmetic, tuples, control flow) execute at compile time and produce constants

The result is the same performance as hand-written CUTLASS C++ templates, but with Python readability.

## Blackwell FP16 GEMM in CuTe DSL: Level 0

### Tiling
```python
mma_inst_shape_mnk = (128, 256, 16)  # one warp issues an MMA instruction producing a 128×256 output
mma_tiler_mnk = (128, 256, 64)  # equals CtaTile in 1-CTA mode, or 2×CtaTile_M in 2-CTA mode
threads_per_cta = 128 # 4 warps in a cta. warp 0-1: epilogue (write result from TMEM to global mem)
# warp 2: MMA, warp3: TMA (tensor memory accelerator)
```
On Blackwell, the accumulator has been moved to TMEM (Tensor Memory), which frees up register file space and allows much larger instruction tiles than previous architectures.

Blackwell also introduces a **2-CTA mode**, where two CTAs cooperatively execute a single UMMA instruction (`tcgen05.CtaGroup.Two`). In 2-CTA mode, the M dimension of `mma_tiler_mnk` is 2× the per-CTA tile size (`CtaTileM`), since both CTAs contribute to the same MMA operation. 

### software pipeling
```python
ab_stages = 4 # matrix A B 's shared memory ring buffer nums
acc_stage = 1 # there is only one TMEM accumulator buffer, 
# so only after epilogue write TMEM to global, MMA can use TMEM again for accumulation
```

### cute DSL kernel declaration
```python
@cute.kernel # same as CUDA C++ __global__ 
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
):
```
cute.TiledMma is a packed object of MMA HW instruction (inherited from Atom/MmaAtom), data partition rules which is determined by MMA HW instruction.  

MmaAtom.thr_id is a function mapping logic coordinate (i0,i1) to hardware thread id. like hw_id=i0*1+i1*16
MmaAtom.shape_mnk tells instruction shape like (128 256, 16)
MmaAtom.tv_layout_A tells in matrix A which elements are held by threads
MmaAtom.tv_layout_B, and Mmatom.tv_layout_C do similar mapping. 

Atom is smallest harward operation. besides MMA atom, there is copy atom. like 
tma_atom_a: cute.CopyAtom, tma_atom_b: cute.CopyAtom, it encapsulates how the data is moved via TMA.
A tensor's shape, stride, layout is different from B, so it needs two CopyAtom. 

mA_mkl is input tensor A, shape is (M, N, L) where L is batch dimension. mB_nkl is input tensor B, shape is (N, K, L). mC_mnl is output tensor C, shape is (M, N, L). 

ComposedLayout is composed of outer and inner layouts. outer maps coordinates (x, y) into logic address, and inner maps logic address into physical address via swizzle XOR to avoid smem bank conflicts. 

### kernel prologue

```python
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord_mnk = (bidx, bidy, None)
```

mma_coord_mnk can be used by cute.local_tile to which part of A and B will be assigned to block (bidx, bidy) in a grid. Here, row segment with bidx index from A, and column segment with bidy index from B will be assigned to block (bidx, bidy). For K dimension, all columns (K dim in A) and all rows (K dim in B) will be included in the tile, mainloop will iterate over K dimension. 

```python
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
```
SmemAllocator is normal python object. Python interpreter will execute this code to generate some SMem address which will be used as a hard coded magic number in IR.  