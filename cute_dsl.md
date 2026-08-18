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

### Kernel Framework
```python
def kernel(
    tiled_mma,          # how the compute instruction is partitioned
    tma_atom_a, mA_mkl, # A: how it is moved + global gmem view
    tma_atom_b, mB_nkl, # B: how it is moved + global gmem view
    mC_mnl,             # C: global gmem view (output)
    a_smem_layout,      # A's layout in smem
    b_smem_layout,      # B's layout in smem
):
    # 1. Each CTA claims its own tile (K is split into RestK blocks)
    gA = local_tile(mA_mkl, tile_shape, coord, proj=(1, None, 1))  # (bM, bK, RestK)
    gB = local_tile(mB_nkl, tile_shape, coord, proj=(None, 1, 1))  # (bN, bK, RestK)
    gC = local_tile(mC_mnl, tile_shape, coord, proj=(1, 1, None))  # (bM, bN)

    # 2. Prepare the smem destination buffers (multi-stage ring buffer)
    sA = make_smem_tensor(a_smem_layout)  # [bM, bK, Stage]
    sB = make_smem_tensor(b_smem_layout)  # [bN, bK, Stage]

    # 3. Pair up gmem <-> smem coords for TMA, with the pipeline stage as the outermost mode
    tAgA, tAsA = tma_partition(tma_atom_a, ..., sA, gA)
    tBgB, tBsB = tma_partition(tma_atom_b, ..., sB, gB)

    # 4. Main loop (software pipeline)
    for k in range(RestK):
        stage = k % Stage
        copy(tma_atom_a, tAgA[k], tAsA[stage])   # TMA: gmem -> smem (big block)
        copy(tma_atom_b, tBgB[k], tBsB[stage])
        # tensor core iterates over the smem block, one MMA instruction at a time 
        # (small block x many in M, N, K direction. For blackwell, only K direction)
        acc = tiled_mma(acc, sA[stage], sB[stage])

    # 5. Write C back
    copy(acc, gC)
```

### Tiling
```python
mma_inst_shape_mnk = (128, 256, 16)  # one warp issues an MMA instruction producing a 128×256 output
mma_tiler_mnk = (128, 256, 64)  # equals CtaTile in 1-CTA mode, or 2×CtaTile_M in 2-CTA mode
threads_per_cta = 128 # 4 warps in a cta. warp 0-1: epilogue (write result from TMEM to global mem)
# warp 2: MMA, warp3: TMA (tensor memory accelerator)
```
On Blackwell, the accumulator has been moved to TMEM (Tensor Memory), which frees up register file space and allows much larger instruction tiles than previous architectures.

Blackwell also introduces a **2-CTA mode**, where two CTAs cooperatively execute a single UMMA instruction (`tcgen05.CtaGroup.Two`). In 2-CTA mode, the M dimension of `mma_tiler_mnk` is 2× the per-CTA tile size (`CtaTileM`), since both CTAs contribute to the same MMA operation. 

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

ComposedLayout is composed of outer and inner layouts. outer maps coordinates (x, y) into logic address, and inner maps logic address into physical address via swizzle XOR to avoid smem bank conflicts. outer 's shape is (atom, rest_m, rest_k, stages). atom is a 2d block repeated rest_m times in M dim, rest_k times in K dim. "rest" is a strange naming anyway. Tile in shared memory is composed of atoms, which has swizzle to avoid bank conflict. 

When cute.select(smem_layout, mode=[0, 1, 2]), (atom, rest_m, rest_k) determines a complete tile of one stage in shared memory. 

### Partition tensors for MMA

```python
    # (bM, bK, RestK): RestK = number of bK chunks needed to cover the full K dimension
    gA = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    # (bN, bK, RestK)
    gB = cute.local_tile(mB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    # (bM, bN)
    gC = cute.local_tile(mC_mnl, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))
```

`gA` and `gB` are the global memory tiles of A and B needed to compute the `(bM, bN)` output tile. `gC` is the corresponding output tile in global memory.

`proj` controls which dimensions of `mma_tiler_mnk` are applied: `1` means tile that dimension using the coordinate, `None` means keep the full extent (for iteration). For `gA`, M is tiled by `bidx`, K is kept in full (split into `RestK` chunks for the mainloop), and N is projected out.

```python
    thr_mma = tiled_mma.get_slice(0)
    # (MMA, MMA_M, MMA_K)
    # MMA dim = the A-operand block one atom reads (an M×K sub-block);
    # MMA_M, MMA_K = how many times the atom is tiled along M and K.
    # MMA's view of A in gmem — establishes the logical coord/shape
    # correspondence. It is not the copy operand itself: tma_partition later
    # consumes tCgA and re-cuts it into TMA copy units (tAgA), and the actual
    # gmem->smem transfer runs on tAgA/tAsA (see the TMA partition step below).
    tCgA = thr_mma.partition_A(gA)
    # (MMA, MMA_N, MMA_K)
    # MMA dim = the B-operand block one atom reads (an N×K sub-block);
    # MMA_N, MMA_K = atom tiling counts along N and K.
    tCgB = thr_mma.partition_B(gB)
    # (MMA, MMA_M, MMA_N)
    # MMA dim = the 2D output block one atom produces;
    # MMA_M, MMA_N = atom tiling counts along M and N.
    # gmem view of the accumulator for write-back; used by the epilogue's
    # tmem->gmem path (accumulator lives in TMEM, not smem/registers).
    tCgC = thr_mma.partition_C(gC)
    # (MMA, MMA_M, MMA_K)
    # A descriptor view over sA — NOT a copy, no data moves. The `r` in the
    # name (register) is misleading on Blackwell: the operand stays in shared
    # memory (an SMEM matrix descriptor). UMMA is warp-uniform, so every thread
    # sees the same view.
    tCrA = tiled_mma.make_fragment_A(sA)
    # (MMA, MMA_N, MMA_K)
    tCrB = tiled_mma.make_fragment_B(sB)
    # partition_shape_C returns a SHAPE (not a tensor): the accumulator tile
    # (bM, bN) partitioned per the MMA instruction layout → (MMA, MMA_M, MMA_N)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    # (MMA, MMA_M, MMA_N)
    # Turn the shape into a real accumulator tensor: same shape as acc_shape,
    # plus a layout (strides) and a pointer into TMEM. This is the MMA's C/D
    # operand — cute.gemm(tiled_mma, tCrA, tCrB, tCtAcc) accumulates each
    # K-tile's partial product into it; the epilogue later reads it back out
    # of TMEM. The `t` in the name (tmem) is accurate here, unlike the `r` in
    # tCrA/tCrB.
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)
```

Blackwell's Unified MMA (UMMA) differs from Ampere's MMA, which requires each thread to provide its own operand slice in registers. With UMMA, every thread issues the same MMA instruction referencing shared memory and TMEM — the hardware handles the data distribution internally. Because of this, `tiled_mma.get_slice(0)` uses thread 0's view to partition `gA`, `gB`, and `gC`, and every thread in the MMA warp sees the same partition.

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

SharedStorage is a class holding barriers for pipeline sync. The name of storage is misleading. It should be barrier_storage, or ctrl_storage. 

```python
sA = smem.allocate_tensor(layout=..., swizzle=...)
tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=threads_per_cta,
) # use hw barrier 1, and all threads in a cta for sync
tmem = utils.TmemAllocator(
    storage.tmem_holding_buf.ptr,
    barrier_for_retrieve=tmem_alloc_barrier,
)
num_tmem_cols = 512
tmem.allocate(num_tmem_cols)
```
Allocated Memory size can be calculated using layout.outer. Swizzle will not affect the total size needed.

tmem is a Tmem allocator. After allocating a Tmem, the Tmem address will be written into tem_holding_buffer, so other warps can read it. barrier is used to notify other warp that the Tmem is ready and can be read. 

Tmem is a special hardware. It has 128 rows and 512 columns. You can specify how many columns needed to contain accumulator (output tile in mma instruction). One Tmem cell is 4 bytes. 

In tmem.allocate, only lane0 thread in a warp emit instruction to allocate Tmem. Other threads in the CTA just wait for broadcast via tmem_holding_buf. 

```python
    if warp_idx == 0: # will generate bra ptx instruction, no warp divergence
        cpasync.prefetch_descriptor(tma_atom_a) # prefetch tma descriptor to cache in tma_atom_a
        cpasync.prefetch_descriptor(tma_atom_b) # prefetch tma descriptor to cache tma_atom_a
        cpasync.prefetch_descriptor(tma_atom_b) # prefetch tma descriptor to cache tma_atom_b
```
tma atom is a moving tool, which includes instruction and descriptor (a "map" telling TMA where is data, what's the data layout). prefetch the "map" to L1/L2 cache will accelerate data move. 
### TMA 
```python
    # Partition tensors for TMA; this consumes the MMA-partitioned tensors
    # (tCgA / sA) and re-cuts them into TMA copy units, so the leading mode of
    # each result is "one TMA transfer" and the rest are iteration modes.
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        0,                    # this CTA's rank in the multicast group (0 = no multicast)
        cute.make_layout(1),  # multicast CTA layout; size 1 = single CTA, no multicast
        # group_modes(t, start, end): coalesce modes [start, end) into ONE mode,
        # leaving the rest untouched. It's a logical reshape (no data moves).
        # sA / tCgA arrive as (MMA, MMA_M, MMA_K, ...); folding [0, 3) merges
        # (MMA, MMA_M, MMA_K) into a single mode -> a 2-mode (folded, rest)
        # tensor. TMA wants this because one transfer treats the whole tile as
        # one unit (the folded mode) and iterates over the rest (stages / k-tiles).
        cute.group_modes(sA, 0, 3),    # dest: smem A; (MMA, MMA_M, MMA_K) folded, last mode = stage
        cute.group_modes(tCgA, 0, 3),  # src:  gmem A; (MMA, MMA_M, MMA_K) folded, last mode = RestK
    )
    # tAsA -> dest view (smem), tAgA -> src view (gmem); return order follows
    # arg order. Naming: tAgA = [t]iled-for-the-[A]-copy, [g]mem, tensor [A];
    # tAsA is the [s]mem counterpart. Actual copy later:
    #   cute.copy(tma_atom_a, tAgA[None, k], tAsA[None, pipe])

    # CTA-wide sync before retrieving the pointer to the start of the allocated
    # TMEM. Only one warp (e.g. warp 0) does the allocation, so we must sync
    # before reading the TMEM start address.
    tmem.wait_for_alloc()
    # Every warp reads the allocated TMEM base address from its smem slot,
    # recast to an acc_dtype pointer.
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    # Swap in the real TMEM pointer: keep tCtAcc's layout, but point it at the
    # freshly allocated TMEM base instead of the placeholder from make_fragment_C.
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)
```
### software pipeling
```python
ab_stages = 4 # matrix A B 's shared memory ring buffer nums
acc_stage = 1 # there is only one TMEM accumulator buffer, 
# so only after epilogue write TMEM to global, MMA can use TMEM again for accumulation
....
num_tma_copy_bytes = cute.size_in_bytes(
        io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
    ) + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
# compile time calculation of one TMA move bytes, include A tile and B tile 
ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create（...)
# compile time pipeline producer consumer creation
acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(...)
```
There are two pipelines in this GEMM. For ACC pipeline, since acc_stages=1, it is not enabled.
By using producer, consumer abstract, simplifies synchronization. 

|          | AB pipeline (PipelineTmaUmma) | ACC pipeline (PipelineUmmaAsync) |
|----------|-------------------------------|----------------------------------|
| Producer | TMA warp                      | MMA warp                         |
| Consumer | MMA warp                      | Epilogue warps                   |
| Buffer   | Shared memory (A/B tiles)     | TMEM (accumulator)               |
| Stages   | ab_stages = 4                 | acc_stages = 1                   |
| tx_count | Yes (TMA hardware counting)   | No (software commit)             |

### epilog tiling
