# TPipe, TQue, GlobalTensor, LocalTensor - reimplemented for the host

A working miniature of the Ascend C programming model in ~1400 lines of plain
C++17, plus a vector add kernel that runs on it. No NPU, no CANN, no toolchain:
`./build.sh --run` and you have a trace of every instruction the kernel issues
and which pipe issued it.

The point is not to emulate an Ascend core. It is to answer one question -
**what does `TQue` actually do?** - by writing the part that matters and leaving
out the part that doesn't. What it does is insert four flag operations around
your buffers, and the interesting consequence is that a kernel which forgets
them still produces the right numbers on a simulator. So this implementation
also ships a race checker, and one of the examples is deliberately broken.

## Build

```bash
./build.sh --run            # g++ -std=c++17, nothing else

# or
cmake -S . -B build && cmake --build build && ctest --test-dir build
```

Three examples:

| | |
|---|---|
| `examples/add_simple.cpp` | one tile per core - a port of the official `add_tpipe_tque` sample |
| `examples/add_double_buffer.cpp` | tiled and pipelined, `BUFFER_NUM` 1 vs 2, same source text |
| `examples/sync_check.cpp` | the same add written four ways, one of them wrong |

## The machine, in three tables

A kernel runs on one **core** at a time (`GetBlockIdx()` / `GetBlockNum()`), and
a core has its own local memories plus several **pipes** - instruction queues
that issue in program order individually, but run concurrently with each other.

| pipe | does |
|---|---|
| `S` | scalar arithmetic, address computation, `GetValue`/`SetValue` |
| `V` | vector instructions: `Add`, `Mul`, `Duplicate`, ... |
| `M` | cube (matmul) |
| `MTE1` | L1 -> L0A/L0B/UB |
| `MTE2` | GM -> UB/L1/L0 |
| `MTE3` | UB -> GM/L1 |

A tensor's **position** is logical; the compiler maps it to a physical memory:

| TPosition | Hardware | size |
|---|---|---|
| `VECIN`, `VECOUT`, `VECCALC`, `CO2` | UB | 192 KB |
| `A1`, `B1`, `C1`, `TSCM` | L1 | 512 KB |
| `A2` / `B2` | L0A / L0B | 64 KB |
| `CO1` | L0C | 128 KB |

And a position is really an *edge*, which is what picks the events - `VECIN`
means "GM to UB", so it is filled by MTE2 and drained by V:

| queue | src -> dst | enQueEvt | freeBufEvt |
|---|---|---|---|
| `TQue<VECIN, n>` | GM -> UB | `MTE2_V` | `V_MTE2` |
| `TQue<VECOUT, n>` | UB -> GM | `V_MTE3` | `MTE3_V` |
| `TQue<A1, n>` | GM -> L1 | `MTE2_MTE1` | `MTE1_MTE2` |
| `TQue<A2, n>` | L1 -> L0A | `MTE1_M` | `M_MTE1` |

## What TQue does

Four queue calls, four flag operations. That is the entire abstraction:

| you write | TQue emits | meaning |
|---|---|---|
| `AllocTensor<T>()` | `WaitFlag<freeBufEvt>` | wait until the consumer has finished with this slot |
| `EnQue(t)` | `SetFlag<enQueEvt>` | the producer has filled it |
| `DeQue<T>()` | `WaitFlag<enQueEvt>` | ...so the consumer may read it |
| `FreeTensor(t)` | `SetFlag<freeBufEvt>` | the consumer is done, recycle it |

`SetFlag` is non-blocking and `WaitFlag` blocks only the consumer pipe, so the
producer runs ahead as far as the number of buffers allows. Run `add_simple`
and the whole thing is in the trace:

```
  [core 0][S   ] InitBuffer VECIN x1 x8192B -> ub[0x00000, 0x02000)
  [core 0][MTE2] DataCopy gm -> ub[0x00000] 8192 B
  [core 0][MTE2] SetFlag<MTE2_V>(0)          <- EnQue(xLocal)
  [core 0][V   ] WaitFlag<MTE2_V>(0)         <- inQueueX.DeQue<float>()
  [core 0][V   ] Add ub[0x04000] = f(0x00000, 0x02000) 2048 elems (32 repeats)
  [core 0][V   ] SetFlag<V_MTE3>(0)          <- outQueueZ.EnQue(zLocal)
  [core 0][V   ] SetFlag<V_MTE2>(0)          <- inQueueX.FreeTensor(xLocal)
  [core 0][MTE3] WaitFlag<V_MTE3>(0)         <- outQueueZ.DeQue<float>()
  [core 0][MTE3] DataCopy ub[0x04000] -> gm 8192 B
```

Nowhere in that kernel is there an event, an id, or a barrier.

## What double buffering is

`add_double_buffer` compiles the *same kernel source* twice, once with
`BUFFER_NUM = 1` and once with `2` (`examples/add_kernel.inc`, included into two
namespaces so it is provably the same text). With one buffer, `CopyIn(i+1)`
cannot start until `Compute(i)` has released the buffer, and the trace shows the
wait sitting right before the copy:

```
  [core 0][MTE3] DataCopy ub[0x00200] -> gm 256 B     <- tile 0 finishes
  [core 0][MTE2] WaitFlag<V_MTE2>(0)                  <- tile 1 blocked here
  [core 0][MTE2] DataCopy gm -> ub[0x00000] 256 B
```

With two, `AllocTensor` hands out the other slot, which nobody is waiting on,
so MTE2 issues tile 1 immediately and runs a whole tile ahead of V:

```
  [core 0][MTE3] DataCopy ub[0x00400] -> gm 256 B     <- tile 0 finishes
  [core 0][MTE2] DataCopy gm -> ub[0x00100] 256 B     <- tile 1, no wait at all
```

Cost: twice the UB. Benefit: the copy of tile i+1 overlaps the compute of tile
i. The kernel body did not change, and that is the part worth internalising -
the pipelining is a property of the queue, not of the code that uses it.

## The race checker

Since everything here executes in program order, a missing flag cannot produce
a wrong result - which would make this a useless teaching tool. So instead of
simulating the concurrency, the runtime checks it, with vector clocks:

- every pipe has a clock, ticked once per instruction;
- every 32-byte granule of local memory remembers who last wrote it and at which
  tick, and who last read it;
- `SetFlag` snapshots the producer's view of all clocks; `WaitFlag` merges that
  snapshot into the consumer's view;
- an access from a pipe that has not observed the last writer's tick is a
  missing `WaitFlag`, and the checker says which one.

```
!! sync error (core 0): Add on pipe V touches ub[0x00000] last written by MTE2
   at tick 1, but V has only observed MTE2 up to tick 0 - missing WaitFlag<MTE2_V>
```

Write-after-read is caught the same way, which is the failure mode of a
too-small buffer count: MTE2 overwriting a tile that V is still reading.

`sync_check` runs all of it: no flags (3 reports, correct numbers anyway),
manual flags, `TQue`, and a forgotten `FreeTensor` - caught not by the clocks
but by the buffer state machine, when `TPipe` is destroyed with a slot still in
`DEQUE`.

## Faithful / not faithful

Faithful, because these are the things that change how you write a kernel:

- the `TPosition` -> `Hardware` -> event mapping, taken from the same tables the
  real `GetBufferLogicPos` / `GetQueEvt` use;
- the buffer state machine (`FREE` -> `OCCUPIED` -> `ENQUE` -> `DEQUE`) and
  where each flag is raised and waited;
- event ids are allocated from a pool of 8 per event type and released on wait,
  as on hardware - a queue that never dequeues runs out;
- `TPipe` is a bump allocator with no free list, so `InitBuffer` order decides
  the UB layout and total UB is 192 KB;
- 32-byte alignment on every DMA, and the 256-byte vector repeat shown in the
  trace.

Not faithful, deliberately:

- one core at a time, blocks run sequentially, no AIC/AIV split and no cube
  path (the L1/L0 positions exist and pick the right events, but nothing
  computes on them);
- `DataCopy` is a `memcpy` - no `DataCopyParams`, no strides, no nd2nz;
- vector instructions take an element count, not a mask/repeat triple;
- `half`/`bfloat16` are absent; examples use `float`;
- `TBufPool`, `TSCM`, matmul and the `Std::tuple` `InitBuffer` overloads are out
  of scope.

## Gotchas worth knowing

- **`InitBuffer(que, num, len)` takes bytes**, and `len` must be a multiple of
  32. Pass `tileLength * sizeof(T)`, not `tileLength`.
- **`num` is the buffer count, the template argument is the queue depth.** They
  are usually equal, and `num > depth` is rejected.
- **A `LocalTensor` is not a pointer** - it is `(position, offset, length)`, and
  a slice `t[k]` keeps the buffer handle so `FreeTensor(t[k])` still works.
- **`FreeTensor` is what unblocks the producer**, so free the inputs right after
  the compute, not at the end of the iteration.
- **`TBuf` gives you memory and nothing else.** If two pipes touch it you own
  the flags.

## Where this lives in CANN

The real implementation, if you want to read it next: `kernel_tpipe.h` declares
`TPipe`/`TQueBind`/`TQue`/`TBuf`; `kernel_tquebind_impl.h` has `AllocBuffer`,
`EnQue`, `DeQue`, `FreeBuffer` with the same four flag operations wrapped in a
lot of architecture `#if`s; `kernel_event.h` has `GetBufferLogicPos` and
`GetQueEvt`, the two tables this file copies; `kernel_tensor.h` has the tensors.
