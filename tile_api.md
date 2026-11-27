---
layout: page
title: "A Tile-level API for NPU kernels"
---

# What sits between intrinsics and a DSL

Writing an NPU kernel against raw builtins means filling in hardware
configuration structures by hand: source and destination storage ids, base
addresses in the unit the engine happens to encode, per-dimension strides and
lengths, fractal sizes, padding modes, barrier ids. The arithmetic is trivial;
the bookkeeping is where the bugs are, and none of it is checked — a stride
expressed in elements instead of 128-byte units compiles perfectly and copies
the wrong bytes.

The other extreme is a tile DSL such as TileLang, where you describe the
computation and the compiler picks layouts, placement and pipelining. That is
the right level for most kernels, but it needs a backend, and the backend still
has to emit the configuration images described above.

A **tile-level API** sits in between. It is a header-only C++ library whose
vocabulary is a *tile* — a rectangular piece of a tensor that lives in one
named on-chip buffer — and a handful of instructions over tiles: bind, move,
add, matmul. All layout information is a template parameter, so every shape,
stride and alignment rule is a `static_assert`, and the generated code is the
same register image you would have written by hand. This article describes what
such an API looks like: how a tile is defined, how memory location is part of
its type, how data movement is expressed, and what the vector instruction
interface looks like. The examples use the namespace `tile_api` and a
representative NPU core with a vector engine, a tensor (matmul) engine and two
DMA engines.

# The tile

```cpp
template <TileType Loc, typename Element, int Rows, int Cols,
          OuterLayout OuterFractal = OuterLayout::RowMajor,
          int RowValid = Rows, int ColValid = Cols,
          InnerLayout InnerFractal = InnerLayout::NoneBox,
          int InnerFractalSize = TileConfig::fractalABSize,
          auto PadVal = PadValue::Null,
          CompactMode Compact = CompactMode::Null>
struct Tile;
```

Two things are worth noticing immediately.

**A tile does not own storage.** It is not an array, and declaring one
allocates nothing. The only runtime state inside a `Tile` object is its
address — a byte offset inside the buffer named by `Loc` — plus, for a remote
tile, one extra index discussed below. Everything else is a compile-time
constant. That is what makes the API free: after inlining, a tile instruction
is a sequence of constants handed to a builtin.

**The valid region is part of the type.** `Rows x Cols` is the *capacity*
shape — the rectangle the tile occupies in memory — while
`RowValid x ColValid` is the part that actually holds data. The elements in
between are padding, and `PadVal` says what they are (nothing, zero, the type's
max or min, or a custom value). Splitting the two is what lets a kernel handle
a tensor whose last tile is ragged without changing its addressing: the
capacity shape keeps strides constant, the valid shape shrinks the transfer.
The valid shape must be static, because it is encoded into the engine
configuration.

Layout comes in two levels. `OuterFractal` is the order of the tile as a plain
matrix, row- or column-major. `InnerFractal` describes the order *inside* a
small box, the fractal that the tensor engine consumes — typically 16x16
elements, 512 bytes for a 16-bit type. `NoneBox` means a plain 2-D layout, the
usual choice for vector data; a boxed layout is what a matmul operand looks
like:

A non-boxed tile is the layout everyone already has in mind: with
`OuterLayout::RowMajor`, a whole row of elements is contiguous, and the next
row starts `Cols` elements later.

```text
NoneBox, RowMajor  —  32 x 48 fp16
┌────────────────────────────────┐
│ row 0 ───────────────────────► │  48 contiguous elements, then row 1
│ row 1 ───────────────────────► │
│ ...                            │
└────────────────────────────────┘
```

A boxed tile has two orders, and keeping them apart is the whole trick. The
same 32 x 48 tile, with `InnerFractal` set and 16x16 boxes, is a 2 x 3 grid of
boxes. Each box is 256 elements — 512 bytes for fp16 — stored contiguously;
the *outer* layout only decides in which order those 512-byte blocks follow
each other in memory.

```text
box grid (position)              where each box starts (outer ColMajor)
┌───────┬───────┬───────┐
│ (0,0) │ (0,1) │ (0,2) │        offset  0   512  1024  1536  2048  2560
├───────┼───────┼───────┤        box   (0,0) (1,0) (0,1) (1,1) (0,2) (1,2)
│ (1,0) │ (1,1) │ (1,2) │                └─col 0─┘ └─col 1─┘ └─col 2─┘
└───────┴───────┴───────┘
```

So with outer `ColMajor`, the box that immediately follows `(0,0)` in memory is
the one **below** it, `(1,0)` — not the one to its right. The tile walks a full
column of boxes, then jumps to the top of the next column. With outer
`RowMajor` the order would be `(0,0) (0,1) (0,2) (1,0) ...` instead.

Inside a box the elements are laid out by `InnerFractal`: `RowMajor` means
16 contiguous elements per box row, `ColMajor` means 16 per box column. Outer
and inner are chosen independently — a left-hand matmul operand is typically
outer ColMajor with inner RowMajor, a right-hand one ColMajor/ColMajor.

From these parameters the tile derives `RowStride`, `ColStride`, `InnerRows`,
`InnerCols` and so on, and asserts the rules the hardware imposes — the
contiguous dimension of a non-boxed tile must be a multiple of 32 bytes, the
shape must be divisible by the box shape, the valid region must fit inside the
capacity. All at compile time, with the offending tile type in the message.

For matmul operands the layouts are fixed by the tensor engine, so the API
gives them names:

| Alias | Location | Layout |
|-------|----------|--------|
| `TileLeft<T, M, K>`  | L2     | outer ColMajor, box RowMajor |
| `TileRight<T, K, N>` | Weight | outer ColMajor, box ColMajor |
| `TileAcc<T, M, N>`   | L2     | like `TileLeft`, 1024-byte box (fp32) |

A kernel author writes `TileLeft<_Float16, 64, 64> a;` and does not have to
remember which of the four combinations the engine wants.

# Where a tile lives

`TileType` is the first template parameter because on an NPU *where* data sits
is not an implementation detail — it decides which engine can touch it and
which instruction can move it.

| `TileType` | Storage |
|------------|---------|
| `VecL1`    | vector engine L1 buffer, the only memory vector instructions read and write |
| `L2`       | the core's L2, shared by all engines of that core |
| `TCM`      | the core's TCM, byte addressed, where a host-side DMA lands |
| `NocTCM`   | another core's TCM, reached over the NoC |
| `NocL2`    | another core's L2, reached over the NoC |
| `Weight`   | tensor engine weight buffer |
| `LUT`      | lookup table buffer on the tensor output path |
| `Bias`     | tensor engine bias buffer |

Because the location is in the type, "vector instructions only work on L1" and
"the right-hand matmul operand must be in the weight buffer" are not comments
in a manual — they are overload constraints. Passing an `L2` tile to a vector
add is a compile error, not a silent read of whatever the engine's L1 happened
to contain.

## There is no global memory

A PTO-style API has a `GlobalTensor`: a pointer into a memory every core
shares. A pointer alone names the data, and the load instruction takes a
pointer for the source and a tile for the destination.

A NUMA chip has no such memory. Each core owns its TCM and its L2; an address
is only meaningful *inside one core*. So the source of a transfer is a tile
too, and a remote tile carries one more piece of state: which core it belongs
to.

That is `addrIdSlot`. The natural design would be to store a core id in the
tile, but DMA configurations typically have no field for one — they have a
small (say 4-bit) `addr_id` that indexes a 16-entry table in the issuing core,
whose entries hold physical core ids and are filled at run time:

```cpp
set_addr_id(/*slot=*/3, peerPhysicalCoreId);   // table entry, runtime
TASSIGN(remote, 0x4000, /*addrIdSlot=*/3);     // tile binding, compile time
```

The indirection is not an inconvenience, it is the point: the DMA
configuration stays a compile-time constant while the core it talks to changes
at run time. The API therefore exposes the slot, not a core id, and the
documentation has to say so loudly — passing a core id where a slot is
expected reaches whichever core slot *n* happens to hold, and the transfer
succeeds.

Two consequences for the kernel author: fill the slot before the transfer
runs, and do not refill it while a transfer using it is still pending.

# Binding and moving

## `TASSIGN` — bind, do not copy

```cpp
TASSIGN(tile, addr);                 // byte offset inside Loc
TASSIGN(tile, addr, addrIdSlot);     // remote tile: offset + core slot
TASSIGN<Addr>(tile);                 // static address, fully checked
```

Binding moves no data; it is the moment a compile-time shape acquires a
runtime address. The templated form is the one to prefer: with the address as
a template parameter, the API can check that the tile's full capacity fits in
the buffer (16 KiB of L1, 256 KiB of L2, 64 KiB of weight buffer), that the
address is in range, and that it meets the base alignment of that storage.
With a runtime address none of that is possible, and the burden moves back to
the caller.

Base alignment is a floor, not the whole story: each engine encodes addresses
in its own unit — DMA encodes a local L2 address in 128-byte units, the vector
engine encodes L1 in 64-byte units, the tensor engine encodes weights in
512-byte units. The practical rule is to give every tile the strictest
alignment it will ever need rather than reason case by case.

## `TLOAD` / `TSTORE` — one transfer, one task

```cpp
AGTask ld = TLOAD(l2Tile, tcmTile, deps...);    // into L2 / Weight / LUT / Bias
AGTask st = TSTORE(tcmTile, l2Tile, deps...);   // L2 back out to TCM
```

Both take a destination tile and a source tile with the same element type and
the same valid shape — the capacity shapes may differ, which is how a
sub-rectangle of a larger buffer is moved. Only the valid region travels; what
is outside it at the destination keeps its old contents.

Whether the hardware can do the copy at all depends on the shape of the valid
region:

```text
contiguous valid region          strided valid region
┌───────────────────┐            ┌───────────────────┐
│███████████████████│            │█████████░░░░░░░░░░│  segment
│███████████████████│            │█████████░░░░░░░░░░│  segment
│░░░░░░░░░░░░░░░░░░░│            │█████████░░░░░░░░░░│  segment
└───────────────────┘            └───────────────────┘
  one linear transfer,             remap: segment size AND
  total % 128B == 0                row stride % 128B == 0
```

The first case lowers to a linear DMA, the second to a strided (remap)
descriptor, and the 128-byte granularity rules are different for the two. Both
are `static_assert`s on the tile types, which is the single largest practical
win of the whole API: the most common way to lose a day on this hardware is a
transfer whose stride is legal in elements and illegal in bytes.

Each call submits one task and returns a handle. There is no "append" for DMA
because one DMA configuration holds exactly one transfer — a difference from
vector instructions that the API deliberately does not paper over.

## Tasks and dependencies

Engines run concurrently, and the scalar core only *issues* work. Ordering is
expressed by passing task handles into later submissions:

```cpp
AGTask t0 = TLOAD(dst0, src0);          // no deps: t0 and t1 may run together
AGTask t1 = TLOAD(dst1, src1);
AGTask t2 = TLOAD(dst2, src2, t0, t1);  // waits for both
AGTask t3 = TLOAD(dst3, src3, t2);      // chained
TWAIT(t3);                              // scalar blocks here
```

This is the same producer/consumer pipeline described in
[Producer and consumer](producer_consumer), expressed as a dependency graph
instead of barriers. Two rules carry most of the weight in practice: `TWAIT`
is the only place the scalar core blocks, so wait as late as possible; and when
an L2 buffer is reused, the next `TLOAD` into it must depend on the `TSTORE`
that drained it, or the DMA engines will race over the same bytes.

# The vector tile API

Vector instructions differ from DMA in one structural way: the engine runs a
*program*, a sequence of up to 32 instructions, from a single configuration.
Giving each `TADD` its own task would waste that. So vector instructions append
to a container, and the container is what gets submitted:

```cpp
VecKernel k;
TADD(k, dst, src0, src1);
TADD(k, dst2, dst, src2);
AGTask t = k.submit(ld0, ld1);   // one task, runs after both loads
TWAIT(t);
```

`VecKernel` is a compile-time object: `append` produces no runtime code, it
builds the configuration image that ends up in the task-graph blob the engine
executes. Hence the one constraint that surprises people: **tile addresses used
by vector instructions must fold to constants after inlining.** The address is
a field in the configuration, so it cannot come from a runtime variable. In
practice that means binding L1 tiles with `TASSIGN<Addr>` or with a `constexpr`
address, never with a loop-dependent one.

Dependencies apply to the whole kernel — nothing in it starts until every task
passed to `submit()` has completed. If one instruction only needs an early
load, it belongs in a separate, earlier kernel. Grouping is a scheduling
decision, and the API leaves it to the author instead of guessing.

`TADD(kernel, dst, src0, src1)` is representative of the element-wise family.
Its preconditions are all checked:

- all three tiles in `VecL1`, since that is the only memory the engine reads;
- same element type, same valid shape, row-major and non-boxed;
- column count a multiple of 16, the engine's lane width;
- addresses 64-byte aligned;
- `dst` not sharing an L1 bank with `src0` or `src1` — a bank conflict here is
  not a slowdown, it is wrong data.

The last one is the kind of rule that is impossible to remember and trivial to
encode once.

Matmul sits on the other side of the split. One tensor configuration performs
one multiply, so `TMATMUL` behaves like `TLOAD`, not like `TADD`:

```cpp
TileRight<_Float16, 64, 64> b;         // weight buffer
AGTask lw = TLOAD(b, bInTcm);          // weights must be resident first
AGTask mm = TMATMUL(c, a, b, lw);      // c = a * b, c and a in L2
```

When the tensor engine reads A from and writes C to L2 directly there are no
L0A/L0B/L0C staging buffers to fill, so no `TMOV` step appears around the
multiply — a good example of why a tile API is worth writing per architecture
rather than copying wholesale from another chip's ISA library.

# Putting it together

A complete kernel: load two tiles from TCM into L2, add them in vector L1,
store the result back.

```cpp
#include <tile_api/instr.hpp>
using namespace tile_api;

constexpr int Rows = 16, Cols = 64;
constexpr unsigned Bytes = Rows * Cols * sizeof(float);

using TcmTile = Tile<TileType::TCM,   float, Rows, Cols>;
using L1Tile  = Tile<TileType::VecL1, float, Rows, Cols>;

int main() {
  TcmTile a, b, out;
  L1Tile  x, y, z;

  TASSIGN(a,   0x10000);
  TASSIGN(b,   0x10000 + Bytes);
  TASSIGN(out, 0x10000 + 2 * Bytes);
  TASSIGN<0 * Bytes>(x);            // checked: fits L1, 64-byte aligned
  TASSIGN<1 * Bytes>(y);
  TASSIGN<2 * Bytes>(z);

  AGTask l0 = TLOAD(x, a);
  AGTask l1 = TLOAD(y, b);

  VecKernel k;
  TADD(k, z, x, y);
  AGTask add = k.submit(l0, l1);

  AGTask st = TSTORE(out, z, add);
  TWAIT(st);
  return 0;
}
```

Twenty lines, no register fields, and every hardware rule in it is enforced by
the compiler. Unroll it over a stream of tiles, keep two sets of buffers and
let the dependency handles overlap the loads with the adds, and this is the
software pipeline from the producer/consumer article — written in terms of
tiles rather than barriers.

# What the level is good for

A tile API is not a replacement for a tile DSL. It does not choose tile sizes,
it does not allocate buffers, it does not pipeline for you. What it does is
turn the chip's configuration space into a typed vocabulary, so that:

- a kernel is readable — `TLOAD(l2, tcm)` instead of twelve register writes;
- the hardware's rules fail at compile time, where they are cheap;
- the generated code is identical to the hand-written image, because every
  layout decision was a template parameter;
- a DSL backend has something to emit that is not raw builtins.

That last point is the strategic one. Writing the tile layer first gives the
compiler team a target that is already validated by hand-written kernels, and
gives the kernel team something usable long before the DSL exists.
