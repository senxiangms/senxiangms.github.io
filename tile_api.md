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

TBD
