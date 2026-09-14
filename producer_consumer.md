---
layout: page
title: "Producer and consumer: why NPU programming needs an abstraction"
---

# Why an NPU needs a programming abstraction

Take the simplest kernel there is, `vec_c = vec_a + vec_b`.

Each AI core is responsible for one segment of the vectors. That segment usually
does not fit in the core's on-chip memory - an Ascend core has 192 KB of unified
buffer to work with, and a segment of a real tensor is measured in megabytes -
so the segment is split further into **tiles**, and the core walks them one at a
time:

```
move tile of a, b : global memory -> on-chip memory   (DMA engine)
add               : on-chip memory -> on-chip memory  (vector engine)
move tile of c    : on-chip memory -> global memory   (DMA engine)
```

Three stages, and they run on three different hardware units. That single fact
is what the rest of this article is about.

## Doing one stage at a time is correct, and slow

Nothing stops you from running the three stages strictly in order. It gives the
right answer. It also leaves two of the three engines idle at any moment:

```
DMA in    [in 0]              [in 1]              [in 2]
vector           [add 0]             [add 1]             [add 2]
DMA out                [out 0]             [out 1]             [out 2]
          |------ tile 0 ------|------ tile 1 ------|
```

The vector engine waits for the copy, the copy waits for the previous store.
On a memory-bound kernel like vector add - three memory round trips per element,
one add - the arithmetic is nearly free and the copies are the whole cost, so
this schedule spends most of its time with the DMA engines stopped.

What you want instead is for the copy of tile *i+1* to run **while** the vector
engine is still adding tile *i*:

```
DMA in    [in 0][in 1][in 2][in 3]
vector          [add 0][add 1][add 2]
DMA out               [out 0][out 1]
```

Same instructions, same results, roughly a third of the time. The stages have
become a software pipeline.

## The hardware will not do this for you

On an Ascend-style core the engines are independent instruction queues - call
them pipes: `MTE2` (global to on-chip), `V` (vector), `MTE3` (on-chip to
global), plus scalar and cube. The scalar unit dispatches instructions into
these queues in program order and does not wait for them; each queue executes
its own instructions in order, and the queues run concurrently with each other.

That concurrency is free, and it comes with no safety net. **There is no
interlock between pipes.** The hardware does not notice that the vector
instruction it is about to issue reads a buffer the DMA engine has not finished
filling. There is no scoreboard, no stall, no fault - just whatever bytes happen
to be in the buffer at that moment.

So a pipelined kernel has exactly two ways to go wrong, and both of them are
races on a tile buffer. They carry the usual architecture names, which are worth
reading carefully: a hazard is named after the order the program *requires*, and
the hazard is that concurrent execution delivers the opposite one.

| hazard | the program requires | the race delivers |
|---|---|---|
| read-after-write | copy-in writes the tile, then the add reads it | the add reads the buffer before the copy-in has landed, and gets the previous tile |
| write-after-read | the add reads the tile, then copy-in of tile i+1 overwrites it | the copy-in overwrites a buffer the add - or the copy-out - is still reading |

Both produce plausible-looking wrong numbers, intermittently, on one core out of
many. Neither can be reproduced in a CPU simulator, because there the stages
really do run one at a time.

The primitive that fixes this is an explicit event flag: `SetFlag` on the
producing pipe, which does not block, and `WaitFlag` on the consuming pipe,
which blocks only that pipe. Written out by hand, one tile of the pipeline looks
like this:

```cpp
// ---- MTE2 ----------------------------------------------------
copy_in(tile);
ev_in = alloc_event_id(MTE2_V);
set_flag (MTE2_V, ev_in);        // publish; does not block MTE2

// ---- V -------------------------------------------------------
wait_flag(MTE2_V, ev_in);        // blocks V until the copy landed
free_event_id(MTE2_V, ev_in);
add(tile);
ev_out = alloc_event_id(V_MTE3);
set_flag (V_MTE3, ev_out);       // publish; does not block V

// ---- MTE3 ----------------------------------------------------
wait_flag(V_MTE3, ev_out);       // blocks MTE3 until the add is done
free_event_id(V_MTE3, ev_out);
copy_out(tile);
```

Each block is one pipe, and each one opens with the wait that lets it start and
closes with the flag that releases the next - which is the shape the queue will
turn out to have. The `alloc_event_id` / `free_event_id` calls are scalar-side
bookkeeping rather than instructions on those pipes; they are grouped with the
stage that borrows and returns the flag register.

...and that is only the forward direction, for one buffer, ignoring the
backward edge that tells the copy-in it may reuse the buffer. Multiply by three
buffers, by every tile of the loop, by every kernel you will ever write. The
flag registers are a fixed hardware resource too - a handful per event type - so
they have to be allocated and released, and a leaked one deadlocks the core.

This is why device programming at this level is error-prone: the compiler
cannot check it, the simulator cannot reproduce it, and the failure mode is
silent.

## Overlap is bought with memory, not with instructions

One more thing the flags alone will not give you. Put the events in, keep a
single buffer per stage, and the pipeline still runs serially - because the
copy-in of tile *i+1* has nowhere to write until the add of tile *i* has
released the one buffer there is.

Overlap requires a **second** buffer, so that the producer can be filling one
while the consumer drains the other. That is all double buffering is, and it
sets the terms of the trade: the depth of the pipeline is the number of
buffers, and buffers come out of a 192 KB budget. Deeper pipeline, smaller
tiles. This is the central tuning knob of an NPU kernel, and it is a property of
how the buffers are managed - not of the code that computes on them.

## What the abstraction has to provide

Look at the three stages again and the shape is familiar: each adjacent pair is
a **producer and a consumer** sharing a buffer. Copy-in produces what the add
consumes; the add produces what the copy-out consumes. The classical answer to
that shape is a queue, and it is the right answer here, with one addition - the
queue must also hand the empty buffer *back*, because on-chip memory is far too
scarce to allocate a fresh one per tile.

Which gives four operations, and their mapping onto the hardware primitives:

| the programmer writes | the queue emits | meaning |
|---|---|---|
| `AllocTensor` | `WaitFlag<free>` | wait until the consumer has released this slot |
| `EnQue` | `SetFlag<full>` | the producer has filled it |
| `DeQue` | `WaitFlag<full>` | ...so the consumer may read it |
| `FreeTensor` | `SetFlag<free>` | the consumer is done, recycle it |

Two flags forward, two flags backward, per buffer. The forward pair is
correctness - no read before the write lands. The backward pair is back
pressure - no overwrite before the read finishes, and no runaway producer. The
number of buffers behind the queue is the pipeline depth, and it is the one
number the programmer still tunes.

In Ascend C this is `TQue`, carved out of on-chip memory by a `TPipe`, handing
out `LocalTensor` views of tiles that `GlobalTensor` supplies from global
memory. A kernel written against it contains no event, no flag, no id - and
switching from a serial pipeline to a double-buffered one changes exactly one
constant.

That is the whole argument for the abstraction: it is not that the events are
hard to write. It is that they are impossible to review, and the queue makes
them structural.

*A working miniature of `TPipe` / `TQue` / `GlobalTensor` / `LocalTensor` that
runs on the host, with a race checker that reports the missing flags, is in
[`src/TPipe`](https://github.com/senxiangms/senxiangms.github.io/tree/tpipe/src/TPipe)
of this site's repository.*
