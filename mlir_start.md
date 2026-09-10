---
layout: page
title: How to create new device programming language by leveraging MLIR/CLang to bringup new AI chip
---

# Creating a Device Programming Language with MLIR and Clang

Silicon bring-up has two halves. The hardware half ends when the first wafer returns and a diagnostic blinks an LED. The software half ends when someone outside your team writes a kernel, compiles it, and gets the number they expected. The second half is longer, and it is almost entirely a compiler problem.

The temptation at the start is to write an assembler, hand it to the kernel team, and promise a "real compiler later." That works for the first three kernels and collapses at the thirtieth: every scheduling decision, every double-buffer, every bank-conflict workaround is now hand-encoded in source files nobody can regenerate when the ISA revs at tape-out 2. The alternative — fork LLVM and write a full backend before anything runs — spends nine months before the first `matmul` executes.

MLIR exists precisely to make the middle path viable. You define the abstractions your hardware actually has, you reuse everything above and below them, and you get a language that runs on day 60 instead of day 300. This article walks the whole path: what to build, what to reuse, in what order, and where the honest costs are.

## What "a programming language" actually means here

Before writing any TableGen, be precise about the deliverable. A device programming language for an accelerator is five separable artifacts, and conflating them is the most common planning error:

1. **A surface syntax** — what the kernel author types. C++ with attributes, a Python-embedded DSL, or a standalone grammar.
2. **A memory and execution model** — how many address spaces, what is coherent with what, what a "thread" is (if anything), how synchronization is expressed.
3. **A lowering pipeline** — the sequence of IR transformations from surface syntax to machine code.
4. **A code generator** — the thing that emits bytes the chip executes.
5. **A host runtime and ABI** — how a kernel is launched, how arguments are marshalled, how the binary is packaged.

MLIR gives you enormous leverage on (3). Clang gives you (1) and part of (5) nearly for free if you can live inside C/C++. Items (2) and (4) are genuinely yours, and no amount of infrastructure will write them for you. Plan your headcount accordingly.

## The shape of a typical AI chip, and why generic IR fights you

The reason you need a new language at all is that a modern accelerator violates the assumptions baked into LLVM IR. A representative NPU has:

- **A software-managed memory hierarchy.** Explicit scratchpads (L1 per-core, L2 shared), no coherent cache, DMA engines that move tiles asynchronously. There is no `load` that "just works" — the compiler or the programmer must stage data.
- **Coarse-grained instructions.** The ISA's unit of work is not `fmul` but "multiply a 128×128 tile by a 128×256 tile and accumulate into the tile register file." An instruction takes microseconds, not nanoseconds.
- **Explicit asynchrony.** DMA, compute, and the vector unit run concurrently and are ordered by semaphores or barriers the program manages by hand.
- **Layout as a first-class property.** Tiles are swizzled, blocked, or interleaved for the sake of a specific engine, and a "reshape" can be free or catastrophically expensive depending on which.

LLVM IR can express all of this — with target intrinsics, address spaces, and inline assembly — but it cannot *reason* about it. Once a DMA is an opaque intrinsic call, no LLVM pass will double-buffer it, and no LLVM pass will fuse two tile ops because it does not know they are tile ops.

MLIR's answer is that you should not lower to LLVM IR until you no longer need to reason. Keep your own operations, with your own semantics, for as long as the interesting optimizations live there.

## Layering: what to build and what to inherit

Design your dialect stack top-down as a sequence of narrowing abstractions. A workable four-level stack:

| Level | Dialect | Owner | What it expresses |
|-------|---------|-------|-------------------|
| L3 — framework | `linalg`, `tensor`, `stablehlo` | upstream | value-semantic tensor ops; where fusion and tiling happen |
| L2 — language | `mychip` (yours) | you | tiles, scratchpad allocation, `matmul` on tiles, layouts |
| L1 — device | `mychip_ll` (yours) | you | DMA descriptors, semaphores, engine-specific instructions, register file |
| L0 — machine | `llvm`, `emitc`, or your own | mostly upstream | what the code generator consumes |

The split between L2 and L1 is the one that earns its keep. L2 is what the user's program means; L1 is what the machine does. Tiling and fusion run at L3/L2; pipelining, semaphore insertion, and allocation run at L1. If you collapse them into one dialect you will find yourself pattern-matching on "is this DMA op still abstract or already concrete?" — a smell that always means a missing level.

Note also what you are *inheriting*: `scf` for structured control flow, `memref` for buffers and strides, `affine` if you want polyhedral analysis, `vector` for SIMD-ish work, `transform` for schedule-as-IR, plus bufferization, canonicalization, CSE, and the whole conversion framework. That is several engineer-years you do not spend.

## Defining your dialect

Everything user-facing is declared in ODS (TableGen). Start with the dialect and a type, because your type system is what makes the dialect worth having.

```tablegen
// include/MyChip/MyChipDialect.td
def MyChip_Dialect : Dialect {
  let name = "mychip";
  let cppNamespace = "::mlir::mychip";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let dependentDialects = ["memref::MemRefDialect", "scf::SCFDialect"];
}

class MyChip_Op<string mnemonic, list<Trait> traits = []>
    : Op<MyChip_Dialect, mnemonic, traits>;

// A tile living in a named memory level, with an explicit hardware layout.
def MyChip_TileType : TypeDef<MyChip_Dialect, "Tile"> {
  let mnemonic = "tile";
  let parameters = (ins
      ArrayRefParameter<"int64_t">:$shape,
      "Type":$elementType,
      "MemLevelAttr":$level,     // L1 | L2 | DRAM
      "LayoutAttr":$layout);     // ROW | ZN | SWIZZLE_XOR32
  let assemblyFormat = "`<` $shape `x` $elementType `,` $level `,` $layout `>`";
}
```

Encoding `level` and `layout` *in the type* rather than in an attribute on each op is the single highest-leverage decision in the whole design. It means the verifier catches "you fed an L2 tile to an engine that only reads L1" at compile time, and it means a layout conversion has to be a visible op — you can count them, and you can write a pass that eliminates them. Chips whose compilers track layout as an informal side-table end up with silent correctness bugs that appear only at certain shapes.

Now the operations:

```tablegen
def MyChip_AllocOp : MyChip_Op<"alloc", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Reserve a tile in a software-managed memory level";
  let arguments = (ins);
  let results   = (outs MyChip_TileType:$tile);
  let assemblyFormat = "attr-dict `:` type($tile)";
}

def MyChip_DmaOp : MyChip_Op<"dma", [AttrSizedOperandSegments]> {
  let summary = "Asynchronous tile copy between memory levels";
  let arguments = (ins MyChip_TileType:$src,
                       MyChip_TileType:$dst,
                       OptionalAttr<I32Attr>:$channel);
  let results   = (outs MyChip_TokenType:$token);   // completion handle
  let assemblyFormat =
      "$src `->` $dst attr-dict `:` type($src) `to` type($dst)";
}

def MyChip_WaitOp : MyChip_Op<"wait"> {
  let arguments = (ins Variadic<MyChip_TokenType>:$tokens);
  let assemblyFormat = "$tokens attr-dict";
}

def MyChip_MatmulOp : MyChip_Op<"matmul", [SameOperandsAndResultElementType]> {
  let summary = "Tile-granularity MMA with accumulation";
  let arguments = (ins MyChip_TileType:$a,
                       MyChip_TileType:$b,
                       MyChip_TileType:$acc,
                       DefaultValuedAttr<BoolAttr, "false">:$transposeB);
  let results   = (outs MyChip_TileType:$out);
  let hasVerifier = 1;
}
```

Two things to notice. First, `dma` returns a **token** rather than being a side-effecting void call. Asynchrony modelled as SSA data flow means the existing dependency machinery does your ordering analysis: a `wait` that consumes a token cannot be hoisted above the `dma` that produced it, for free, in every pass you will ever write. Modelling it as an opaque effect instead forces you to reimplement dependence analysis by hand.

Second, `hasVerifier = 1` on `matmul` is where you encode the ISA's real constraints — K must be a multiple of 32, `acc` must live in the accumulator file, `transposeB` is illegal for int8. Write these on day one. Every constraint not in the verifier becomes a hardware hang discovered by someone else at 2am.

A kernel in this dialect, printed:

```mlir
func.func @gemm_tile(%A: memref<1024x1024xf16>, %B: memref<1024x1024xf16>,
                     %C: memref<1024x1024xf16>) {
  %acc = mychip.alloc : !mychip.tile<128x256xf32, L1, ROW>
  scf.for %k = %c0 to %c1024 step %c64 {
    %sa = mychip.alloc : !mychip.tile<128x64xf16, L1, SWIZZLE_XOR32>
    %sb = mychip.alloc : !mychip.tile<64x256xf16, L1, SWIZZLE_XOR32>
    %ga = mychip.view %A[%m, %k] : !mychip.tile<128x64xf16, DRAM, ROW>
    %gb = mychip.view %B[%k, %n] : !mychip.tile<64x256xf16, DRAM, ROW>
    %t0 = mychip.dma %ga -> %sa : ... 
    %t1 = mychip.dma %gb -> %sb : ...
    mychip.wait %t0, %t1
    %acc2 = mychip.matmul %sa, %sb, %acc : ...
  }
  return
}
```

That is already a legal, verifiable, round-trippable program before a single line of code generation exists — and `mlir-opt` will canonicalize and CSE it for you.

## The front end: three routes

### Route A — Clang with a new target

If your users are C/C++ programmers and your chip has anything resembling a general-purpose scalar core, extending Clang is the cheapest path to a language people already know.

The work is bounded and well-trodden:

```cpp
// clang/lib/Basic/Targets/MyChip.h
class LLVM_LIBRARY_VISIBILITY MyChipTargetInfo : public TargetInfo {
  static const char *const GCCRegNames[];
public:
  MyChipTargetInfo(const llvm::Triple &T, const TargetOptions &)
      : TargetInfo(T) {
    PointerWidth = PointerAlign = 64;
    SuitableAlign = 128;
    LongWidth = LongAlign = 64;
    AddrSpaceMap = &MyChipAddrSpaceMap;   // language AS -> target AS
    UseAddrSpaceMapMangling = true;
    resetDataLayout("e-p:64:64-i64:64-v128:128:128-n32:64-S128"
                    "-p3:32:32-p4:32:32");  // p3 = L1, p4 = L2
  }
  void getTargetDefines(const LangOptions &, MacroBuilder &Builder) const override {
    Builder.defineMacro("__MYCHIP__");
    Builder.defineMacro("__MYCHIP_ARCH__", "200");
  }
  ArrayRef<Builtin::Info> getTargetBuiltins() const override;
  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::VoidPtrBuiltinVaList;
  }
};
```

The address space map is the load-bearing part. Declaring L1 and L2 as distinct address spaces gives you `__attribute__((address_space(3))) half *tile;` in user code, alias analysis that knows L1 and DRAM never overlap, and a type error when someone passes the wrong pointer to an intrinsic. Wire it up through `LangAS` so that a named qualifier (`__l1`, via a `-fdeclspec`-style keyword or a header `#define`) is what users actually type.

Builtins expose your coarse instructions:

```
// Classic .def form (older trees); recent Clang has been migrating these to
// TableGen (BuiltinsMyChip.td) — check which your branch uses.
BUILTIN(__builtin_mychip_dma_start, "ivC*v*Ii", "n")
BUILTIN(__builtin_mychip_dma_wait,  "vi",       "n")
TARGET_BUILTIN(__builtin_mychip_mma_f16, "vV128fV64hV64h", "nc", "mma-f16")
```

Then a header ships `mychip_intrinsics.h` wrapping those in typed inline functions, and your users write recognizable C++.

For the host/device split, do not invent a new offloading model. Clang already has three, in ascending order of effort: OpenMP `target` offload (least work, works today, awkward for tile-level control), a CUDA/HIP-style two-pass compilation with `__global__`-equivalent attributes and `clang-linker-wrapper` producing a fat binary (most familiar to your users), or SYCL. Copy the HIP flow — it is the one with the fewest NVIDIA-specific assumptions baked into the driver.

The connection to MLIR from this route used to be a gap. **ClangIR** (`clang/lib/CIR`) closes it: Clang emits an MLIR dialect (`cir`) instead of going straight to LLVM IR, so a C++ front end and an MLIR pipeline can meet in the middle. It is genuinely usable now but still moving; if you bet on it, pin a revision and expect to carry patches. The conservative alternative is to have Clang emit LLVM IR with your intrinsics and keep MLIR entirely on the DSL side — two front ends, one backend.

### Route B — an embedded DSL

If your users are kernel authors chasing peak FLOPs rather than porting legacy C, a Python-embedded DSL is a better product. This is the Triton / CuTe DSL / TileLang shape: Python executes at trace time as the metaprogramming language, and calls to your primitives build MLIR directly through the Python bindings.

```python
@mychip.kernel(grid=lambda M, N: (M // 128, N // 256))
def gemm(A: Tensor, B: Tensor, C: Tensor, M: int, N: int, K: int):
    acc = mychip.zeros((128, 256), dtype=f32, level=L1)
    for k in mychip.range(0, K, 64, stages=3):        # pipelining is a schedule hint
        a = mychip.load_tile(A[pid_m*128 : +128, k : k+64], layout=SWIZZLE_XOR32)
        b = mychip.load_tile(B[k : k+64, pid_n*256 : +256], layout=SWIZZLE_XOR32)
        acc += mychip.matmul(a, b)
    mychip.store_tile(C[pid_m*128 : +128, pid_n*256 : +256], acc.to(f16))
```

The entire front end is a few thousand lines of Python that calls `mlir.ir` builders — no parser, no type checker of your own, no LSP to write, and shapes are ordinary Python integers folded to constants at trace time. `stages=3` is a hint your pipelining pass reads; the user never writes a semaphore.

Cost of this route: you own the diagnostics. A Python traceback pointing at line 4 of a generated IR builder is a bad error message, so budget real time for source-location threading (`mlir.ir.Location` from Python frame info) and for a diagnostic handler that maps IR verification failures back to user lines.

### Route C — a standalone grammar

Occasionally justified — a configuration-heavy dataflow language, or a domain where C syntax actively misleads. Rarely worth it. You will spend a year on tooling (parser, formatter, LSP, debugger integration) that Routes A and B inherit. Choose this only when the semantic distance from C and Python is large enough that embedding is a lie.

**A reasonable default is A and B together**: Clang for the systems-level and host code, an embedded DSL for the hot kernels, both meeting in your L2 dialect.

## The lowering pipeline

The pipeline is where MLIR pays for itself. A workable ordering:

```
stablehlo / linalg-on-tensors
  │  fusion, shape inference
  ▼
linalg + transform dialect          ← tiling schedule lives here
  │  tile-to-scf, promote-to-scratchpad
  ▼
mychip (L2: tiles, dma, matmul)
  │  layout propagation & conversion elimination
  │  double-buffering / software pipelining
  ▼
mychip_ll (L1: descriptors, semaphores, engine queues)
  │  register/scratchpad allocation, semaphore insertion
  ▼
llvm dialect  ──or──  emitc  ──or──  direct binary emission
```

Two conventions worth adopting early.

**Use the transform dialect for schedules.** Tiling factors, unroll counts, and pipeline depth expressed as IR rather than as C++ pass options means an autotuner can enumerate schedules without recompiling your compiler, and a kernel author can pin a schedule that works. This is the same insight that makes Halide and TVM productive, with the scheduling language itself represented as MLIR.

**Write conversions as dialect conversions, not ad-hoc rewrites.** The framework's type converter is what makes an L2→L1 lowering tractable when types change shape:

```cpp
struct MatmulLowering : public OpConversionPattern<mychip::MatmulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mychip::MatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto lhsTy = cast<TileType>(op.getA().getType());
    auto rhsTy = cast<TileType>(op.getB().getType());

    // The ISA does 128x128x32 per instruction; emit the loop nest over that.
    const int64_t M = lhsTy.getShape()[0], K = lhsTy.getShape()[1];
    const int64_t N = rhsTy.getShape()[1];
    if (K % 32 != 0)
      return rewriter.notifyMatchFailure(op, "K must be a multiple of 32");

    Value acc = adaptor.getAcc();
    Location loc = op.getLoc();
    for (int64_t m = 0; m < M; m += 128)
      for (int64_t n = 0; n < N; n += 128)
        for (int64_t k = 0; k < K; k += 32)
          acc = rewriter.create<mychip_ll::MmaInstOp>(
              loc, acc.getType(), adaptor.getA(), adaptor.getB(), acc,
              rewriter.getDenseI64ArrayAttr({m, n, k}));

    rewriter.replaceOp(op, acc);
    return success();
  }
};

void MyChipToLowLevelPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<mychip_ll::MyChipLLDialect, scf::SCFDialect,
                         arith::ArithDialect>();
  target.addIllegalDialect<mychip::MyChipDialect>();

  TypeConverter converter;
  converter.addConversion([](TileType t) { return DescriptorType::get(t); });
  converter.addConversion([](Type t) { return t; });

  RewritePatternSet patterns(&getContext());
  patterns.add<MatmulLowering, DmaLowering, AllocLowering>(converter, &getContext());
  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
```

`notifyMatchFailure` rather than a bare `return failure()` is a small habit with a large payoff: `-debug-only=dialect-conversion` then tells you *why* a kernel failed to lower, which during bring-up is most of your debugging.

## Code generation: three options, honestly priced

This is the decision that most affects your schedule.

| Approach | Time to first kernel | Code quality ceiling | Ongoing cost |
|----------|---------------------|----------------------|--------------|
| **EmitC → vendor C compiler** | days | bounded by that compiler | low, if the C compiler exists |
| **Custom emitter → assembler** | 2–4 weeks | whatever you hand-schedule | you own scheduling and allocation forever |
| **Full LLVM backend** | 4–9 months | highest | a backend to maintain across LLVM releases |

If the chip already has a C compiler for its scalar core (many licensed DSP and RISC-V cores do), **start with EmitC**. Lower your L1 dialect to `emitc` calls into a hand-written intrinsics header, compile with the existing toolchain, and you have end-to-end execution in a week. The generated C is ugly; nobody reads it. This buys you the ability to validate the entire upper pipeline against real silicon while the backend work proceeds in parallel — and it is a permanent fallback when a new instruction lands before backend support does.

Write the full LLVM backend when scalar code quality starts to matter — control flow, address arithmetic, spill placement around the tile ops. Its TableGen target description is a real investment (register classes, instruction selection patterns, a scheduling model, ABI lowering), but it is also where twenty years of register allocation and instruction scheduling live, and hand-written schedulers do not catch up.

The middle option — your own emitter from MLIR straight to your assembler — is right when the ISA is so unlike a general-purpose CPU that LLVM's abstractions fight you, which is genuinely the case for some dataflow and systolic designs. Be clear-eyed: choosing this means writing your own register allocator eventually.

## Runtime, ABI, and packaging

The pattern to copy is upstream's GPU compilation flow, which is already target-agnostic: a `gpu.module` is compiled to a `gpu.binary` by a target attribute, and the host code calls a small runtime API. Model your kernel module the same way, and your host-side lowering becomes a solved problem.

Pin down these four things in writing, before the first kernel ships, because changing any of them later is a flag day:

- **Kernel entry ABI** — where the argument buffer lives, alignment, how the grid coordinate reaches the kernel.
- **Binary container** — an ELF with your ISA's `e_machine` is better than a bespoke format; you inherit `objdump`, symbols, and debug sections.
- **Launch and completion** — doorbell, command queue format, completion signalling.
- **Versioning** — an arch field in the binary, checked by the runtime. Tape-out 2 will change an encoding, and you want a clean error rather than a hang.

## A bring-up order that works

Sequence matters more than any individual choice. This ordering front-loads the risk:

1. **Simulator and ISA spec first.** A cycle-approximate simulator with a trace log is the compiler team's oracle. Without it, every bug is ambiguous between compiler and hardware.
2. **Dialect and verifier, round-tripping under `mlir-opt`.** One day of work, and it forces the ISA constraints into writing.
3. **One kernel, hand-written in your L2 dialect, lowered through EmitC, running on the simulator.** This is the milestone that proves the pipeline. Pick a `matmul` — it exercises DMA, scratchpad, the MMA unit, and the epilogue.
4. **The DSL front end**, targeting that same L2 dialect. Now kernel authors can start, and they are your best source of missing-op reports.
5. **Optimization passes**, in the order the profile demands — almost always double-buffering first, then layout conversion elimination, then fusion.
6. **The LLVM backend or custom emitter**, in parallel with (4) and (5), swapped in behind a flag once it beats EmitC.
7. **Clang target support**, when host-side and systems code become the bottleneck rather than kernels.

Resist reordering (6) before (3). A backend with nothing above it cannot be tested, and a compiler team that cannot run anything for six months makes design decisions on speculation.

## Testing, which is the actual product

Compiler bugs on new silicon are diabolical to debug because three things are unproven at once — the compiler, the simulator, and the chip. Instrument accordingly:

- **`lit` + FileCheck for every pass.** MLIR's testing story is its second-best feature after the dialect system. A pass without a `.mlir` test that pins its output is not finished.
- **Round-trip tests on the dialect.** `mlir-opt %s | mlir-opt | FileCheck %s` catches printer/parser drift, which otherwise surfaces as a corrupted IR dump at the worst moment.
- **Op-level differential testing.** For each op, generate random shapes and inputs, run through your pipeline on the simulator, and compare against a NumPy reference with a documented tolerance. Catch it here or catch it in a transformer three months later.
- **A crash-and-hang corpus.** Every hardware hang gets a minimized `.mlir` reproducer checked into the tree, with an assertion in the verifier that would have rejected it. The verifier grows from experience; that is expected and healthy.

## Honest caveats

- **MLIR's API is not stable.** Upstream renames things, changes pass registration, and moves headers between releases. Pin an LLVM commit, update deliberately on a schedule, and keep your out-of-tree footprint small enough to rebase. Teams that track `main` continuously spend a visible fraction of their capacity on it.
- **The learning curve is real.** The dialect conversion framework, the interface/trait system, and bufferization each take a competent engineer weeks. Budget it as onboarding, not as slippage.
- **Dialect proliferation is the classic failure mode.** Every ad-hoc pass tempts you toward a new dialect. Three or four is a healthy stack; twelve means nobody knows where a transformation belongs.
- **Bufferization will hurt.** The tensor→memref boundary is the single most common source of "correct IR that runs 5× slow" and of allocations appearing in inner loops. Learn how it makes decisions before you need to debug it.
- **Reuse before you write.** Check the existing accelerator work — IREE, `mlir-aie`, `tt-mlir`, Triton's structure, the upstream GPU compilation flow — before designing a subsystem. Most bring-up problems have a public precedent, and reading one saves a quarter.

## Takeaways

Bringing up a new AI chip's software stack is a layering exercise, not a heroics exercise. Define your own abstraction exactly where your hardware differs — memory levels, tile layouts, explicit asynchrony — encode that in a dialect with a strict type system and a real verifier, and inherit everything else. Reach LLVM IR only when there is nothing left worth reasoning about.

Get one kernel running end-to-end in week three, even through an embarrassing code generator, and let the quality of every layer improve behind a stable interface. The chip that ships is not the one with the most elegant compiler; it is the one whose compiler was executing real kernels early enough that the hardware team could still act on what it found.
