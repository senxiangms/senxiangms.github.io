---
layout: page
title: TileLang Lower — from TIR to device code, and how to add a backend
---

# TileLang Lower: from TIR to device code, and how to support new hardware

The [frontend](tilelang_frontend) and [trace](tilelang_trace) articles ended with a `PrimFunc`: a TVM TIR function that still speaks in *tiles* — `T.copy`, `T.gemm`, fragments and shared buffers, a `T.Kernel` launch scope. That IR is deliberately architecture-neutral. Nothing in it says how a copy becomes `cp.async` or TMA, which registers of which thread hold the accumulator, or whether the target even *has* tensor cores.

**Lowering** is the second half of the compiler: the sequence of TIR passes that turns that tile-level IR into a concrete, target-specific kernel, and finally into device source (CUDA C++/PTX, HIP, Metal Shading Language, C). This note walks the lowering pipeline in `tilelang/engine/lower.py`, shows how it dispatches per target, and ends with the concrete extension points you implement to teach TileLang a new accelerator.

## The entry point

Everything funnels through `lower()` in `tilelang/engine/lower.py`. Its core is a smaller function, `lower_to_host_device_ir`, that does the actual IR work:

```python
def lower_to_host_device_ir(func_or_mod, target="auto", ...):
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})   # wrap PrimFunc

    target      = determine_target(target)                    # "auto" -> concrete Target
    target_host = tvm.target.Target(canon_target_host(...))
    target      = tvm.target.Target(target, target_host)

    PreLowerSemanticCheck(mod)                                 # backend-independent checks

    pipeline = resolve_pipeline(target)                       # pick backend pass list
    mod = pipeline.lower(mod, target)                         # <-- THE WHOLE PIPELINE

    host_mod   = tirx.transform.Filter(_is_host_call)(mod)    # split
    device_mod = tirx.transform.Filter(_is_device_call)(mod)
    return host_mod, device_mod, params, target, target_host
```

Four things happen here, and each is an extension point later:

1. **`determine_target`** resolves `"auto"` (or a string) into a real TVM `Target` — this is where hardware auto-detection lives.
2. **`PreLowerSemanticCheck`** runs target-independent validation on the tile-level IR.
3. **`resolve_pipeline(target)`** looks up the *pass pipeline* for this backend — the heart of the dispatch.
4. After the pipeline runs, the module is a flat TIR program that mixes a host-side launcher and one or more device kernels; two `Filter` passes split them, and `lower()` then hands each to codegen.

The two-step shape (`lower_to_host_device_ir` producing IR, `lower` adding codegen) matters because TileLang's JIT usually does its own host codegen — `lower(..., enable_host_codegen=False)` stops at IR and lets the runtime take over. But the pipeline that produces the device IR is the same either way.

## Dispatch: one pipeline per target kind

The single most important structural idea is that **there is no monolithic lowering function**. Each backend registers its own ordered list of passes, keyed by the target's kind string:

```python
# tilelang/backend/pass_pipeline/pipeline.py
class PassPipeline:
    def __init__(self, name, lower):     # name matches target.kind.name
        self.name, self._lower = name, lower
    def lower(self, mod, target):
        return self._lower(mod, target)

def register_pipeline(pipeline):         # backends call this at import time
    _PIPELINES[pipeline.name] = pipeline

def resolve_pipeline(target):            # dispatch by "cuda" / "hip" / "metal" / "c"
    return get_pipeline(target.kind.name)
```

So `resolve_pipeline` is a dictionary lookup on `target.kind.name`. The CUDA backend registers `PassPipeline("cuda", CUDAPassPipelineBody)`, ROCm registers `"hip"`, CPU registers both `"c"` and `"llvm"`, Metal registers `"metal"`, and WebGPU falls back to reusing the CPU pipeline:

```python
# tilelang/backend/common.py
register_pipeline(PassPipeline("webgpu", CPUPassPipelineBody))
```

A backend is, first and foremost, *a function `(IRModule, Target) -> IRModule`* registered under a name. Everything else — intrinsics, templates, codegen — hangs off that.

## What the pipeline actually does

A pipeline is a long straight-line sequence of TIR passes. The CUDA one (`tilelang/cuda/pipeline.py`) runs 50+ of them, but they group into three conceptual stages. The first stage is **almost entirely target-agnostic** — it is the same shape across CUDA, HIP, Metal, and CPU — and this is exactly why porting is tractable.

### Stage 1 — legalize and lower the tile abstraction

This is where tile-level IR becomes ordinary loop-and-buffer TIR. The backbone, in order:

| Pass | What it does |
|------|--------------|
| `LayoutReducer` | fixes the layout of reduction accumulators |
| `IfStmtBinding` | canonicalizes `if`-without-`else` before pipeline planning |
| `PipelinePlanning` | analyzes a `T.Pipelined` loop and plans its stages |
| `InjectSoftwarePipeline` | rewrites it into a multi-stage ring buffer (prologue/body/epilogue) |
| **`LayoutInference`** | infers the shared-memory and register (fragment) layouts from how tiles feed `T.gemm`/`T.copy` |
| **`LowerTileOp`** | lowers `T.copy`, `T.gemm`, `T.reduce`, … into concrete loops + intrinsic calls |

`LayoutInference` and `LowerTileOp` are the crux. Everything the [introduction](tilelang_intro) described as "the compiler infers the layouts" happens in `LayoutInference`; `LowerTileOp` is where a single `T.gemm` becomes a nest of MMA intrinsic calls with the inferred thread→value mapping. Note the ordering deliberately runs pipeline planning *before* layout inference, so inferred layouts already see the pipelined structure.

After this stage the IR still has abstract memory scopes (`shared`, `local.fragment`) but no more tile ops — it is loops, buffer stores, and target intrinsic calls.

### Stage 2 — legalize, vectorize, allocate, synchronize

The middle stage is generic optimization and lowering of the now-concrete loops:

- `LegalizeVectorizedLoop`, `VectorizeLoop`, `DecoupleTypeCast` — turn element loops into vector loads/stores.
- `LegalizeSafeMemoryAccess`, `LowerAccessPtr` — bounds handling and pointer lowering.
- `FlattenBuffer`, `NarrowDataType(32)`, `ConfigIndexBitwidth` — index/storage normalization.
- `PlanAndUpdateBufferAllocationLocation`, `StorageRewrite`, `MergeSharedMemoryAllocations` — decide where each buffer is allocated and pack shared memory.
- `UnrollLoop`, `LoopUnswitching`, `Simplify`, `RemoveNoOp` — cleanup.
- `InferFragment`, `LowerThreadAllreduce`, `ThreadSync("shared")` — register-fragment finalization and barrier insertion.

Almost all of these are stock or lightly-extended TVM passes. They don't know or care what GPU they're targeting.

### Stage 3 — the target-specific tail, then host/device split

This is where the pipelines diverge. The CUDA body interleaves passes that only make sense on NVIDIA hardware, gated on the detected architecture:

```python
if allow_warp_specialized(target):
    mod = tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(mod)  # Hopper+
mod = tilelang.cuda.transform.LowerBlackwell2SM()(mod)                    # SM100+
...
mod = tilelang.cuda.transform.LowerLDGSTG()(mod)      # ramp loads -> ldg/stg
mod = tilelang.cuda.transform.LowerHopperIntrin()(mod)
mod = tilelang.cuda.transform.InjectFenceProxy()(mod) # TMA async fences
mod = tilelang.cuda.transform.PersistThreadblock()(mod)
```

The HIP, Metal, and CPU pipelines replace this tail with their own concerns — Metal, for example, runs `MetalFragmentToSimdgroup` *before* layout inference to map fragments onto Metal's opaque `simdgroup` type. The stage closes with the universal finish: `SplitHostDevice`, `AnnotateDeviceRegions`, `MakePackedAPI`, `LowerDeviceKernelLaunch` — after which one `IRModule` contains both a host launcher and the device kernel(s), ready for the `Filter` split back in `lower_to_host_device_ir`.

The takeaway: **the target-agnostic front (stages 1–2) is roughly 70% of the passes, and it is shared. A new backend mostly reuses it and supplies stage 3.**

## Codegen: from TIR to source

Once host and device modules are separated, `device_codegen` dispatches — again on `target.kind.name` — to an FFI-registered C++ code generator:

```python
# tilelang/engine/lower.py
if target.kind.name == "cuda":
    fn = "target.build.tilelang_" + ("cutedsl" if "cutedsl" in target.keys else "cuda")
    device_mod = tvm.ffi.get_global_func(fn)(device_mod, target)
elif target.kind.name == "hip":
    device_mod = tvm.ffi.get_global_func("target.build.tilelang_hip")(device_mod, target)
elif target.kind.name == "metal":
    device_mod = tvm.ffi.get_global_func("target.build.tilelang_metal")(device_mod, target)
```

Each `target.build.tilelang_*` is a C++ function registered from the native side. The registration is the plug:

```cpp
// src/cuda/codegen/rt_mod_cuda.cc
TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef()
      .def("target.build.tilelang_cuda", BuildTileLangCUDA)
      .def("target.build.tilelang_cuda_without_compile", BuildTileLangCUDAWithoutCompile);
}
```

`BuildTileLangCUDA` instantiates a `CodeGenTileLangCUDA` — a subclass of TVM's `CodeGenC` — that walks the TIR and prints CUDA C++. The other backends mirror this exactly:

| Target | Codegen class | Emits |
|--------|---------------|-------|
| CUDA | `CodeGenTileLangCUDA` (`src/cuda/codegen/`) | CUDA C++ / PTX |
| HIP | `CodeGenTileLangHIP` (`src/rocm/codegen/`) | HIP C++ |
| Metal | `CodeGenTileLangMetal` (`src/backend/common/codegen/`) | Metal Shading Language |

All three extend `CodeGenC`, so the common structure of a kernel (signatures, loops, arithmetic) is inherited; each subclass only overrides the target-specific printing — intrinsic call syntax, address-space qualifiers, vector types. The generated source is then compiled by a Python-side callback (`tilelang_callback_cuda_compile` shells out to `nvcc`; the HIP one to `hipcc`), and the tile-op intrinsics resolve against hand-written headers in `src/tl_templates/{cuda,hip,cpu}/`.

## Supporting new hardware

Put the pieces together and the extension surface is small and well-defined. To add a backend `foo`, you supply, in order of how far down the stack they sit:

**1. Target registration** — teach `determine_target` about your hardware. In `tilelang/foo/target.py`:

```python
from tilelang.backend.target import register_target_detector, register_target_normalizer

register_target_detector("foo", _detect_foo_target)      # auto-detect the device
register_target_normalizer("foo", _normalize_foo_target) # canonicalize "foo -arch=..."
```

This makes `target="foo"` (and, if you want, `target="auto"`) resolve to a `Target` whose `kind.name` is `"foo"`.

**2. A pass pipeline** — the `(IRModule, Target) -> IRModule` function, registered under that same name. In `tilelang/foo/pipeline.py`:

```python
from tilelang.backend.pass_pipeline.pipeline import PassPipeline, register_pipeline

def FooPassPipelineBody(mod, target):
    mod = tirx.transform.BindTarget(target)(mod)
    # --- reuse the shared front: LayoutInference, LowerTileOp, vectorize, ... ---
    mod = tilelang.transform.LayoutInference()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    # ...
    # --- your stage-3 tail: lower ops to your ISA, insert your sync primitives ---
    mod = tilelang.foo.transform.LowerFooIntrin()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    return mod

register_pipeline(PassPipeline("foo", FooPassPipelineBody))
```

In practice you start by copying the CPU or HIP pipeline (the simplest ones, no exotic async machinery) and swapping the tail. The bulk of the passes are reused unchanged.

**3. Tile-op lowering and intrinsics** — the pipeline calls `LowerTileOp`, but *what* a `T.gemm` lowers to on your hardware is your job. This is `tilelang/foo/op/` (how a tile GEMM/copy maps to your matmul engine) and `tilelang/foo/intrinsics/` (the emitters), plus the memory-scope mapping: how TileLang's `shared` / `local.fragment` correspond to your on-chip buffers. The [intro article](tilelang_intro) shows the Ascend example, where `shared` maps to the L1 buffer and `local.fragment` to the cube unit's L0A/L0B/L0C — and where a whole new *compute* scope (`T.Scope("C")` / `T.Scope("V")`) was added because the NPU splits matmul and vector work across separate units.

**4. A C++ code generator** — in `src/foo/codegen/`, a `CodeGenTileLangFoo : public CodeGenC` that prints your device language, plus the FFI registration:

```cpp
// src/foo/codegen/rt_mod_foo.cc
TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("target.build.tilelang_foo", BuildTileLangFoo);
}
```

and one `elif target.kind.name == "foo"` branch in `device_codegen`. Back it with template headers in `src/tl_templates/foo/` that implement your intrinsics.

**5. An execution backend (optional)** — register how compiled kernels are loaded and launched at runtime, via `register_execution_backend("foo", spec)`, if the default TVM-FFI runtime path doesn't fit.

That's the whole contract. The reason it stays this contained is the design choice the [intro](tilelang_intro) leads with: because a TileLang kernel is *tiles moving through named memory scopes*, not SIMT threads, stages 1–2 of the pipeline — layout inference, tile-op lowering, vectorization, buffer planning — are architecture-agnostic and shared. Porting is concentrated in the parts that are genuinely different: the memory-scope map, the tile-op → ISA lowering, and the code printer. This is exactly the work you would otherwise rewrite by hand in CUTLASS/CuTe for every new architecture — here it is factored into four registration hooks. The two real out-of-tree adapters, [`tilelang-ascend`](https://github.com/tile-ai/tilelang-ascend) and [`tilelang-metax`](https://github.com/tile-ai/tilelang-metax), are built precisely this way.

## Takeaways

- Lowering is dispatched, not monolithic: `resolve_pipeline(target)` looks up a per-backend pass list keyed by `target.kind.name`, and `device_codegen` dispatches codegen the same way.
- A pipeline is a straight-line sequence of TIR passes in three stages: **(1)** legalize and lower the tile abstraction (`LayoutInference`, `LowerTileOp`) — largely target-agnostic; **(2)** vectorize, allocate, synchronize — generic; **(3)** a hardware-specific tail plus the host/device split.
- Codegen is a C++ `CodeGenC` subclass per target, registered over FFI as `target.build.tilelang_*`, with intrinsics backed by template headers.
- Adding a backend is four small hooks: register a **target**, register a **pipeline** (reusing the shared front), supply **tile-op lowering + intrinsics + scope mapping**, and register a **codegen**. The dataflow-centric design is what keeps that list short.

With this, the arc is complete: the [frontend](tilelang_frontend) rewrites Python so it can be intercepted, the [trace phase](tilelang_trace) runs it to emit tile-level TIR, and lowering turns that TIR into a real kernel for whatever hardware you register.
