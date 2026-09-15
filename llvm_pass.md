---
layout: page
title: How to use a customized LLVM pass to lower an intrinsic
---

# The problem to solve

A device intrinsic on some new chip does not map to a single instruction. Issuing one means programming a long sequence of hardware registers — source and destination layouts, per-dimension strides and lengths, barrier IDs — and that register image is what the hardware actually consumes. The image is fixed once the operation is known, so there is no reason to recompute it on every launch: it can be produced once, dumped as a **task descriptor** file, and afterwards simply loaded and replayed.

The real question is *where* that descriptor gets produced.

The obvious answer is an **offline descriptor builder**: a separate host tool that takes the operation parameters, encodes the register image, and writes the file. It works, but it splits one logical thing into two artifacts built from two different sources. The builder and the device kernel must agree on every register field, every layout convention, and every encoding rule — and nothing enforces that agreement. Change a field in the hardware definition and you must remember to update both sides in lockstep; forget one, and you get a descriptor that the kernel launches but the hardware misinterprets. On top of that, the builder becomes an extra dependency in the build graph, with its own version to track.

The better answer is to generate both from the **same compilation**. The intrinsic call site already carries everything the descriptor needs, as compile-time constants. A custom LLVM pass can read those operands straight out of the IR, encode the register image, write the descriptor file, and rewrite the call into the short launch sequence that references it. One build, one source of truth, two artifacts that cannot disagree — and as a bonus, an illegal layout becomes a compile-time error instead of a silent hardware fault at run time.

# How does llvm pass work?

## Prerequisite background

### LLVM IR

LLVM IR is the typed, SSA-form intermediate representation that every LLVM tool
reads and writes. It has three interchangeable forms: a textual one (`.ll`), a
binary one (`.bc`, "bitcode"), and the in-memory C++ object graph — `Module`
holding `Function`s holding `BasicBlock`s holding `Instruction`s. `llvm-as` and
`llvm-dis` convert between the first two; a pass always works on the third.

The textual form is what makes this whole approach pleasant to develop. An
intrinsic call and the constant it refers to show up in `.ll` exactly as the
pass will see them:

```llvm
%struct.LayoutC = type { i32, [4 x i32], [4 x i64] }

@descriptor = internal constant %struct.DescriptorC { i32 2, i32 0, ... }

call void @device_submit(ptr @descriptor, i64 128, i64 256, ptr @start_impl)
```

You can read it, diff it before and after your pass, and hand-write it as a test
input without going through the front end at all.

### The pipeline is four separate tools

A plain `clang -O2 a.c -o a` hides it, but a full LLVM compilation is four
distinct stages, and each one is available as a standalone binary:

| Tool | In | Out | Role |
|------|----|-----|------|
| `clang` | source | IR | front end: parsing, type checking, template instantiation |
| `opt` | IR | IR | middle end: target-independent analysis and transformation |
| `llc` | IR | object file | back end: instruction selection, register allocation, emission |
| `ld` | object files | ELF | linking |

Splitting them apart is what gives you a place to stand:

```text
kernel code with intrinsic call
  --> clang --> input.ll
  --> opt with load-pass-plugin --> lowered.ll
  --> llc --> program.o
  --> ld --> elf
```

Two flags make the split work. `-S -emit-llvm` tells `clang` to stop after
producing IR, and `-Xclang -disable-llvm-passes` tells it to skip its own
built-in optimization pipeline, so what lands in `input.ll` is exactly what the
front end produced — nothing has been folded, inlined, or reordered behind your
back.

### `opt` and the pass pipeline

`opt` is the middle-end driver: it parses IR, runs a pipeline of passes over it,
and writes IR back out. The pipeline is spelled out as a string:

```bash
opt -passes='always-inline,function(sroa,instcombine),default<O2>' -S in.ll -o out.ll
```

Three things about that string matter later. Order is execution order, left to
right. Nesting expresses granularity — `function(...)` runs those passes once per
function, while a bare name at the top level is a module pass that sees the whole
module at once. And `default<O2>` is just shorthand for a preset list of several
dozen passes, which means you can put your own pass *before* it, *after* it, or
sandwiched between two hand-picked groups.

That last property is the reason this project drives `opt` directly instead of
letting `clang` run the pipeline: a pass with preconditions needs its
predecessors named explicitly, and `-passes=` is the only place you can name
them.

### Pass plugins

A pass does not have to live inside the LLVM source tree. `opt` can `dlopen` a
shared library at startup and let it register passes of its own:

```bash
opt -load-pass-plugin=./my_pass.so -passes='my-pass' -S in.ll -o out.ll
```

The library must export one symbol, which `opt` looks up by name:

```cpp
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "MyPlugin", LLVM_VERSION_STRING,
          [](PassBuilder& builder) {
            builder.registerPipelineParsingCallback(
                [](StringRef name, ModulePassManager& manager, auto) {
                  if (name == "my-pass") { manager.addPass(MyPass()); return true; }
                  return false;
                });
          }};
}
```

`LLVM_PLUGIN_API_VERSION` and `LLVM_VERSION_STRING` are an ABI handshake — the
plugin and the host must come from the same LLVM release. The callback registers
a *name*, which is what makes `my-pass` a legal token inside `-passes=`. Nothing
is inserted automatically; a plugin pass runs only when you ask for it by name.

One limitation is worth knowing up front. This plugin interface is consumed by
`opt` and by `clang -fpass-plugin`, but **not** by `llc` — the back end has no
`-load-pass-plugin` option. So the middle end is the only stage an out-of-tree
pass can reach without patching and rebuilding LLVM itself.

So the approach would be to write an LLVM pass plugin that performs the lowering
and the task descriptor generation.

## Write your own LLVM pass

There is no base class to derive from and no registration macro to invoke. A
pass is simply a copyable type that exposes three public members:

```cpp
PreservedAnalyses run(Module& module, ModuleAnalysisManager&);
static StringRef name();
void printPipeline(raw_ostream& os, function_ref<StringRef(StringRef)> map);
```

The requirement is structural, not nominal. `ModulePassManager::addPass<T>()`
wraps your type in an internal `PassModel<T>` template that simply calls those
three members; if one of them is missing or private, the template fails to
instantiate and you get a compile error pointing into `PassManagerInternal.h`.
Nothing is checked at run time.

Only the first of the three is interesting. `run` is where the work happens: it
receives the whole `Module` by reference, is free to inspect and rewrite it, and
returns a `PreservedAnalyses` describing which cached analyses survived the
edit — `none()` if the pass changed anything structural. The other two exist so
the pass manager can name your pass in diagnostics and reproduce it in a
`-passes=` string, and both can be inherited from the CRTP helper
`PassInfoMixin`, which derives the name from the type itself:

```cpp
struct MyPass : PassInfoMixin<MyPass> {
  PreservedAnalyses run(Module& module, ModuleAnalysisManager&) {
    // core logic goes here
    return PreservedAnalyses::none();
  }
};
```

That is the entire contract. A `class` works just as well as a `struct`, as long
as the inheritance and the members are both spelled `public` — `struct` is the
convention in LLVM's own tree mostly because a pass rarely has state worth
hiding.

The second parameter is the module-level analysis manager, the lazily evaluated
cache that hands out results like the call graph or the profile summary. A pass
that only needs to walk instructions and read constants never touches it, which
is why the parameter is so often left unnamed.

### LLVM pass for lowering and task descriptor generation

Back to the problem defined at the article beginning. You need to implement an LLVM pass to lowering intrinsic and generate task descriptor.  

For each intrinsic invoked, you need to specifiy a intrinsic handler for lowering. The handler need following information:

target intrinsic function name (symbol) in LLVM IR 
arch code for correct lowering
core method (process) when target intrisic function invoked is found in llvm::module (compilaton unit)
 
 put all registered intrisic handlers in a registry. 

 #### find intrisic calls in llvm::module

```cpp
 for (Function& function : module)
    for (BasicBlock& block : function)
      for (Instruction& instruction : block) {
        auto* call = dyn_cast<CallBase>(&instruction);
        if (!call) continue;
        auto* callee =
            dyn_cast<Function>(call->getCalledOperand()->stripPointerCasts());
        if (!callee) continue;
        const StringRef symbol = callee->getName(); 
        // match callee symbol with handler's symbol field
        auto intrinsic_handler = find_handler_in_registry(symbol);
        CallInst *plain_call = dyn_cast<CallInst>(call);
        intrinsic_handler->process(plain_call);
      }
```

#### The handler's `process` method

Once the registry hands back a handler for the callee symbol, everything that is
specific to that one intrinsic happens inside a single call, `process(plain_call)`.
The `CallInst*` is the only argument it needs, because the call site already
carries the whole operation: the callee says *which* intrinsic this is, the
operands carry the compile-time constants the descriptor is encoded from, and
the instruction's position in its basic block says *where* the replacement code
has to go.

A `process` implementation does three things, in this order:

1. **Read the operands.** Walk `getArgOperand(i)` and `dyn_cast` each one to
   `ConstantInt` / `ConstantDataArray` / `GlobalVariable`. This is also the
   natural place to validate: an operand that is not a constant, or a layout
   that the hardware cannot express, should become a compile-time diagnostic
   here rather than a silent misencoded descriptor later.
2. **Encode and dump the descriptor.** Turn those constants into the register
   image and write it to the descriptor file. Nothing about this step touches
   the IR.
3. **Rewrite the call.** Emit the short launch sequence that references the
   descriptor, forward any uses of the old result to the new value, then delete
   the original call.

Only the third step needs LLVM's mutation API:

```cpp
// inside process(CallInst* call)

// IRBuilder<> is the default instantiation (ConstantFolder + IRBuilderDefaultInserter);
// constructing it from an instruction sets the insert point *before* that instruction,
// so everything emitted below lands where the intrinsic call used to be.
IRBuilder<> builder(call);

// Inherit the original call's debug location, otherwise the generated
// instructions carry no !dbg and the lowering becomes invisible to the debugger
// and to -pass-remarks output.
builder.SetCurrentDebugLocation(call->getDebugLoc());

// Reading the call site:
//   call->arg_size()            -- number of call arguments
//   call->getArgOperand(iArg)   -- argument iArg, for iArg in [0, arg_size)
//   call->getCalledFunction()   -- the callee, i.e. the intrinsic declaration
//   call->getType()             -- the intrinsic's return type
for (unsigned iArg = 0; iArg < call->arg_size(); ++iArg) {
  Value* arg = call->getArgOperand(iArg);
  // dyn_cast<ConstantInt>(arg), etc. -- feed the descriptor encoder
}

// Emitting the replacement. The builder inserts into the block that `call`
// belongs to, so the new instructions are part of the module immediately --
// there is nothing to "commit" afterwards.
CallInst* lowered = builder.CreateCall(launch_callee, launch_args);

// If the intrinsic returned a value, every user of the old result has to be
// redirected before the old call can go away.
if (!call->getType()->isVoidTy())
  call->replaceAllUsesWith(lowered);

// Drop the intrinsic call. After this the `call` pointer is dangling -- read
// everything you need from it *before* this line.
call->eraseFromParent();
```

One detail that the loop in the previous section glosses over: `eraseFromParent`
deletes the instruction the enclosing range-based `for` is currently standing
on, which invalidates its iterator. Either advance the iterator first with
`make_early_inc_range(block)`, or — usually clearer once a handler may emit more
than one instruction — do the walk and the rewriting in two phases: collect the
matching `(CallInst*, handler)` pairs into a worklist, and only then drain the
worklist calling `process` on each. The second shape also makes the pass's
return value easy to get right: `PreservedAnalyses::all()` when the worklist
came up empty, `none()` when anything was rewritten.
