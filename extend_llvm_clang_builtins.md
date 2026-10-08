---
layout: page
title: Extending LLVM to create your own device programming language 2: clang builtins
---

## Problem

Suppose your AI chip has a RISC-V scalar core that handles address arithmetic
and task scheduling/orchestration. To move weights and feature maps from DRAM
to SRAM, it has to issue a DMA operation. That DMA operation is a coarse-grained,
high-level instruction: "coarse" because a single transfer is made up of many
register configurations.

How do you expose such an operation to C/C++ programmers, so they can write a
single function call while the compiler takes care of expanding it into the
right sequence of register writes? This post shows how to do that with a clang
builtin.

## clang builtin

The coarse-grained instruction is like :

```cpp
struct DMAConfig{
    // configuration fields to determine a dma operation
    // like src, dst, and layout remapping
};
TaskHandle __builtin_dma_submit(DMAConfig config, ...); // declartion only, no implementation
```

`__builtin_dma_submit` is a clang builtin: a function the compiler knows by
name. There is no header that declares it and no library that defines it. The
programmer calls it like an ordinary function, and the compiler decides what
code it becomes.

Builtins live in clang's *Basic* layer (`clang/include/clang/Basic`), the
foundation shared by every later stage of the front end: lexer, parser, Sema,
and CodeGen. Each target has its own builtin table, such as `BuiltinsX86.td` or
`BuiltinsRISCV.td` (`.def` files in older releases). An entry lists the
builtin's name, its type signature, and attributes such as `nothrow` or
"custom type checking". For your chip you add a table like `BuiltinsNPU.td`
and register it from your target's `TargetInfo::getTargetBuiltins()` in
`clang/lib/Basic/Targets/`.

A builtin does three jobs:

1. **It gives Sema a name and a signature.** A call to `__builtin_dma_submit`
   is type-checked like any other call, so a wrong argument is a compile-time
   error, not a hung DMA engine. Builtins marked for custom type checking, like
   the variadic one above, get a hook in Sema's target checks. There you can
   require an argument to be a compile-time constant, or reject a `DMAConfig`
   whose layout remapping the engine cannot perform.
2. **It is the contract between the programmer and the compiler.** The
   programmer states *what* to transfer; the builtin hides *how*: which
   registers to program, in what order, and how to kick off the engine. When
   the next chip revision changes the register layout, only the compiler
   changes. Kernel source stays the same.
3. **It gives CodeGen a hook.** When clang emits LLVM IR, each builtin is
   lowered in `CGBuiltin.cpp` (split per target under `TargetBuiltins/` in
   recent releases), usually to a target intrinsic such as
   `llvm.riscv.npu.dma.submit`. The intrinsic's attributes tell the optimizer
   that it has side effects, so the DMA is neither deleted nor reordered past
   the code that depends on it. The backend then expands the intrinsic into the
   actual register writes.

The `TaskHandle` return value matters too. Later operations that consume the
transferred data can take the handle as an argument, which turns the
DMA-then-compute ordering into data flow the compiler can see. That lets a
later pass insert the wait for DMA completion instead of leaving it to the
programmer.

## Add __builtin_dma_submit

You can add `__builtin_dma_submit` to the target's builtin TableGen file, e.g. `clang/include/clang/Basic/BuiltinsRISCV.td`:

```tablegen
def DMASubmit : TargetBuiltin { // only available when compiling for this target
  let Spellings = ["__builtin_dma_submit"]; // function name used in the device developer's code
  let Prototype = ""; // no signature: the parameter is a custom struct type
  let Attributes = [NoThrow, IgnoreSignature, RequireDeclaration];
}
```

This builtin has 3 attributes. `NoThrow` means CodeGen emits a plain `call` marked `nounwind`, with no `invoke` or landing pad. `IgnoreSignature` means clang recognizes a declaration of this name as the builtin even though its type doesn't match the (empty) builtin signature. `RequireDeclaration` means the builtin needs a declaration from a header before it can be used.

With this definition in place, the clang frontend recognizes `__builtin_dma_submit` calls in your device code.

## How clang processes your __builtin_dma_submit

### Basic layer

When the target is RISC-V, `getTargetBuiltins()` registers `__builtin_dma_submit` as a builtin identifier. Only the `.td` definition is needed.

### Lex / Parse

The call is parsed as an ordinary identifier and function call. Nothing builtin-specific is needed.

### Sema (semantic analysis) declaration

When clang sees the declaration in the header, `IgnoreSignature` lets it attach a `BuiltinAttr` to that `FunctionDecl` (`SemaDecl.cpp`), and `NoThrow` adds a `NoThrowAttr`. Nothing builtin-specific is needed.

### Sema : call

The arguments get ordinary C/C++ type checking against the header's signature. The call then goes through the target-specific check `SemaRISCV::CheckBuiltinFunctionCall`, where `__builtin_dma_submit` falls into `default: break` and nothing happens. Nothing builtin-specific is needed.

### AST

The result is an ordinary `CallExpr`. The only difference is that the callee's `getBuiltinID()` is non-zero. Nothing builtin-specific is needed.

### CodeGen

`CGExpr.cpp` sees that the callee has a builtin ID and calls `EmitBuiltinExpr`, which dispatches by target to `EmitRISCVBuiltinExpr`. There, a `case` for `__builtin_dma_submit` emits the target intrinsic. This is the one place that needs builtin-specific code.

Lex/Parse, Sema, and AST are shared infrastructure, so as a compiler author you rarely need to pay much attention to them.
In the next article, I will show how to generate low-level code for `__builtin_dma_submit`.
