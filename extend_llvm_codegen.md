---
layout: page
title: Extending LLVM to create your own device programming language 3: codegen
---

# Builtin CodeGen

There are two options for emitting low-level code for a builtin function in the CodeGen phase:

1. Lower the call directly into a sequence of RISC-V instructions.
2. Transform it into an LLVM intrinsic, so the lowering can be done later in the LLVM layer.

Compared to option 1, option 2 lets the optimizer see a DMA operation instead of a sequence of store instructions, so LLVM IR passes can analyze the graph made up of all these operations.
An LLVM intrinsic also adapts more easily to new versions of the chip.

## define LLVM intrinsic

```tablegen
let TargetPrefix = "arch" in {
  // DMA is submitted directly as one by-value configuration aggregate. It
  // contains 27 scalar i32 fields and four arrays of four i32 loop fields.
  def int_arch_dma_submit : DefaultAttrsIntrinsic<
      [llvm_i32_ty], [llvm_any_ty, llvm_vararg_ty], [IntrHasSideEffects]>;
      // llvm_i32_ty is return type
      // [llvm_any_ty, llvm_vararg_ty] is args type list
      // [IntrHasSideEffects] attributes list
}
```

LLVM intrinsic must be started with prefix int_. LLVM TableGen will remove the prefix to get enum name arch_dma_submit, add "llvm." and replace "_" with ".", so above intrinsc name is llvm.arch.dma.submit.
LLVM tableGen generates some lookup tables for this intrinsic definition.  

CGM.getIntrinsic(Intrinsic::arch_dma_submit, {Ops[0]->getType()}) 
// second param is overloaded type DMAConfig

getIntrinsic First use enum name arch_dma_submit to lookup the name table get llvm.arch.dma.submit, and concatenate with overloaded type T, get full name.  looup signature table and attributes table. All lookup results will be combined into llvm::Function returned.

Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::arch_dma_submit, {Ops[0]->getType()}),Ops);

this createCall will generate 
%1 = call i32 (%struct.DMAConfig, ...) @llvm.arch.dma.submit.s_struct.DMAConfigs(%struct.DMAConfig %0)
i32 is return type

## LLVM instrinsic lowering
