---
layout: page
title: Extending LLVM to create your own device programming language 1: Introduction
---

# Extending LLVM to Create Your Own Device Programming Language

In this series, I'll show how to build a device programming language on top of the LLVM compiler infrastructure.

## The problem

You have a new NPU. On the die are several kinds of execution unit — a tensor engine for matrix work, a vector engine for elementwise work, a scalar unit for address arithmetic and control flow, and one or more DMA engines that move tiles between DRAM and the on-chip scratchpads. They are independent instruction queues. Nothing in the hardware stops one from reading a buffer another has not finished writing.

At bring-up time, the only way to program that chip is to speak to it in its own terms:

- lay out every tile by hand in a scratchpad you allocate yourself, because there is no cache and no allocator;
- issue each engine's work by writing its configuration registers;
- and order the engines against each other with explicit `SetFlag` / `WaitFlag` pairs — forward edges so a consumer does not read early, backward edges so a producer does not overwrite a buffer still in use.

That works, and the first few kernels get written that way. It stops working around the thirtieth. The flags are invisible to any reviewer, a missing one produces plausible wrong numbers on one core out of many, and a leaked flag register deadlocks the core. None of it survives the next ISA revision, because every scheduling decision is baked into source files nobody can regenerate. (The argument for why this layer needs an abstraction at all is spelled out in [Producer and consumer](producer_consumer); this article is about building the compiler that provides it.)

So what you actually want to hand the kernel team is a *language*: something close enough to C/C++ that they can write a kernel in an afternoon, with a programming model that makes tiles, memory levels, and producer/consumer ordering first-class — and a compiler that turns that into the register writes and barriers you would otherwise have written by hand.

## Why build it on Clang/MLIR/LLVM

Writing a compiler from scratch to get there is the wrong trade. Almost everything between "the kernel author types a `for` loop" and "the machine gets an instruction" is not specific to your chip:

| You inherit | You write |
|---|---|
| C/C++ parsing, name lookup, template instantiation, diagnostics (Clang) | the surface extensions: address-space qualifiers, tile types, kernel attributes |
| structured control flow, loop transforms, bufferization, canonicalization, CSE, the conversion framework (MLIR) | your dialects — what a tile *is*, what the memory levels are, what ordering means |
| register allocation, scheduling, instruction selection infrastructure, object emission, DWARF (LLVM) | the backend: your ISA, your calling convention, your asynchrony |
| debuggers, profilers, `-fsanitize`, IDE integration that already speak these formats | the host runtime: launch, argument marshalling, binary packaging |

The right-hand column is genuinely yours and no infrastructure will write it for you. The left-hand column is several engineer-years, and it is free. That ratio — not ideology about MLIR — is the reason to extend an existing toolchain rather than start one.

The other reason is timing. Forking LLVM and writing a complete backend before anything runs spends the better part of a year before the first `matmul` executes. Building the stack in layers, with MLIR in the middle, lets you run kernels on a simulator in week eight and keep the same front end when the silicon arrives.

