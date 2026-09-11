# Toy: a hand-written MLIR dialect

A complete, self-contained MLIR dialect in one C++ file — no ODS/TableGen — with
three ops (`toy.constant`, `toy.add`, `toy.print`), custom assembly syntax,
verifiers, and a driver that builds IR with the builder API, parses the same IR
from text, walks it, and shows the verifier rejecting bad IR.

## Build

```bash
sudo apt-get install -y libmlir-18-dev mlir-18-tools      # Ubuntu 24.04

cmake -G Ninja -S . -B build \
  -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
ninja -C build
./build/toy
```

For a different LLVM version, point `MLIR_DIR` at that install's
`lib/cmake/mlir`. `./build.sh` is a CMake-free fallback.

## Gotchas when hand-writing a dialect

These are the things ODS would have generated for you, and the exact points
where a hand-written dialect fails to compile:

- **`MLIR_DECLARE_EXPLICIT_TYPE_ID` / `MLIR_DEFINE_EXPLICIT_TYPE_ID`** must both
  appear at global scope for the dialect class. ODS emits them.
- **`getAttributeNames()` is mandatory on every op.** There is no default;
  `addOperations<>` fails to instantiate without it. Return `{}` for ops with no
  inherent attributes.
- **`OpTrait::OneResult` does not give you `getResult()`.** It supplies
  `replaceAllUsesWith` and the trait verifier only. Either declare your own
  accessor (what this file does) or add
  `OpTrait::OneTypedResult<T>::Impl` *before* `OneResult` in the trait list —
  note that the typed variant `cast`s, so it asserts on a wrong result type
  before your verifier can produce a readable diagnostic.
- **Every op must be listed in `initialize()`** via `addOperations<>`, or
  creating one asserts at runtime.
- **Custom syntax cannot express every mistake.** `toy.constant`'s parser
  derives the result type from its attribute, so a type mismatch is only
  reachable through the generic form — which is why invariants belong in
  `verify()`, not in the parser.

## What ODS replaces

The same three ops in TableGen are roughly 25 lines total, and `mlir-tblgen`
generates the accessors, builders, parser, printer, verifier, and registration.
This file is ~380 lines. That ratio is the argument for ODS — but writing it by
hand once makes the generated code readable.
