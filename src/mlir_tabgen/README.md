# Toy: the same MLIR dialect, defined in TableGen (ODS)

The dialect from [`../mlir_dialect`](../mlir_dialect) — three ops on f64
tensors, `toy.constant`, `toy.add`, `toy.print` — written in TableGen's
**Operation Definition Specification** instead of by hand. The textual IR, the
builder calls and the driver output are identical; only the source of the op
classes changed.

```
ToyOps.td       107 lines   the dialect: ops, types, traits, syntax, builders
ToyDialect.h     30 lines   includes the two generated headers
ToyDialect.cpp   64 lines   op registration + the one verifier ODS cannot express
                 ---
                201 lines   vs 387 lines of hand-written C++ in ../mlir_dialect
```

`mlir-tblgen` expands `ToyOps.td` into **1142 lines** of C++ across four `.inc`
files. Reading them is the fastest way to learn what ODS actually promises.

## Build

```bash
sudo apt-get install -y libmlir-18-dev mlir-18-tools      # Ubuntu 24.04

cmake -G Ninja -S . -B build \
  -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++-18
ninja -C build
./build/toy
```

`./build.sh` is a CMake-free fallback: it calls `mlir-tblgen` four times into
`gen/` and then compiles, which is exactly what the CMake rules automate.

## The four generators

One `.td` file feeds four invocations of `mlir-tblgen`, and each output is
`#include`d at one specific point in the C++:

| Flag | Output | Included by | Contains |
|------|--------|-------------|----------|
| `-gen-dialect-decls` | `ToyOpsDialect.h.inc` | `ToyDialect.h` | `class ToyDialect`, `MLIR_DECLARE_EXPLICIT_TYPE_ID` |
| `-gen-dialect-defs` | `ToyOpsDialect.cpp.inc` | `ToyDialect.cpp` | its constructor/destructor, `MLIR_DEFINE_EXPLICIT_TYPE_ID` |
| `-gen-op-decls` | `ToyOps.h.inc` | `ToyDialect.h`, under `GET_OP_CLASSES` | one op class per `def`, with accessors and builder declarations |
| `-gen-op-defs` | `ToyOps.cpp.inc` | `ToyDialect.cpp`, under `GET_OP_CLASSES` and `GET_OP_LIST` | the bodies: builders, `parse`, `print`, `verifyInvariants`, and the op list for `addOperations<>` |

The `GET_OP_*` macros matter: `ToyOps.cpp.inc` is included **twice** in
`ToyDialect.cpp`, once with `GET_OP_LIST` (a bare comma-separated list of class
names, to paste inside `addOperations<...>()`) and once with `GET_OP_CLASSES`
(the method definitions). Forget the `#define` and you get a file that compiles
to nothing.

## What each ODS line replaced

- **`let arguments` / `let results`** — the operand and result accessors
  (`getLhs()`, `getValue()`, `getResult()`), the `NOperands`/`OneResult` traits,
  and the per-argument type checks. The hand-written version declares every
  accessor itself, and must also supply `getAttributeNames()`, which has no
  default.
- **`let assemblyFormat`** — `parse()` and `print()`. Two format strings here
  replace six methods and ~80 lines there.
- **`AllTypesMatch<["value", "result"]>`** — both the check that the attribute
  and the result agree *and* the type inference that lets the parser omit the
  result type: the generated parser ends with
  `result.addTypes(valueAttr.getType())`.
- **`SameOperandsAndResultType`** — `toy.add`'s entire verifier, plus a
  `build(builder, state, lhs, rhs)` overload that infers the result type. That
  is why `toy.add` needs neither a `builders` block nor `hasVerifier`.
- **`Pure`** — `getEffects()`, which makes the ops eligible for CSE and DCE.
  The hand-written dialect has no side-effect information at all, so those
  passes must leave its ops alone.
- **`let builders`** — a custom overload only where the generated ones are not
  enough; `toy.constant` uses one to derive its result type from the attribute.
- **`let hasVerifier = 1`** — the hook for invariants no constraint can state.
  Here it is a domain rule (rank ≤ 2, standing in for a hardware limit), which
  is the honest use of a custom verifier: the structural checks are already
  generated.

## Where ODS stops

Running `./build/toy` ends with the two failure modes side by side:

```
=== trait verifier (the error below is expected) ===
error: 'toy.constant' op failed to verify that all of {value, result} have same type
=== custom verifier (the error below is expected) ===
error: 'toy.constant' op target supports at most 2-D constants, got rank 3
```

The first is reachable only through the generic form `"toy.constant"() {...}`,
because the custom syntax derives the result type from the attribute and cannot
express the mismatch. The second is expressible in the pretty syntax — a
well-typed constant that the target still cannot accept. Structural invariants
belong in constraints and traits; everything else belongs in `verify()`.
