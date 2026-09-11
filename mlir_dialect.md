---
layout: page
title: How to create a MLIR dialect using mlir::Dialect
---

# Creating an MLIR Dialect by Hand with `mlir::Dialect`

There are two ways to build a dialect. The [previous article](mlir_start) used ODS — you declare operations in TableGen and `mlir-tblgen` generates the C++. The other way is to derive from `mlir::Dialect` directly and write every piece yourself.

In production you will use ODS. But writing one dialect by hand is worth a day, because after that the generated `*.cpp.inc` files stop being a black box. When a build fails deep inside a template instantiation, or an op refuses to parse, you know what the generated code was supposed to look like and where to go.

This article walks through a complete hand-written dialect — three operations, custom syntax, verifiers — and then names the exact places where the compiler will stop you, because those are the parts ODS silently handles.

Everything here is verified against LLVM/MLIR 18; the full source is in [`src/mlir_dialect/toy.cpp`](https://github.com/senxiangms/senxiangms.github.io/blob/mlir/src/mlir_dialect/toy.cpp).

## What a dialect actually is

Strip away the terminology and `mlir::Dialect` is two things:

1. **A namespace.** The `toy` in `toy.constant`. It keeps your operation names from colliding with `arith`, `scf`, or anyone else's.
2. **A registry.** A table, owned by the `MLIRContext`, mapping mnemonics to the C++ classes that implement them — plus hooks for parsing your custom types and attributes.

It is not a compilation unit, not a pass, and not a container that owns IR. Operations belong to their parent `Operation`, and everything is ultimately owned by the context. A dialect object just tells the context how to interpret names.

The consequence: dialects are cheap, and mixing them in one module is normal, not exceptional. A single function can hold `func.func`, `scf.for`, `arith.addf`, and `toy.add` side by side. That is the whole point of MLIR — you lower one dialect into another gradually, and intermediate states are legal IR.

## The dialect class

Start with the declaration. This is exactly what the MLIR tutorial gives you:

```cpp
class ToyDialect : public mlir::Dialect {
public:
  explicit ToyDialect(mlir::MLIRContext *ctx);

  /// Provide a utility accessor to the dialect namespace.
  static llvm::StringRef getDialectNamespace() { return "toy"; }

  /// An initializer called from the constructor of ToyDialect that is used to
  /// register attributes, operations, types, and more within the Toy dialect.
  void initialize();
};
```

Three members, and each one is load-bearing.

**`getDialectNamespace()`** must be `static` and must return the prefix used by every operation name in the dialect. If `getOperationName()` on an op returns `"toy.add"` but the namespace is `"tiny"`, registration asserts at runtime — the mnemonic prefix is checked against the dialect that registers it.

**The constructor** hands three things to the base class:

```cpp
ToyDialect::ToyDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx,
                    mlir::TypeID::get<ToyDialect>()) {
  initialize();
}
```

The `TypeID` is how the context identifies your dialect at runtime — it is MLIR's home-grown RTTI, used because LLVM builds with `-fno-rtti`. Calling `initialize()` from the constructor rather than doing registration inline is a convention, but ODS follows it, so following it keeps hand-written and generated dialects looking the same.

**`initialize()`** is where everything gets registered:

```cpp
void ToyDialect::initialize() {
  addOperations<ConstantOp, AddOp, PrintOp>();
}
```

`addOperations` is variadic (`template <typename... Args>`) and has no type constraint on `Args`, which matters for error messages later. Real dialects call it once with a generated list:

```cpp
void ArithDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Arith/IR/ArithOps.cpp.inc"
      >();
}
```

Alongside it live `addTypes<>()`, `addAttributes<>()`, and `addInterfaces<>()`, all with the same shape.

### The TypeID macros

Here is the first thing ODS does for you that is easy to miss:

```cpp
MLIR_DECLARE_EXPLICIT_TYPE_ID(toy::ToyDialect)
MLIR_DEFINE_EXPLICIT_TYPE_ID(toy::ToyDialect)
```

Both must appear at **global scope**, outside any namespace, and they must both be present — the declare-only case is a link error. They give the dialect a `TypeID` backed by a single definition rather than a template-static, which is what makes `TypeID::get<ToyDialect>()` return the same value across shared library boundaries. Skip them and a dialect registered in one `.so` will not be found from another.

## Registration versus loading

A subtlety worth internalizing early, because it produces confusing runtime asserts:

```cpp
mlir::MLIRContext context;
context.getOrLoadDialect<toy::ToyDialect>();       // construct it now
```

versus:

```cpp
mlir::DialectRegistry registry;
registry.insert<toy::ToyDialect>();                // just record how to build it
mlir::MLIRContext context(registry);               // constructed on demand
```

**Registration** says "here is how to construct this dialect if someone needs it." **Loading** actually constructs it and populates the operation table. A `DialectRegistry` gives you lazy loading: the parser sees `toy.add`, looks up the `toy` prefix, and loads the dialect on demand. This is what `mlir-opt` does, and it is why an unused dialect costs nothing.

If you build IR programmatically, nothing triggers that lazy load — `builder.create<toy::AddOp>()` on a context where the dialect was never loaded asserts with "operation being parsed with an unregistered dialect". Hence the explicit `getOrLoadDialect` in a hand-written driver.

## Defining an operation by hand

An operation class derives from `mlir::Op` via CRTP, with traits as extra template arguments:

```cpp
class AddOp : public mlir::Op<AddOp, mlir::OpTrait::NOperands<2>::Impl,
                              mlir::OpTrait::OneResult,
                              mlir::OpTrait::ZeroRegions> {
public:
  using Op::Op;                                   // inherit constructors

  static llvm::StringRef getOperationName() { return "toy.add"; }
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

  mlir::Value getLhs() { return getOperand(0); }
  mlir::Value getRhs() { return getOperand(1); }
  mlir::Value getResult() { return (*this)->getResult(0); }

  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Value lhs, mlir::Value rhs);
  mlir::LogicalResult verify();
  static mlir::ParseResult parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result);
  void print(mlir::OpAsmPrinter &printer);
};
```

The critical property, and the reason `using Op::Op` is enough: **an Op class is a value, not a pointer.** It holds exactly one `Operation *` and nothing else — `mlir::Op` enforces this with a `static_assert(hasNoDataMembers())`. You pass op classes around by value, and `(*this)->` reaches the underlying `Operation`. Adding a member variable to your op class is a compile error, which surprises people once.

Traits are mixins that inject verification and sometimes accessors. `NOperands<2>::Impl` verifies the operand count; `ZeroRegions` verifies there are no regions. They run *before* your `verify()`, so by the time your code executes, the structure is already guaranteed and `getOperand(0)` is safe.

### What an op class must provide

This is the implicit concept `addOperations<>` requires. Miss one and you get a template instantiation error, not a clean diagnostic:

| Member | Required | Purpose |
|---|---|---|
| `getOperationName()` | always | mnemonic; prefix must match the dialect |
| `getAttributeNames()` | **always** | inherent attribute names; return `{}` if none |
| `using Op::Op` | always | inherit the `Operation *` constructor |
| `build(...)` | to use `builder.create<>` | populate `OperationState` |
| `verify()` | optional | op-specific invariants |
| `parse` / `print` | optional | custom syntax; without both you get generic form |

## Verification: two layers

`mlir::verify()` on an op runs this, from `OpDefinition.h`:

```cpp
static LogicalResult verifyInvariants(Operation *op) {
  return failure(
      failed(op_definition_impl::verifyTraits<Traits<ConcreteType>...>(op)) ||
      failed(cast<ConcreteType>(op).verify()));
}
```

Traits first, then yours. The division of labour:

- **Traits** check structure — operand count, result count, region count, terminator-ness.
- **`verify()`** checks semantics — type relationships, attribute validity, value ranges.

`OpState` supplies a default `verify()` returning `success()`, which your definition shadows. Note it is *not* virtual: this is CRTP static dispatch, resolved at compile time through `cast<ConcreteType>(op).verify()`.

A real verifier:

```cpp
mlir::LogicalResult ConstantOp::verify() {
  auto value = getValue();
  if (!value)
    return emitOpError("requires a 'value' DenseElementsAttr");

  auto resultType =
      llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be a ranked tensor, got ")
           << getResult().getType();

  if (value.getType() != resultType)
    return emitOpError("attribute type ")
           << value.getType() << " does not match result type " << resultType;

  return mlir::success();
}
```

`emitOpError` does three jobs in one call: it prefixes the op name, attaches the source location, and returns a value convertible to `LogicalResult`. Streaming MLIR objects into it with `<<` works for `Type`, `Attribute`, and `Value`. The output:

```
loc("-":3:8): error: 'toy.constant' op attribute type 'tensor<2xf64>'
              does not match result type 'tensor<2x2xf64>'
```

Do not write an empty `verify()`. If traits cover the op's entire contract — as with `PrintOp`, which is just `OneOperand + ZeroResults` — omit it and let the default run.

Verification fires at three moments: on explicit `mlir::verify()`, after every pass in a `PassManager`, and after parsing. It does **not** fire on `builder.create<>()`, which is deliberate — rewrite patterns routinely pass through invalid intermediate states.

## Custom assembly syntax

Without `parse`/`print`, an op prints in generic form:

```mlir
%2 = "toy.add"(%0, %1) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
```

That is valid and round-trips fine, but it is unpleasant to read and write in tests. The custom form is a pair of functions that must be exact mirrors:

```cpp
mlir::ParseResult AddOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand lhs, rhs;
  mlir::Type type;
  if (parser.parseOperand(lhs) || parser.parseComma() ||
      parser.parseOperand(rhs) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  if (parser.resolveOperand(lhs, type, result.operands) ||
      parser.resolveOperand(rhs, type, result.operands))
    return mlir::failure();

  result.addTypes(type);
  return mlir::success();
}

void AddOp::print(mlir::OpAsmPrinter &printer) {
  printer << " " << getLhs() << ", " << getRhs();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getResult().getType();
}
```

Two idioms to absorb. Parser methods return `ParseResult`, which is truthy on *failure*, so chaining with `||` reads as "if any step fails, bail" — inverted from normal C++ intuition, and universal across MLIR. And parsing happens in two phases: `parseOperand` collects an `UnresolvedOperand` (just a name and location), then `resolveOperand` binds it to a type and appends it to `result.operands`. You cannot resolve until you have parsed the type, which is why the type comes last in the syntax.

The reason to care about round-tripping is testing. MLIR's entire test methodology is `mlir-opt %s | FileCheck %s` — if printer and parser disagree, you cannot write a test for any pass that touches your dialect.

There is also a lesson hidden in `ConstantOp`'s syntax. Its parser derives the result type from the attribute:

```cpp
result.addTypes(value.getType());
```

which means a mismatch between the two is *unrepresentable* in the custom form. To exercise the verifier you have to fall back to generic syntax:

```mlir
%0 = "toy.constant"() {value = dense<[1.0, 2.0]> : tensor<2xf64>}
     : () -> tensor<2x2xf64>
```

This is precisely why invariants belong in `verify()` and not in the parser. The parser only sees one of several ways an op can be constructed; the verifier sees them all.

## The four things that will stop your build

These are the actual compile and link errors from writing this dialect by hand, in the order they appeared. Every one is something ODS generates silently.

**1. Missing TypeID macros.** Symptom: link error on `TypeIDResolver<toy::ToyDialect>::id`, or a dialect that mysteriously fails to resolve across library boundaries. Fix: both `MLIR_DECLARE_EXPLICIT_TYPE_ID` and `MLIR_DEFINE_EXPLICIT_TYPE_ID` at global scope.

**2. Missing `getAttributeNames()`.** There is no default. The error surfaces from inside the registry:

```
error: no member named 'getAttributeNames' in 'toy::AddOp'
  684 |  insert(std::make_unique<Model<T>>(&dialect), T::getAttributeNames());
note: in instantiation of function template specialization
      'mlir::Dialect::addOperations<toy::ConstantOp, toy::AddOp, toy::PrintOp>'
```

Because `addOperations` is unconstrained, the diagnostic points at the registry rather than at your class. The useful line is the `note:` — it names which type in the parameter pack failed. Fix: `static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }`.

**3. `OpTrait::OneResult` does not give you `getResult()`.** Reading the trait is surprising:

```cpp
template <typename ConcreteType>
class OneResult : public TraitBase<ConcreteType, OneResult> {
public:
  void replaceAllUsesWith(Value newValue) { ... }
  static LogicalResult verifyTrait(Operation *op) { ... }
};
```

It supplies `replaceAllUsesWith` and the trait verifier — no accessor. `getResult()` lives on `OpTrait::OneTypedResult<T>::Impl`, a *different* trait that also provides implicit conversion to `Value`.

You have two options. Add `OneTypedResult<TensorType>::Impl` before `OneResult` in the trait list, which is what ODS does when the result type is constrained — but note it `cast`s internally, so a wrong result type asserts *before* your verifier can produce a readable message. Or declare your own accessor, which is what the example does:

```cpp
mlir::Value getResult() { return (*this)->getResult(0); }
```

**4. Forgetting an op in `initialize()`.** This one compiles cleanly and fails at runtime, which makes it the worst of the four. `addOperations<>` is the only thing that populates the registry; an op left out asserts on first `create<>` or fails to parse with "unregistered operation".

## Beyond operations

`mlir::Dialect` has a set of virtual hooks worth knowing about, even if a first dialect uses none of them:

| Hook | When you need it |
|---|---|
| `parseType` / `printType` | custom types, e.g. `!toy.struct<...>` — required once you call `addTypes<>()` |
| `parseAttribute` / `printAttribute` | custom attributes |
| `materializeConstant` | lets canonicalization rematerialize a folded constant as your op |
| `getCanonicalizationPatterns` | dialect-wide patterns not attached to a single op |
| `verifyOperationAttribute` | validate discardable attributes in your namespace on *other* dialects' ops |

The last one is subtle and useful: it lets `toy.something = 42` be attached to an `scf.for` and still be validated by your dialect. That is the mechanism behind things like target-specific annotations surviving on generic ops.

## When to hand-write, and when not to

The same three ops in ODS are roughly 25 lines of TableGen. The hand-written file is 387 lines. That ratio is the argument, and it gets worse with scale — ODS also generates fold hooks, effect interfaces, and builders you did not think to write.

Hand-write when:

- **You are learning.** Once. The generated code becomes readable afterward, and that pays off every time a tblgen build breaks.
- **The op resists declaration.** Highly dynamic operand structures, or parsing that needs real lookahead, sometimes end up shorter in C++ than in `assemblyFormat` plus a custom parser hook.
- **You are debugging registration.** Reproducing a problem in a small hand-written dialect isolates whether the bug is in your `.td` or in your understanding.

Otherwise use ODS. Mixed dialects are fine — ODS ops and hand-written ops register through the same `addOperations<>` call and are indistinguishable afterward.

## Building it

An out-of-tree dialect needs `find_package(MLIR CONFIG)` and LLVM's CMake modules:

```cmake
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_llvm_executable(toy toy.cpp)
llvm_update_compile_flags(toy)
target_link_libraries(toy PRIVATE MLIRIR MLIRParser MLIRSupport MLIRFuncDialect)
mlir_check_all_link_libraries(toy)
```

On Ubuntu 24.04:

```bash
sudo apt-get install -y libmlir-18-dev mlir-18-tools
cmake -G Ninja -S . -B build -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
ninja -C build && ./build/toy
```

`llvm_update_compile_flags` matters more than it looks: MLIR headers are compiled with `-fno-rtti`, and linking a `-frtti` translation unit against them produces link errors on vtables. The macro copies LLVM's flags onto your target.

The `Could NOT find LibEdit / zstd / CURL` lines during configuration are LLVM probing its own optional dependencies. They are `--` status messages, not warnings, and are irrelevant to an out-of-tree dialect.

## What the example proves, and what it does not

The driver does four things, and it is worth being precise about what each one demonstrates:

1. **Builds IR with the C++ API** — the path a language front end takes.
2. **Parses the same IR from text** — proves `parse` and `print` are symmetric.
3. **Walks it with a typed callback** — `parsed->walk([](toy::ConstantOp op) {...})` — the path an optimization pass takes.
4. **Fails verification on bad IR** — proves the constraints are live.

What it does **not** do is compute anything. There is a `toy.add` over `[[1,2],[3,4]]` and `[[10,20],[30,40]]`, and the program never produces `[[11,22],[33,44]]`. It builds, checks, and prints a *representation* of a program — nothing executes.

That gap is the whole remaining compiler. Mapping back to the five artifacts from the [previous article](mlir_start): this covers the type system and part of the surface syntax. The lowering pipeline, the code generator, and the runtime are all still ahead. In the bring-up ordering, a dialect that round-trips under `mlir-opt` is step 2 of seven — a day's work, and the day that forces your ISA constraints into writing.

## Takeaways

`mlir::Dialect` is a namespace plus a registry, and an operation is a value wrapping a single `Operation *`. Everything else — traits, verifiers, parsers — is machinery layered on those two facts.

Write one dialect by hand to learn where the seams are: the TypeID macros, the mandatory `getAttributeNames()`, the accessor that `OneResult` does not give you, and the `initialize()` list that fails at runtime rather than compile time. Then switch to ODS and never write those 360 lines again — but read the generated code once, because now you can.
