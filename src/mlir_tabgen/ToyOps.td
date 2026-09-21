//===- ToyOps.td - The Toy dialect written in ODS ----------*- tablegen -*-===//
//
// The same dialect as ../mlir_dialect/toy.cpp -- three ops on f64 tensors --
// but declared in TableGen's Operation Definition Specification (ODS) instead
// of hand-written C++. `mlir-tblgen` turns this file into the op classes,
// accessors, builders, parser, printer, trait verifiers, and the dialect
// registration hook.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_OPS_TD
#define TOY_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// The dialect
//===----------------------------------------------------------------------===//

// `-gen-dialect-decls` turns this into the ToyDialect class, including the
// TypeID boilerplate that a hand-written dialect has to spell out with the
// MLIR_DECLARE/DEFINE_EXPLICIT_TYPE_ID macros.
def Toy_Dialect : Dialect {
  let name = "toy";
  let summary = "A minimal f64-tensor dialect, defined in ODS.";
  let description = [{
    Three operations -- `toy.constant`, `toy.add`, `toy.print` -- kept
    deliberately small so that the generated C++ stays readable.
  }];
  let cppNamespace = "::toy";
}

// Every op in a dialect shares this base class: it binds the op to its dialect
// and is the place to hang dialect-wide defaults.
class Toy_Op<string mnemonic, list<Trait> traits = []>
    : Op<Toy_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// toy.constant
//===----------------------------------------------------------------------===//

// %0 = toy.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
def Toy_ConstantOp : Toy_Op<"constant",
    [Pure, AllTypesMatch<["value", "result"]>]> {
  let summary = "materialize a constant f64 tensor";

  // Inherent attributes and operands go in `arguments`; ODS derives the
  // getValue() accessor and the "is this a 64-bit float elements attr" check
  // from the constraint itself.
  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor:$result);

  // The attribute is typed, so it already prints as `dense<...> : tensor<...>`;
  // AllTypesMatch lets the generated parser recover the result type from it
  // instead of asking for it a second time.
  let assemblyFormat = "attr-dict $value";

  // A convenience builder, so callers can write
  // `builder.create<ConstantOp>(loc, attr)` and let the op work out its own
  // result type. ODS still generates the explicit-result-type builders.
  let builders = [
    OpBuilder<(ins "::mlir::DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>
  ];

  // Invariants that no type constraint can express live in C++; see
  // ConstantOp::verify() in ToyDialect.cpp. Without this flag ODS does not
  // emit the call to it.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// toy.add
//===----------------------------------------------------------------------===//

// %2 = toy.add %0, %1 : tensor<2x2xf64>
def Toy_AddOp : Toy_Op<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "elementwise addition of two f64 tensors";

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor:$result);

  // SameOperandsAndResultType does three jobs at once: it verifies the shapes
  // agree (the hand-written version needed a verify() for that), it lets the
  // format print a single type, and it makes ODS emit a
  // `build(builder, state, lhs, rhs)` that infers the result type -- so this op
  // needs no `builders` block and no `hasVerifier`.
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// toy.print
//===----------------------------------------------------------------------===//

// toy.print %2 : tensor<2x2xf64>
def Toy_PrintOp : Toy_Op<"print"> {
  let summary = "print an f64 tensor to stdout";

  let arguments = (ins F64Tensor:$input);
  // No results, no custom builder, no verifier: the whole op is its signature.
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif // TOY_OPS_TD
