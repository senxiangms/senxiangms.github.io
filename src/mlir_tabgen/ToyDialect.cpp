//===- ToyDialect.cpp - The C++ that ODS cannot generate -----------------===//
//
// Three things live here, and nothing else:
//   1. the generated dialect definition (constructor, destructor, TypeID),
//   2. ToyDialect::initialize(), which registers the op list,
//   3. the one verifier that a type constraint cannot express.
//
// Compare with ../mlir_dialect/toy.cpp, where the same dialect needs ~280 lines
// of accessors, builders, parsers and printers before the driver starts.
//
//===----------------------------------------------------------------------===//

#include "ToyDialect.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

// From `mlir-tblgen -gen-dialect-defs`: the ToyDialect constructor and the
// MLIR_DEFINE_EXPLICIT_TYPE_ID that a hand-written dialect must write itself.
#include "ToyOpsDialect.cpp.inc"

void toy::ToyDialect::initialize() {
  // The op list is generated too, so adding an op to ToyOps.td is enough --
  // there is no second place to keep in sync.
  addOperations<
#define GET_OP_LIST
#include "ToyOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

// From `mlir-tblgen -gen-op-defs`: accessors, builders, parse(), print(),
// verifyInvariants() and the trait plumbing for all three ops.
#define GET_OP_CLASSES
#include "ToyOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstantOp custom verifier (`let hasVerifier = 1;`)
//===----------------------------------------------------------------------===//

// The generated verifyInvariants() already checked what ODS could state
// declaratively: the attribute is a 64-bit float elements attr, the result is
// an f64 tensor, and AllTypesMatch made the two agree. What it cannot state is
// a domain rule -- here, that the imaginary target only handles up to 2-D
// constants. Rules like that are exactly what hasVerifier is for; ODS calls
// this after the generated checks pass.
LogicalResult toy::ConstantOp::verify() {
  auto resultType = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be a ranked tensor, got ")
           << getResult().getType();

  if (resultType.getRank() > 2)
    return emitOpError("target supports at most 2-D constants, got rank ")
           << resultType.getRank();

  return success();
}
