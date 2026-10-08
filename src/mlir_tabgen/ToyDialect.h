//===- ToyDialect.h - Public interface of the ODS-defined Toy dialect -----===//
//
// This header is the whole hand-written interface of the dialect: it pulls in
// the two headers that `mlir-tblgen` produced from ToyOps.td. Everything a user
// of the dialect needs -- the ToyDialect class and the ConstantOp / AddOp /
// PrintOp classes with their accessors and builders -- comes from those.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_TOYDIALECT_H
#define TOY_TOYDIALECT_H

// Generated op classes reference these; include them before the .inc files.
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// From `mlir-tblgen -gen-dialect-decls`: class toy::ToyDialect.
#include "ToyOpsDialect.h.inc"

// From `mlir-tblgen -gen-op-decls`: one class per op. GET_OP_CLASSES selects
// the declarations out of the same .inc file that also holds the op list.
#define GET_OP_CLASSES
#include "ToyOps.h.inc"

#endif // TOY_TOYDIALECT_H
