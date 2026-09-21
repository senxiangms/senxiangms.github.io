//===- toy.cpp - Driver for the ODS-defined Toy dialect ------------------===//
//
// The same driver as ../mlir_dialect/toy.cpp: it builds IR with the builder
// API, parses the identical IR back from text, walks it with a typed op class,
// and shows both verifier layers rejecting bad IR. Nothing in this file knows
// that the dialect came from TableGen -- which is the point.
//
//===----------------------------------------------------------------------===//

#include "ToyDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

/// Build `func.func @built() { %0 = c1; %1 = c2; %2 = %0 + %1; print %2 }`
/// entirely through the C++ builder API -- using the builders ODS generated.
static mlir::OwningOpRef<mlir::ModuleOp> buildModule(mlir::MLIRContext &ctx) {
  mlir::OpBuilder builder(&ctx);
  mlir::Location loc = builder.getUnknownLoc();

  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module->getBody());

  auto tensorTy = mlir::RankedTensorType::get({2, 2}, builder.getF64Type());
  auto funcTy = builder.getFunctionType(/*inputs=*/{}, /*results=*/{});
  auto func = builder.create<mlir::func::FuncOp>(loc, "built", funcTy);
  builder.setInsertionPointToStart(func.addEntryBlock());

  const double lhsData[] = {1.0, 2.0, 3.0, 4.0};
  const double rhsData[] = {10.0, 20.0, 30.0, 40.0};
  auto lhsAttr =
      mlir::DenseElementsAttr::get(tensorTy, llvm::ArrayRef<double>(lhsData, 4));
  auto rhsAttr =
      mlir::DenseElementsAttr::get(tensorTy, llvm::ArrayRef<double>(rhsData, 4));

  // ConstantOp: the `builders` block in ToyOps.td gave us this one-argument
  // overload, which derives the result type from the attribute.
  auto lhs = builder.create<toy::ConstantOp>(loc, lhsAttr);
  auto rhs = builder.create<toy::ConstantOp>(loc, rhsAttr);
  // AddOp: no `builders` block at all. SameOperandsAndResultType made ODS emit
  // a build() that takes just the operands and reuses their type.
  auto sum = builder.create<toy::AddOp>(loc, lhs.getResult(), rhs.getResult());
  builder.create<toy::PrintOp>(loc, sum.getResult());
  builder.create<mlir::func::ReturnOp>(loc);

  return module;
}

/// The same program in textual form. The syntax is byte-for-byte the syntax of
/// the hand-written dialect, except that here it came from two
/// `assemblyFormat` strings instead of six parse/print methods.
static const char *kToySource = R"mlir(
func.func @parsed() {
  %0 = toy.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
  %1 = toy.constant dense<[[10.0, 20.0], [30.0, 40.0]]> : tensor<2x2xf64>
  %2 = toy.add %0, %1 : tensor<2x2xf64>
  toy.print %2 : tensor<2x2xf64>
  return
}
)mlir";

/// Parse `source` and return true if it fails to build or to verify.
static bool rejects(mlir::MLIRContext &ctx, const char *source) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &ctx);
  return !module || mlir::failed(mlir::verify(*module));
}

int main() {
  mlir::MLIRContext context;
  // The generated dialect is loaded exactly like a hand-written one.
  context.getOrLoadDialect<toy::ToyDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();

  // 1. Build IR with the generated builders.
  mlir::OwningOpRef<mlir::ModuleOp> built = buildModule(context);
  if (mlir::failed(mlir::verify(*built))) {
    llvm::errs() << "error: built module failed verification\n";
    return 1;
  }
  llvm::outs() << "=== built programmatically ===\n";
  built->print(llvm::outs());

  // 2. Parse the same IR from text, through the generated parser.
  mlir::OwningOpRef<mlir::ModuleOp> parsed =
      mlir::parseSourceString<mlir::ModuleOp>(kToySource, &context);
  if (!parsed) {
    llvm::errs() << "error: failed to parse toy source\n";
    return 1;
  }
  if (mlir::failed(mlir::verify(*parsed))) {
    llvm::errs() << "error: parsed module failed verification\n";
    return 1;
  }
  llvm::outs() << "\n=== parsed from text, then reprinted ===\n";
  parsed->print(llvm::outs());

  // 3. Walk the IR through the generated typed accessors.
  llvm::outs() << "\n=== walk ===\n";
  parsed->walk([](toy::ConstantOp op) {
    llvm::outs() << "toy.constant with " << op.getValue().getNumElements()
                 << " elements of type " << op.getValue().getElementType()
                 << "\n";
  });

  // 4a. A type mismatch between attribute and result. In the hand-written
  // dialect this needed a verify() body; here AllTypesMatch rejects it, and
  // only the generic form can even express it, because the custom syntax
  // derives the result type from the attribute.
  llvm::outs() << "\n=== trait verifier (the error below is expected) ===\n";
  llvm::outs().flush();
  if (!rejects(context, R"mlir(
func.func @bad_types() {
  %0 = "toy.constant"() {value = dense<[1.0, 2.0]> : tensor<2xf64>}
       : () -> tensor<2x2xf64>
  return
}
)mlir")) {
    llvm::errs() << "error: mistyped constant unexpectedly verified\n";
    return 1;
  }
  llvm::outs() << "(rejected as expected)\n";

  // 4b. A well-typed constant that breaks the domain rule. No ODS constraint
  // can state "rank <= 2", so this is the case that reaches
  // ConstantOp::verify() -- and the custom syntax can express it, unlike 4a.
  llvm::outs() << "\n=== custom verifier (the error below is expected) ===\n";
  llvm::outs().flush();
  if (!rejects(context, R"mlir(
func.func @bad_rank() {
  %0 = toy.constant dense<1.0> : tensor<2x2x2xf64>
  return
}
)mlir")) {
    llvm::errs() << "error: rank-3 constant unexpectedly verified\n";
    return 1;
  }
  llvm::outs() << "(rejected as expected)\n";
  return 0;
}
