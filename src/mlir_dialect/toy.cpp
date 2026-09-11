//===- toy.cpp - A hand-written MLIR dialect, end to end ------------------===//
//
// A minimal but complete MLIR dialect written by hand in C++ (no ODS/TableGen),
// with three operations, custom assembly syntax, verifiers, and a driver that
// both builds IR programmatically and parses it back from text.
//
// Build: see CMakeLists.txt in this directory.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/TypeID.h"
#include "llvm/Support/raw_ostream.h"

namespace toy {

//===----------------------------------------------------------------------===//
// The dialect
//===----------------------------------------------------------------------===//

class ToyDialect : public mlir::Dialect {
public:
  explicit ToyDialect(mlir::MLIRContext *ctx);

  /// Provide a utility accessor to the dialect namespace.
  static llvm::StringRef getDialectNamespace() { return "toy"; }

  /// An initializer called from the constructor of ToyDialect that is used to
  /// register attributes, operations, types, and more within the Toy dialect.
  void initialize();
};

//===----------------------------------------------------------------------===//
// toy.constant
//===----------------------------------------------------------------------===//

/// %0 = toy.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
class ConstantOp
    : public mlir::Op<ConstantOp, mlir::OpTrait::ZeroOperands,
                      mlir::OpTrait::OneResult, mlir::OpTrait::ZeroRegions> {
public:
  /// Inherit the constructors from the base Op class.
  using Op::Op;

  /// The unique mnemonic; the dialect prefix must match the dialect namespace.
  static llvm::StringRef getOperationName() { return "toy.constant"; }

  /// Attribute names this op may carry; used when the op is registered.
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() {
    static llvm::StringRef names[] = {llvm::StringRef("value")};
    return llvm::ArrayRef<llvm::StringRef>(names);
  }

  /// OpTrait::OneResult supplies replaceAllUsesWith but not an accessor, so
  /// hand-written ops declare their own. (ODS generates these for you.)
  mlir::Value getResult() { return (*this)->getResult(0); }

  /// Typed accessor for the constant payload.
  mlir::DenseElementsAttr getValue() {
    return (*this)->getAttrOfType<mlir::DenseElementsAttr>("value");
  }

  /// Invoked by OpBuilder::create<ConstantOp>(loc, value).
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::DenseElementsAttr value);

  /// Op-specific invariants, checked on top of the trait verifiers.
  mlir::LogicalResult verify();

  /// Custom textual syntax (without these the op prints in generic form).
  static mlir::ParseResult parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result);
  void print(mlir::OpAsmPrinter &printer);
};

//===----------------------------------------------------------------------===//
// toy.add
//===----------------------------------------------------------------------===//

/// %2 = toy.add %0, %1 : tensor<2x2xf64>
class AddOp : public mlir::Op<AddOp, mlir::OpTrait::NOperands<2>::Impl,
                              mlir::OpTrait::OneResult,
                              mlir::OpTrait::ZeroRegions> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "toy.add"; }

  /// Required by op registration; this op carries no inherent attributes.
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

//===----------------------------------------------------------------------===//
// toy.print
//===----------------------------------------------------------------------===//

/// toy.print %2 : tensor<2x2xf64>
class PrintOp : public mlir::Op<PrintOp, mlir::OpTrait::OneOperand,
                                mlir::OpTrait::ZeroResults,
                                mlir::OpTrait::ZeroRegions> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "toy.print"; }

  /// Required by op registration; this op carries no inherent attributes.
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

  mlir::Value getInput() { return getOperand(); }

  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Value input);

  static mlir::ParseResult parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result);
  void print(mlir::OpAsmPrinter &printer);
};

} // namespace toy

/// A dialect needs a stable TypeID. ODS (Operation Definition Specification) emits these two macros for you; when
/// hand-writing the class you must supply them yourself. They must appear at
/// global scope, and the DECLARE/DEFINE pair must both be present.
MLIR_DECLARE_EXPLICIT_TYPE_ID(toy::ToyDialect)
MLIR_DEFINE_EXPLICIT_TYPE_ID(toy::ToyDialect)

namespace toy {

//===----------------------------------------------------------------------===//
// ToyDialect implementation
//===----------------------------------------------------------------------===//

ToyDialect::ToyDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx,
                    mlir::TypeID::get<ToyDialect>()) {
  initialize();
}

void ToyDialect::initialize() {
  // Every op must be registered here, or building one asserts at runtime.
  addOperations<ConstantOp, AddOp, PrintOp>();
}

//===----------------------------------------------------------------------===//
// ConstantOp implementation
//===----------------------------------------------------------------------===//

void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       mlir::DenseElementsAttr value) {
  state.addTypes(value.getType());
  state.addAttribute("value", value);
}

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

mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return mlir::failure();

  // The result type is implied by the attribute's own type.
  result.addTypes(value.getType());
  return mlir::success();
}

void ConstantOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}

//===----------------------------------------------------------------------===//
// AddOp implementation
//===----------------------------------------------------------------------===//

void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(lhs.getType());
  state.addOperands({lhs, rhs});
}

mlir::LogicalResult AddOp::verify() {
  if (getLhs().getType() != getRhs().getType())
    return emitOpError("operand types must match: ")
           << getLhs().getType() << " vs " << getRhs().getType();
  if (getResult().getType() != getLhs().getType())
    return emitOpError("result type must match operand type");
  return mlir::success();
}

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

//===----------------------------------------------------------------------===//
// PrintOp implementation
//===----------------------------------------------------------------------===//

void PrintOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Value input) {
  state.addOperands(input);
}

mlir::ParseResult PrintOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand input;
  mlir::Type type;
  if (parser.parseOperand(input) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  return parser.resolveOperand(input, type, result.operands);
}

void PrintOp::print(mlir::OpAsmPrinter &printer) {
  printer << " " << getInput();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getInput().getType();
}

} // namespace toy

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

/// Build `func.func @built() { %0 = c1; %1 = c2; %2 = %0 + %1; print %2 }`
/// entirely through the C++ builder API.
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
  auto lhsAttr = mlir::DenseElementsAttr::get(
      tensorTy, llvm::ArrayRef<double>(lhsData, 4));
  auto rhsAttr = mlir::DenseElementsAttr::get(
      tensorTy, llvm::ArrayRef<double>(rhsData, 4));

  auto lhs = builder.create<toy::ConstantOp>(loc, lhsAttr);
  auto rhs = builder.create<toy::ConstantOp>(loc, rhsAttr);
  auto sum = builder.create<toy::AddOp>(loc, lhs.getResult(), rhs.getResult());
  builder.create<toy::PrintOp>(loc, sum.getResult());
  builder.create<mlir::func::ReturnOp>(loc);

  return module;
}

/// The same program in textual form, to prove the custom parser and printer
/// round-trip.
static const char *kToySource = R"mlir(
func.func @parsed() {
  %0 = toy.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
  %1 = toy.constant dense<[[10.0, 20.0], [30.0, 40.0]]> : tensor<2x2xf64>
  %2 = toy.add %0, %1 : tensor<2x2xf64>
  toy.print %2 : tensor<2x2xf64>
  return
}
)mlir";

int main() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<toy::ToyDialect>(); // create a new dialect (constructor will register the operations)
  context.getOrLoadDialect<mlir::func::FuncDialect>(); // load the built-in func dialect, because kToySource uses func.func, func.return. 

  // 1. Build IR with the builder API.
  mlir::OwningOpRef<mlir::ModuleOp> built = buildModule(context); // build the IR programmatically (MLIR-dialect program is is hardcoded in C++ code)
  if (mlir::failed(mlir::verify(*built))) {
    llvm::errs() << "error: built module failed verification\n";
    return 1;
  }
  llvm::outs() << "=== built programmatically ===\n";
  built->print(llvm::outs());

  // 2. Parse the same IR from text.
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

  // 3. Walk the IR and read a typed accessor, to show the op class in use.
  llvm::outs() << "\n=== walk ===\n";
  parsed->walk([](toy::ConstantOp op) {
    llvm::outs() << "toy.constant with " << op.getValue().getNumElements()
                 << " elements of type " << op.getValue().getElementType()
                 << "\n";
  });

  // 4. Demonstrate that ConstantOp::verify() actually rejects bad IR. The
  // custom syntax derives the result type from the attribute, so it cannot
  // express this mistake -- the generic form can, which is exactly why the
  // verifier and not the parser is the place to enforce invariants.
  llvm::outs() << "\n=== verifier check (the error below is expected) ===\n";
  llvm::outs().flush();
  mlir::OwningOpRef<mlir::ModuleOp> bad =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
func.func @bad() {
  %0 = "toy.constant"() {value = dense<[1.0, 2.0]> : tensor<2xf64>}
       : () -> tensor<2x2xf64>
  return
}
)mlir",
                                             &context);
  if (bad && mlir::succeeded(mlir::verify(*bad))) {
    llvm::errs() << "error: bad IR unexpectedly verified\n";
    return 1;
  }
  llvm::outs() << "(rejected as expected)\n";
  return 0;
}
