---
layout: page
title: TileLang Frontend. How is Tilelang DSL code tranformed into TIR (Tensor IR)?
---

# TileLang Frontend: turning Python into TIR by tracing

The [TileLang introduction](tilelang_intro) showed *what* a kernel looks like — tiles moving through explicit memory scopes. This note is about the machinery that turns that Python source into TVM TIR: the **eager frontend**, built around three pieces in `tilelang/language/eager/` — `mutate`, `DSLMutator`, and `IRGenerator`.

## Two ways to read a Python kernel

A Python-embedded DSL has to get from Python source to a compiler IR. There are two classic strategies:

- **Parse** — walk the Python AST yourself and translate each node into IR, node by node. This is what TVMScript (which TileLang inherits from TVM) does: a big visitor with `visit_For`, `visit_Assign`, `visit_If`, each emitting TIR directly. The kernel body is *never executed as Python*; it is read as data.
- **Trace** — rewrite the Python source so that every statement becomes a call to a *builder object* (__tb.*), then **run** the rewritten function. Control flow, arithmetic, and name binding all pass through the builder, which decides — at run time — whether each piece is compile-time Python or a piece of IR to emit.

TileLang's eager frontend takes the **trace** route. The payoff is that ordinary Python metaprogramming just works: a loop over a Python `range` with a constant bound unrolls, an `if` on a Python `bool` is specialized away, closures and tuple-unpacking behave exactly as Python does — because the same Python interpreter is running the body. Only the parts that touch tensors and TIR expressions turn into IR.

The whole scheme rests on one trick: **source-to-source rewriting so the builder intercepts everything.**

## The core idea: rewrite Python into builder calls

Every kernel body is transformed so that each syntactic construct becomes a method call on a builder, conventionally named `__tb` (trace builder?). A rough before/after (simplified — the real output is more verbose):

```python
# ---- what you write ----
for i in range(N):
    B[i] = A[i] + 1.0

# ---- what the frontend runs (conceptually) ----
for __0 in __tb.ctx_for(range(N)):
    i = __tb.bind('i', __0)
    __tb.assign_slice(B, i, __tb.rval('A', A)[i] + 1.0)
```

Now `__tb` is in control. If `__tb` is a plain interpreter, this runs like normal Python. If `__tb` is TileLang's TIR builder, `ctx_for` opens a TIR loop frame, `bind` names the loop variable, and `assign_slice` emits a `BufferStore` — the loop body is *traced* into IR instead of executed.

Three components implement this:

| Component | Role |
|-----------|------|
| `DSLMutator` | the AST rewriter — turns Python syntax into `__tb.*` calls |
| `mutate` | the driver — runs the rewrite and compiles the result into a callable |
| `IRGenerator` | the product — a callable `builder → traced function`, plus its source |

## `mutate` — the entry point

`mutate(func)` is the top-level transform. Given a Python function it:

1. Gets the function's AST and its **closure variables** (`nonlocals`) and source filename.
2. Builds a `DSLMutator(nonlocals, func.__globals__, filename)` and runs `mut.visit(tree)`.
3. Compiles the rewritten tree and wraps it in an `IRGenerator`.

```python
def mutate(func):
    tree     = utils.get_ast(func)
    filename = inspect.getsourcefile(func) or inspect.getfile(func)
    nonlocals = utils.get_func_nonlocals(func)

    mut  = DSLMutator(nonlocals, func.__globals__, str(Path(filename).absolute()))
    tree = mut.visit(tree)
    make_closure = utils.get_compiled_object(tree, "make_closure", filename, func.__globals__)
    fn = make_closure(**nonlocals)
    return IRGenerator(gen=fn, source=ast.unparse(tree), extra_type_hints=mut.extra_type_hints)
```

Notice the rewritten tree does not just redefine the kernel — it defines a wrapper called `make_closure`. `DSLMutator.visit_FunctionDef` emits:

```python
def make_closure(<all closure vars>):
    def add(__tb):
        ...
        return add
    return add
```

Why the extra layer? Closure variables must be injected somewhere the rewritten code can see them. The tempting shortcut — merge them into a copy of the function's globals — is a memory-leak trap: copying `__globals__` creates a fresh dict that pins a reference to the entire original module namespace, which can then never be freed. The source comment in `mutate` spells this out. Passing the closure variables as `make_closure` **parameters** keeps them in a local scope and lets the original globals stay shared and collectable. This same care shows up in `BaseBuilder.get_parent_locals`, which deletes its frame reference in a `finally` to avoid a reference cycle through `f_locals`.

## `DSLMutator` — the AST rewriter

`DSLMutator` is an `ast.NodeTransformer`. Each `visit_*` method rewrites one kind of node into builder calls. The rewrites route through a small helper, `quote()`, which parses a template string and splices AST nodes into `Name` and `Pass` placeholders — so the transformer reads almost like the target code it produces.

**A "builder call" is just a method call on `__tb`.** The transformer never turns a Python node directly into TIR; it turns it into a call on the builder object threaded through the rewritten function (`def add(__tb): …`), and defers the real decision to that object at run time. The vocabulary is fixed — about twenty hooks declared on `BaseBuilder` (`ctx_if`, `ctx_for`, `bind`, `assign_slice`, `rval`, `arg`, `boolop`, `ret`, …). Every construct is expressed in terms of them.

Why a *call* and not a direct translation? Because "is this compile-time Python or a piece of IR?" cannot be answered from the AST alone — it depends on the *runtime type of the value*. In `for i in range(N)`, whether `N` is a Python `int` (unroll) or a symbolic dimension (emit a TIR loop) is unknowable statically, so the transformer emits `__tb.ctx_for(range(N))` and lets the method inspect the actual value. The rewrite is static and mechanical; the semantics are dynamic and live in the builder. That is exactly why the same rewritten function can either run as plain Python (under `BaseBuilder`) or emit TIR (under `Builder`) — the `visit_*` methods only choose *which* hook each construct calls; the builder decides what that call *means* (see *BaseBuilder vs Builder* below).

Here is the map from Python to builder hooks:

| Python construct | Rewritten to | Builder hook |
|------------------|--------------|--------------|
| `if c: …` | `for br in ctx_if(c): for _ in ctx_then(br): …` | `ctx_if` / `ctx_then` / `ctx_else` |
| `for x in it: …` | `for tmp in ctx_for(it): <bind x> …` | `ctx_for` |
| `while c: …` | `for _ in ctx_while(lambda: c): …` | `ctx_while` |
| `a = v` | `a = bind('a', v)` | `bind` |
| `A[i] = v` | `assign_slice(A, i, v)` | `assign_slice` |
| `a += v` | `a = aug_assign('Add', a, v, name='a')` | `aug_assign` |
| `f(x)` (stmt) | `eval(f(x))` | `eval` |
| name load `a` | `rval('a', a)` | `rval` |
| `a and b`, `not a` | `boolop('And', a, lambda: b)` | `boolop` |
| `a < b < c` | split into `And(a<b, b<c)` | `boolop` |
| `x if c else y` | `ifexp(c, lambda: x, lambda: y)` | `ifexp` |
| `return v` | `return ret(v)` | `ret` |
| `with ctx: …` | wrap `ctx` in `ctx_with(ctx)` | `ctx_with` |
| `assert c, m` | `assert_expr(c, m)` | `assert_expr` |
| `break` / `continue` | guarded by `ctx_break()` / `ctx_continue()` | `ctx_break` / `ctx_continue` |

A few of the rewrites are worth dwelling on because they encode design decisions, not just mechanics.

**`if` becomes a loop.** An `if` is rewritten as a `for` over `ctx_if(cond)`:

```python
def visit_If(self, node):
    node = self.generic_visit(node)
    br = self.get_tmp()
    return quote(
        f"for {br} in __tb.ctx_if(cond):\n"
        f"  for _ in __tb.ctx_then({br}):\n    pass\n"
        f"  for _ in __tb.ctx_else({br}):\n    pass\n",
        cond=node.test, passes=[node.body, node.orelse], span=node)
```

The `for`-over-generator shape lets the builder decide *dynamically* whether to enter the body. If `cond` is a compile-time Python value, `ctx_if` yields the plain value and the body runs (or does not) as normal Python — the branch is specialized at trace time and leaves no trace in the IR. If `cond` is a TIR `PrimExpr`, `ctx_if` opens a TIR `If` frame and `ctx_then`/`ctx_else` open the `Then`/`Else` frames, so the branch is emitted as real control flow. Same source, two outcomes, chosen by the value's type.

**Short-circuit operators keep their laziness.** `a and b` becomes `boolop('And', a, lambda: b)` — the right operand is wrapped in a lambda so it is only evaluated if needed. Chained comparisons like `a < b < c` are split into `And(a < b, b < c)`. This preserves Python's evaluation semantics while giving the builder a chance to fold them into TIR `And`/`Or` nodes.

**Tuple unpacking is two-phase.** To make `a, b = b, a` behave like Python's simultaneous assignment, `_emit_assign_target` first binds all right-hand values to temporaries, then binds the targets from those temporaries — so a swap works even when the builder is materializing each binding as a TIR let or store.

**Parameters and `range` are rebound.** `visit_FunctionDef` injects, for each parameter, a `name = __tb.arg("name", name)` binding at the top of the body, strips the original annotations and decorators, and adds `range = __tb.override('range')` so a bare `range` inside a kernel resolves to `T.serial`. It also *scans* the parameter annotations (`A: T.Tensor(...)`, `x: T.float32`) and records them in `extra_type_hints`, which the `prim_func` decorator later uses to build the buffer/scalar arguments.

**Every statement gets a source span.** A companion `SpanAttacher` walks the rewritten body and prefixes each statement with a `__tb.set_fileline(file, lineno, func)` call. When span tracking is enabled, the builder uses these to stamp emitted TIR nodes with their originating source line — so a later compiler error can point back at the exact line of your kernel.

## `IRGenerator` — the product

The output of `mutate` is a small dataclass:

```python
@dataclass
class IRGenerator(Generic[_P, _T]):
    gen: Callable[[BaseBuilder], Callable[_P, _T]]
    source: str
    extra_type_hints: dict[str, Any] = field(default_factory=dict)
```

- **`gen`** is the compiled `make_closure` result: hand it a builder and it returns the traced function, ready to be called with the kernel's arguments.
- **`source`** is the rewritten body, unparsed back to text. It is the single most useful debugging aid in the frontend — when a kernel does something surprising, printing `ir_gen.source` shows exactly what Python is being executed on your behalf.
- **`extra_type_hints`** carries the parameter types recovered from the annotations.

Crucially, `IRGenerator` is **backend-agnostic**: it does not mention TIR at all. What it produces depends entirely on which builder you pass to `gen`.

## `BaseBuilder` vs `Builder` — swappable semantics

`mutate` targets an abstract interface, `BaseBuilder`. Its default implementation is a **transparent interpreter**: `ctx_if` just yields the condition, `bind` returns the value unchanged, `rval`/`arg`/`eval` are identity-ish. Running `ir_gen.gen(BaseBuilder())(...)` reproduces the *original* function's behavior — the rewrite is semantically a no-op by default. That is what makes the whole approach safe to reason about: the transform adds interception points without changing meaning.

The real work lives in `Builder(BaseBuilder)`, which overrides every hook to emit TVM TIR through `tvm.script.ir_builder`. It keeps a stack of frames and, at each hook, distinguishes compile-time Python values from TIR expressions via `unwrap_cond` / `unwrap_expr`:

```python
def ctx_if(self, cond):
    cond = unwrap_cond(cond)
    if isinstance(cond, PrimExpr):          # dynamic → emit a TIR If frame
        with self.with_frame(tirx.If(cond)):
            yield self._has_if_frame
    else:                                    # static → specialize at trace time
        yield cond
```

This one method is the heart of the "Python-where-you-can, IR-where-you-must" behavior. The same split appears throughout `Builder`: `ctx_for` turns a Python `range`/`T.serial` into a TIR loop frame but folds constant trip counts; `bind` names buffers and vars, materializes lets, and routes mutable `alloc_var` writes through `buffer_store`; `arg` turns an annotated parameter into a `tirx.arg` buffer; `boolop`/`ifexp` build `And`/`Or`/`if_then_else` only when their operands are `PrimExpr`.

## Putting it together: the `prim_func` decorator

The `prim_func` decorator wires the pieces into a pipeline:

```python
def prim_func(func):
    ir_gen = mutate(func)                     # 1. rewrite + compile
    annot  = ...                              # 2. resolve parameter types (uses extra_type_hints)
    builder = Builder()
    with builder.prim_func(func.__name__):    # 3. open a PrimFunc frame
        ir_gen.gen(builder)(**annot)          # 4. TRACE the body into TIR
    return builder.get()                      # 5. extract the finished PrimFunc
```

Step 4 is where tracing happens: calling the generated function *with a real `Builder`* walks the (rewritten) body, and every `__tb.*` call appends to the TIR being built. Step 5 hands back a TVM `PrimFunc`, which then enters TileLang's optimization and codegen pipeline described in the [introduction](tilelang_intro).

### Lazy vs eager, and two-phase JIT

The decorator supports two styles. A **lazy** kernel explicitly builds and returns a `PrimFunc` (the classic `@T.prim_func` inside a factory). An **eager** kernel uses the builder-tracing path above and is wrapped in a `JITFunc` that compiles on first call and caches by argument shape.

Eager mode adds a neat two-phase trick for dynamic shapes. You declare symbolic dimensions with `T.const`:

- **Phase 1** traces the body with those dimensions as fresh symbolic `constexpr` vars, producing a shape-generic *template* plus a matcher that records which buffer shape/stride each symbol came from.
- **Phase 2**, on an actual call, reads the real shapes off the tensor arguments, substitutes them for the symbols, and re-traces to a concrete `PrimFunc`.

Because both phases reuse the *same* `IRGenerator.gen`, there is exactly one description of the kernel; the frontend just runs it with different builders and different symbol bindings. `T.macro` uses the identical machinery — a macro is an `IRGenerator` that gets inlined into the current builder at its call site instead of opening a new `PrimFunc`.

## Why trace instead of parse

Stepping back, the trace-based frontend buys three things a node-by-node parser struggles with:

1. **Free metaprogramming.** Compile-time `if`, unrolled Python loops, list/dict manipulation, closures, and helper functions all work because the body is genuinely executed as Python. The IR only records what touches tensors.
2. **One description, many interpretations.** The same rewritten function runs under a transparent `BaseBuilder` (for semantics and testing), a TIR `Builder` (for codegen), or a phase-1/phase-2 builder (for dynamic shapes). Adding a new interpretation means implementing the ~20 `BaseBuilder` hooks, not writing another AST walker.
3. **Honest error locations.** Because the transform threads `set_fileline` through every statement, IR nodes carry the source line they came from — diagnostics point at your kernel, not at generated code.

The cost is a layer of indirection — a kernel does not *mean* what a casual reader thinks until they remember every statement is really a builder call. When that indirection bites, `IRGenerator.source` is the escape hatch: it shows the exact Python being run, and from there the behavior of `DSLMutator` and `Builder` is fully mechanical.
