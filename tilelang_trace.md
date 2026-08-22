---
layout: page
title: TileLang Trace Phase — how running the rewritten DSL emits TIR
---

# TileLang Trace Phase: how running the rewritten DSL emits TIR

The [frontend article](tilelang_frontend) covered the *rewriting* half of TileLang's eager frontend: `mutate` and `DSLMutator` turn your kernel into a function whose every statement is a call on a builder object `__tb`, wrapped up as an `IRGenerator`. That article stopped at the moment the rewritten function is *run*. This note is about that run — the **trace phase** — and the class that drives it: `Builder` in `tilelang/language/eager/builder.py`.

If the rewrite is "make Python interceptable," the trace is "intercept it and write down TIR."

## Where we are

Recall the `prim_func` pipeline:

```python
ir_gen  = mutate(func)                     # rewrite + compile   (frontend article)
builder = Builder()
with builder.prim_func(func.__name__):     # open a PrimFunc frame
    ir_gen.gen(builder)(**annot)           # <-- THE TRACE PHASE
pf = builder.get()                         # extract the finished PrimFunc
```

The single line `ir_gen.gen(builder)(**annot)` *executes* the rewritten kernel with a real `Builder` as `__tb`. Nothing here parses anything — Python runs the *rewritten* body, and each `__tb.*` call mutates the builder's growing TIR. This article is entirely about what happens inside those calls.

## The builder is a stateful tree-writer

`Builder` subclasses `BaseBuilder` (the transparent interpreter) and holds the trace state:

```python
class Builder(BaseBuilder):
    def __init__(self):
        self.frames: list[AnyFrame] = []
        self.ir_builder = IRBuilder()          # tvm.script.ir_builder
        self.name_inside_frame: dict[str, AnyFrame] = {}
        ...
```

Two pieces matter most:

- **`ir_builder`** is TVM's `IRBuilder` (the `tirx` API). It is the actual accumulator — every emitted loop, `if`, buffer store, and let-binding is appended through it, and `ir_builder.get()` returns the finished `PrimFunc` at the end.
- **`frames`** is a stack of *context managers*. Each TIR scope — the `PrimFunc` body, a `for` loop, an `if`'s then-branch — is a frame. Entering a frame opens a new nesting level in the IR; exiting it (`__exit__`) finalizes that subtree and splices it into the parent frame.

The whole trace is a disciplined walk of that frame stack. `with_frame` is the workhorse:

```python
@contextmanager
def with_frame(self, frame):
    pop_idx = len(self.frames)
    yield self.enter_frame(frame)          # push + __enter__
    while len(self.frames) > pop_idx:
        self.frames.pop().__exit__(...)    # pop + __exit__ (finalize into parent)
```

So the *shape* of the Python control flow at trace time becomes the *shape* of the TIR tree. When the rewritten body does `with self.with_frame(tirx.If(cond)): ...`, a TIR `If` node is opened; when that `with` block ends, the node is closed and attached where it belongs. The builder also registers itself in thread-local storage during `prim_func`, so free functions like `T.alloc_shared` or `T.gemm` can find "the builder currently tracing" via `Builder.current()`.

## The central decision: Python value, or IR expression?

Every hook makes the same judgment before doing anything: **is this thing a compile-time Python value, or a piece of TIR?** Two helpers answer it.

`unwrap_expr` normalizes a traced value (unwrapping meta-vars, loading scalar `local.var` buffers, etc.), and `unwrap_cond` reduces a condition to either a Python `bool` or a TIR `PrimExpr`:

```python
def unwrap_cond(expr):
    expr = unwrap_expr(expr)
    if isinstance(expr, (IntImm, FloatImm, StringImm)):
        return bool(expr.value)     # constant-folded → Python bool
    elif isinstance(expr, PrimExpr):
        return expr                 # genuinely dynamic → TIR
    elif isinstance(expr, (int, bool)) or expr is None:
        return bool(expr)
    ...
```

That distinction is the soul of the trace phase. If a value is Python, the builder lets Python handle it and *nothing is emitted*; if it is a `PrimExpr`, the builder emits IR. `ctx_if` is the clearest case:

```python
def ctx_if(self, cond):
    cond = unwrap_cond(cond)
    if isinstance(cond, PrimExpr):
        with self.with_frame(tirx.If(cond)):   # dynamic → emit an If node
            yield self._has_if_frame
    else:
        yield cond                              # static → the branch just runs (or not)
```

A kernel's `if block_M == 128:` on a Python constant is *specialized away* — only the taken branch is traced, and the `if` leaves no mark in the IR. A `if tid < n:` where `tid` is a thread index becomes a real TIR `If`. Same source line, two outcomes, decided by the value's type at trace time.

The same gate recurs everywhere:

- **`ctx_for`** turns `range` / `T.serial` / `T.Pipelined` into a TIR loop frame, but first computes and validates the trip count, folding constant `start/stop/step` (`tirx.ceildiv(stop - start, step)`), and rejects a zero step.
- **`ctx_while`** unwraps its condition; a statically-true condition is an error (*"Infinite while loop detected"*), a statically-false one warns and is skipped, and only a `PrimExpr` becomes a TIR `While`.
- **`boolop` / `ifexp`** build `tirx.And/Or/Not` and `tirx.if_then_else` **only** when the operands are `PrimExpr`; otherwise they fall back to `BaseBuilder`'s plain Python `and`/`or`/ternary.
- **`assert_expr`** emits a TIR `Assert` for a dynamic condition, but for a static one just runs Python's `assert`.

## Binding names: lets, buffers, and mutable vars

Assignment is the richest hook. `a = v` was rewritten to `a = __tb.bind('a', v)`, and `bind` decides what `a` *is* in the IR based on `v`'s type. The cases, in order:

1. **Pure expression at the PrimFunc frame** — returned as-is, no let. This is a deliberate carve-out: shape math like `M2 = M * 2` written before `T.match_buffer` must stay a raw `PrimExpr`, not become a `LetStmt` (which would break buffer matching).
2. **Store into an existing mutable var** — if `a` already names a `local.var` or a `Ref`, and the value is numeric, the builder emits a `buffer_store` instead of a new binding. This is how `T.alloc_var` gives you a genuinely mutable scalar with SSA-free `x = x + 1` semantics.
3. **Trivial Python values** (`int`, `float`, `str`, `tuple`, `list`) — returned unchanged; they live only at trace time.
4. **A `Var` or `Buffer`** — named via `IRBuilder.name(name, value)` so the generated TIR reads with your variable names, and its defining frame is recorded (see scope safety below).
5. **Everything else** — routed to `bind_immutable`, which for a `PrimExpr` creates an SSA let binding:

```python
def bind_immutable(self, name, value):
    ...
    elif isinstance(value, (PrimExpr, BufferRegion)):
        var = tirx.bind(value)          # emit `let name = value`
        register_let_value(var, value)
        IRBuilder.name(name, var)
        return var
```

So a plain `x = a + b` in your kernel becomes an immutable `let`, while `x = T.alloc_var(...)` followed by `x = ...` becomes mutable storage updated by `buffer_store`. Re-binding an immutable name warns you (*"use T.alloc_var to create a mutable variable"*) — a diagnostic that only exists because the tracer knows the difference.

`aug_assign` (`x += v`) mirrors this: it stores through a `Ref` or `local.var`, or, for an SSA `Var`, treats `x += v` as a re-bind `x = x + v` (with the same warning), reusing the pure-expr fast path where it can.

## Stores, reads, and scope safety

**Stores.** `A[i] = v` was rewritten to `__tb.assign_slice(A, i, v)`, which for a TIR `Buffer` emits `tirx.buffer_store(A, v, i)`. `aug_assign_slice` handles `A[i] += v` by loading, applying the op, and storing back.

**Reads with a scope guard.** Every name *load* was rewritten to `__tb.rval('a', a)`. Beyond unwrapping the value, `rval` enforces a rule a naive interpreter cannot:

```python
def rval(self, name, value):
    if name in self.name_inside_frame:
        frame = self.name_inside_frame[name]
        if frame not in self.frames:       # defined in a frame we've already exited
            raise RuntimeError(f"Immutable variable `{name}` is used outside its defining region!")
    return self.unwrap_value(value)
```

Because `bind` recorded which frame each name was defined in (`name_inside_frame`), reading a loop-local outside its loop is caught at trace time with a clear error instead of producing malformed TIR.

**Frame-returning values.** `unwrap_value` also handles the case where a value *is* a frame — e.g. `bx, by = T.Kernel(...)`. `T.Kernel` returns a `KernelLaunchFrame`; unwrapping it *enters* the frame and yields the block indices, which is why the kernel body ends up nested inside the launch scope.

## `eval`: statements that are just expressions

A bare expression statement `f(x)` was rewritten to `__tb.eval(f(x))`. `eval` decides what an unused result means: enter it if it is a frame, `tirx.evaluate` it if it is a `PrimExpr` (so a call with side effects is kept), ignore `None`/buffers, and *warn* if a genuine value is being silently discarded. This is what lets tile primitives like `T.copy(...)` or `T.gemm(...)`, written as standalone statements, land in the IR.

## Assembling the PrimFunc

The `prim_func` context manager brackets the whole trace:

```python
@contextmanager
def prim_func(self, name):
    thread_local_storage.builder = self
    with self.ir_builder, self.with_frame(tirx.prim_func()):
        tirx.func_name(name)
        yield
    ...
    del thread_local_storage.builder
```

Inside, `arg` turns each annotated parameter into a `tirx.arg` buffer or scalar. When the body finishes, `get()` calls `ir_builder.get()` to materialize the `PrimFunc` and stamps any source span. Tensors created with `T.empty` and `return`ed are tracked (`out_idx`) so the compiled kernel knows which arguments are outputs — the builder even checks that every `T.empty` tensor is actually returned.

## Tracing twice: two-phase eager JIT

Dynamic shapes are handled by tracing the *same* `IRGenerator` twice with different builder states.

**Phase 1** traces with symbolic dimensions. `T.const("M, N")` (only legal in eager mode) asks the builder for fresh `constexpr` vars:

```python
def constexpr(self, name, dtype="int32"):
    var = tirx.Var(name, dtype)
    self.constexpr_var.add(var)
    var.orig_name = name
    return var
```

The result is a shape-generic `PrimFunc` plus a **matcher** (`TirTemplate.create`) that records, for each symbol, which buffer's shape or stride it came from — e.g. "`M` is `A.shape[0]`". During phase 1 the builder also sets `eager_jit = "phase1"`, which makes `skip_kernel_ctx()` return `True` so the `T.Kernel` launch scope is skipped — phase 1 only needs the *signature*, not the body's launch structure.

**Phase 2** runs on a real call. `get_tir` reads the actual tensor shapes, resolves each symbol through the matcher, and *re-traces* the body with those concrete values substituted:

```python
def get_tir(self, tensor_args, given_tensor_args, kwargs):
    values = self._parse_phase2_key(**given_tensor_args, **kwargs)
    subs = {name.orig_name: value for name, value in zip(self.matcher, values)}
    builder = Builder()
    builder.eager_jit = "phase2"
    builder.eager_jit_subs = subs
    with builder.prim_func(self.name):
        self.ir_gen.gen(builder)(**tensor_args, **kwargs)   # trace again, concretely
    return builder.get()
```

Because both phases call the identical `ir_gen.gen`, there is exactly one description of the kernel — the frontend just runs it with different builders and symbol bindings. `JITFunc` caches phase-1 templates by compile-time arguments and phase-2 results by shape, so repeated calls with the same shapes skip tracing entirely.

## Macros: tracing inline

`T.macro` reuses the whole machinery. A `Macro` holds an `IRGenerator`, and calling it *inside* a kernel does not open a new `PrimFunc` — it traces the macro body into the **current** builder:

```python
def __call__(self, *args, **kwargs):
    builder = Builder.current()
    with builder.macro(self.name, self.annotations):
        return self.ir_gen.gen(builder)(*args, **kwargs)
```

So a macro is effectively inlined at its call site. `macro_arg` even supports pass-by-reference (`T.Ref`), letting a macro write back into a caller's buffer or var — something a plain function call could not express.

## Source spans

Threaded through all of this is source-location tracking. The rewrite emitted a `__tb.set_fileline(file, line, func)` before every statement; the builder records it and, when spans are enabled, stamps freshly-emitted TIR statements and buffers with that location. The result is that a diagnostic from a *later* compiler pass can point back at the exact line of your kernel, not at generated code — a direct payoff of doing IR generation by tracing live Python.

## Takeaways

The trace phase is a stateful walk that converts a live Python execution into a TIR tree, and almost every decision reduces to one question: **is this value Python or IR?**

- Python values are handled by Python and leave no trace — this is where compile-time specialization, loop unrolling, and shape math come for free.
- `PrimExpr` values are emitted through the `IRBuilder`, with the frame stack mirroring your control flow into nested TIR scopes.
- On top of that, the builder adds things a parser gets no chance to: mutable-vs-SSA binding, out-of-scope-use detection, discarded-value warnings, and honest source spans.

Together with the [frontend rewrite](tilelang_frontend), this is the whole front half of TileLang: rewrite Python so it can be intercepted, then trace it to emit TIR — which then enters the optimization and codegen pipeline from the [introduction](tilelang_intro).
