---
layout: page
title: Writing a GCC Plugin
---

# Writing a GCC Plugin

Customizing the compiler and programming language is an important part of bringing up a new AI chip, and writing a GCC plugin is one way to do that customization.

## Why plugins

GCC is a production compiler with decades of optimizations, but its pipeline is a closed box: source goes in, an object file comes out. Sometimes you want to reach *inside* that box — to add a custom warning, enforce a coding rule, instrument every function, harden generated code, or just understand what the mid-end sees. You have three options:

- **Patch GCC** — fork the tree, add your code, rebuild the whole compiler. Maximal power, but you own a fork forever and a full build takes an hour.
- **Post-process** — parse GCC's textual dumps (`-fdump-tree-all`) after the fact. Easy, but read-only and brittle; you cannot change codegen.
- **Write a plugin** — a shared library GCC loads at startup that hooks into the real compilation, sees the actual internal representation, and can add or modify passes. No fork, no rebuild of GCC, and you run against the same IR the optimizer uses.

Plugins are the middle ground: you get first-class access to GCC's internals without maintaining a fork. This is how the Linux kernel ships hardening passes (`structleak`, `randstruct`, `latent_entropy`) and how tools enforce project-specific static analysis.

Plugin support has been in GCC since 4.5. This article targets the modern C++ pass API (GCC 4.9+; examples tested against GCC 10–14).

## The compilation pipeline, and where you hook in

GCC lowers your code through three intermediate representations. A plugin can observe or transform each one:

| IR | Level | What it looks like | Typical plugin work |
|----|-------|--------------------|---------------------|
| **GENERIC** | front-end | language-level AST (`tree` nodes) | read declarations, attributes, types |
| **GIMPLE** | mid-end | three-address SSA form, language-independent | analysis, instrumentation, custom optimizations |
| **RTL** | back-end | register-transfer list, near-machine | target-specific tweaks (rare in plugins) |

Most plugins live at the **GIMPLE** level: it is language-independent (the same pass works for C, C++, and Fortran), it is in SSA form (so def-use chains are explicit), and it is high enough to reason about the program yet low enough that control flow is already an explicit CFG of basic blocks.

### What GIMPLE (GNU simple IR) looks like

GIMPLE is GCC's mid-level IR, and the one you will spend the most time in. Three properties define it:

- **Three-address code.** Every statement does one thing. Nested expressions from the source are broken apart into elementary operations, with compiler-generated temporaries (`_1`, `_2`, …) holding the intermediate values. So `a = b*c + d` becomes a multiply into a temporary followed by an add.
- **Static Single Assignment (SSA).** Each variable is assigned exactly once; a reassignment in the source produces a fresh *version* (`x_3`, `x_5`). Where control flow merges, a `PHI` node picks the right version. This makes def-use chains explicit — given a value, you know precisely where it came from — which is what most analyses need.
- **An explicit CFG.** The function is already a graph of *basic blocks* (straight-line statement runs) connected by edges. Loops and `if`s from the source are gone; only conditional jumps between blocks remain. A pass walks this graph directly (`FOR_EACH_BB_FN`), rather than re-parsing control structures.

Concretely, this C function:

```c
int sum_to(int n) {
    int s = 0;
    for (int i = 0; i < n; i++)
        s += i;
    return s;
}
```

lowers to roughly this GIMPLE (as shown by `gcc -fdump-tree-ssa`):

```
sum_to (int n)
{
  int s, i;

  <bb 2>:
  s_1 = 0;
  i_2 = 0;
  goto <bb 4>;

  <bb 3>:                       // loop body
  s_3 = s_5 + i_6;
  i_4 = i_6 + 1;

  <bb 4>:                       // loop header
  s_5 = PHI <s_1(2), s_3(3)>;   // s is 0 on entry, else the updated value
  i_6 = PHI <i_2(2), i_4(3)>;
  if (i_6 < n_7(D))
    goto <bb 3>;
  else
    goto <bb 5>;

  <bb 5>:
  return s_5;
}
```

Notice what the source structure became: the `for` loop is now blocks 3 and 4 with explicit `goto`s; each assignment produced a numbered SSA version; and the two ways to reach the loop header (first entry vs. the back-edge) are reconciled by the `PHI` nodes. The `(D)` on `n_7` marks a value defined on entry (a parameter). This block-and-version form is exactly what your `execute(function *fun)` iterates over — you inspect statements with accessors like `gimple_assign_rhs1`, and follow SSA edges to trace where a value is used or defined.

A plugin interacts with GCC in two ways:

1. **Callbacks on events** — GCC fires named events as it runs (start of a translation unit, an attribute being registered, end of compilation…). You register a function for the events you care about.
2. **New passes** — you insert your own pass into the pass manager, positioned relative to an existing pass, and GCC runs it like any built-in optimization.

## The anatomy of a plugin

Every plugin is a shared object that GCC `dlopen`s. It must satisfy two contracts:

```c
#include "gcc-plugin.h"      /* must come first */
#include "plugin-version.h"

/* 1. Assert GPL compatibility — GCC refuses to load the plugin without this. */
int plugin_is_GPL_compatible;

/* 2. The entry point GCC calls right after loading. */
int plugin_init(struct plugin_name_args *plugin_info,
                struct plugin_gcc_version *version)
{
    /* Refuse to run against a GCC whose internals differ from what we built for. */
    if (!plugin_default_version_check(version, &gcc_version))
        return 1;                      /* non-zero aborts the compile */

    /* register callbacks / passes here … */
    return 0;
}
```

Two things are doing real work:

- `plugin_is_GPL_compatible` is a *symbol*, not a function. Its mere presence is GCC's license check — omit it and the compiler exits with an error. This is deliberate: GCC's plugin ABI exposes its GPL internals, so plugins must be GPL-compatible.
- `plugin_default_version_check` compares the GCC that is loading you against the `gcc_version` you compiled against. **GCC has no stable plugin ABI** — the internal data structures change between releases — so a plugin is tied to a specific major version. This check turns a silent crash into a clean error.

`plugin_info` carries the plugin's name (`base_name`, used to register callbacks) and its command-line arguments; `version` is the loading compiler's version stamp.

## A minimal GIMPLE pass

Here is a complete plugin that adds a GIMPLE pass counting the statements in every function it compiles — the "hello world" of GCC internals. It shows the three pieces you always write: the `pass_data` descriptor, the pass class, and the registration.

```c
#include "gcc-plugin.h"
#include "plugin-version.h"
#include "tree.h"
#include "tree-pass.h"
#include "context.h"          /* the global `gcc::context *g` */
#include "function.h"
#include "basic-block.h"
#include "gimple.h"
#include "gimple-iterator.h"

int plugin_is_GPL_compatible;

/* Static description of the pass: type, name, timing/property flags. */
const pass_data count_stmts_data = {
    GIMPLE_PASS,          /* type — operates on GIMPLE            */
    "count_stmts",        /* name — shows up in -fdump-passes     */
    OPTGROUP_NONE,        /* optinfo flags                        */
    TV_NONE,              /* timevar id (for -ftime-report)       */
    PROP_gimple_any,      /* properties required as input         */
    0, 0, 0, 0,           /* provided / destroyed / start / finish */
};

/* A pass is a C++ class deriving from the IR-appropriate base. */
struct count_stmts_pass : gimple_opt_pass {
    count_stmts_pass(gcc::context *ctxt)
        : gimple_opt_pass(count_stmts_data, ctxt) {}

    /* Called once per function GCC compiles. */
    unsigned int execute(function *fun) override {
        int n = 0;
        basic_block bb;
        FOR_EACH_BB_FN(bb, fun) {                 /* walk the CFG        */
            for (gimple_stmt_iterator gsi = gsi_start_bb(bb);
                 !gsi_end_p(gsi); gsi_next(&gsi)) /* walk statements     */
                n++;
        }
        fprintf(stderr, "%s: %d GIMPLE statements\n",
                function_name(fun), n);
        return 0;             /* return TODO flags; 0 = nothing changed */
    }
};

int plugin_init(struct plugin_name_args *plugin_info,
                struct plugin_gcc_version *version)
{
    if (!plugin_default_version_check(version, &gcc_version))
        return 1;

    /* Insert our pass right after SSA construction, so we see SSA GIMPLE. */
    struct register_pass_info pass_info;
    pass_info.pass = new count_stmts_pass(g);
    pass_info.reference_pass_name = "ssa";
    pass_info.ref_pass_instance_number = 1;
    pass_info.pos_op = PASS_POS_INSERT_AFTER;

    register_callback(plugin_info->base_name,
                      PLUGIN_PASS_MANAGER_SETUP, NULL, &pass_info);
    return 0;
}
```

The pieces:

- **`pass_data`** is a static descriptor — what kind of pass, its name (which is also its handle in `-fdump-passes` and `-fdump-tree-count_stmts`), and property flags telling the pass manager what IR shape it needs and leaves behind.
- **The pass class** overrides `execute(function *)`, called once per function. Its return value is a bitmask of *TODO* flags (`TODO_update_ssa`, `TODO_cleanup_cfg`, …) telling GCC what you invalidated; returning `0` means "I only looked."
- **Registration** builds a `register_pass_info` pinning the pass into the pipeline: `reference_pass_name` + `ref_pass_instance_number` name an anchor pass, and `pos_op` (`PASS_POS_INSERT_AFTER` / `_BEFORE` / `_REPLACE`) says where you sit relative to it. Passes can run more than once, hence the instance number.

`g` is the global `gcc::context` GCC hands to plugins — you pass it to every pass you construct.

## Building and running

A plugin compiles against GCC's plugin headers, which live in a directory GCC can tell you:

```sh
PLUGIN_DIR=$(gcc -print-file-name=plugin)

g++ -I"$PLUGIN_DIR/include" -fPIC -fno-rtti -shared \
    count_stmts.cc -o count_stmts.so
```

`-fno-rtti` matters: GCC itself is built without RTTI, so your plugin must match or linking the pass classes fails. Then load it on any compile:

```sh
gcc -fplugin=./count_stmts.so -O2 -c program.c
# program.c: for each function → "<name>: <n> GIMPLE statements"
```

You can pass arguments through to the plugin, keyed by plugin name:

```sh
gcc -fplugin=./count_stmts.so -fplugin-arg-count_stmts-threshold=50 -c program.c
```

Inside `plugin_init` these arrive as `plugin_info->argv`, an array of `{ char *key; char *value; }` of length `plugin_info->argc` — so the flag above shows up as `key="threshold", value="50"`.

## Beyond passes: the event system

Passes are for transforming IR. For everything else, GCC fires **events** you subscribe to with `register_callback(name, event, handler, user_data)`. The useful ones:

| Event | Fires when | Use it to |
|-------|-----------|-----------|
| `PLUGIN_PASS_MANAGER_SETUP` | at startup | insert a pass (as above) |
| `PLUGIN_ATTRIBUTES` | attributes are registered | define a custom `__attribute__` |
| `PLUGIN_PRAGMAS` | pragmas are registered | define a custom `#pragma` |
| `PLUGIN_START_UNIT` / `PLUGIN_FINISH_UNIT` | a translation unit begins / ends | per-file setup and reporting |
| `PLUGIN_PRE_GENERICIZE` | a function's GENERIC AST is ready | inspect the front-end AST |
| `PLUGIN_FINISH_TYPE` / `PLUGIN_FINISH_DECL` | a type / decl is parsed | record or check declarations |
| `PLUGIN_FINISH` | GCC is shutting down | flush results, free state |

For example, registering a custom attribute you can then read in a pass:

```c
static void register_attrs(void *event_data, void *user_data) {
    static const struct attribute_spec my_attr = {
        "my_check",  /* usage: __attribute__((my_check)) */
        0, 0,        /* min/max args */
        false, false, false, false,
        NULL, NULL
    };
    register_attribute(&my_attr);
}
/* in plugin_init: */
register_callback(plugin_info->base_name,
                  PLUGIN_ATTRIBUTES, register_attrs, NULL);
```

Now user code can annotate functions with `__attribute__((my_check))`, and your GIMPLE pass can look up `lookup_attribute("my_check", DECL_ATTRIBUTES(current_function_decl))` to decide whether to act — the standard pattern for opt-in analysis.

## What plugins are good for

- **Custom static analysis and warnings** — walk GIMPLE to flag project-specific mistakes (missing error checks, forbidden API calls, tainted-data flows) that no built-in `-W` catches.
- **Instrumentation** — inject calls at function entry/exit or around memory accesses for profiling, coverage, or sanitizer-style checks. This is IR-level instrumentation, so it survives optimization better than source rewriting.
- **Security hardening** — the Linux kernel's GCC plugins zero stack variables, randomize struct layout, and gather entropy, all as GIMPLE/RTL passes.
- **Coding-standard enforcement** — fail the build when code violates a rule (e.g. no floating point in an interrupt handler), using custom attributes to mark intent.
- **Understanding the compiler** — dumping IR at a chosen point is the fastest way to learn what GCC actually does to your code.

## Honest caveats

- **No stable ABI.** A plugin is bound to a GCC major version; internal structures shift between releases, so expect to `#ifdef` on `GCCPLUGIN_VERSION` or maintain per-version builds. This is the single biggest maintenance cost.
- **You are writing against GCC internals.** The headers are sparsely documented; the real reference is GCC's own source and existing plugins. Expect to read `tree-pass.h`, `gimple.h`, and the passes in `gcc/`.
- **GPL only.** The `plugin_is_GPL_compatible` gate is enforced, by design.
- **GIMPLE is a lot.** The `tree`/`gimple` type zoo (predicates like `gimple_code`, accessors like `gimple_assign_rhs1`, the SSA machinery) has a real learning curve. Start by *reading* IR in a pass before you try to *rewrite* it.
- **LLVM is the alternative.** If you are not tied to GCC, LLVM's pass and Clang tooling APIs are better documented and more stable — but if your toolchain is GCC (embedded targets, the kernel, GCC-only language features), a plugin is the way in.

For a first project, write a read-only GIMPLE pass like the one above, get it printing what you expect, then graduate to modifying the IR once the type system stops surprising you.
