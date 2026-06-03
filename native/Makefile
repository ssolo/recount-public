# Native CPU backend for recount.
# Builds librecount.{dylib,so} (loaded by recount.native_backend via ctypes).
#
#   make             -> build the shared library (auto-detect platform)
#   make native      -> rebuild with -mcpu=native (this machine only, ~3% gain)
#   make test        -> build + run the standalone C test
#   make clean       -> remove build artifacts
#
# Platforms supported:
#
#   Apple Silicon (Darwin, arm64) — the production default.
#     CFLAGS  : -O3 -mcpu=apple-mN -ffast-math -fno-finite-math-only -funroll-loops
#                              (N auto-detected from sysctl machdep.cpu.brand_string;
#                               falls back to apple-m1 baseline if detection fails)
#     SIMD    : Accelerate vForce (vvexp, vDSP) in the inner exp/logsumexp loops
#     Threads : libdispatch (Grand Central Dispatch) via dispatch_apply
#     LIBS    : -framework Accelerate
#
#   Intel Mac (Darwin, x86_64) — same compiler features, but -mcpu=apple-mN
#   is replaced by -march=native; Accelerate + libdispatch still available.
#
#   Linux (any arch, x86_64 / aarch64) — falls back to plain libm + OpenMP.
#     CFLAGS  : -O3 -march=native -ffast-math -fno-finite-math-only -funroll-loops -fopenmp
#     SIMD    : compiler auto-vectorization on the scalar path (no Accelerate)
#     Threads : OpenMP via #pragma omp parallel for
#     LIBS    : -lm -fopenmp
#
# The portability layer is a single header (include/recount_parallel.h) that
# defines RECOUNT_PARALLEL_APPLY(n, BODY) — libdispatch closure on Apple,
# OpenMP parallel-for on Linux. Numerical results are identical on both paths.

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

CC      ?= clang

# ---- Per-platform compile + link flags --------------------------------------
ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    # Apple Silicon — auto-detect chip family for matching -mcpu flag.
    # apple clang 17 supports apple-m1/m2/m3/m4; older clang falls back to
    # apple-m1 baseline (still produces valid arm64 code).
    APPLE_BRAND := $(shell sysctl -n machdep.cpu.brand_string 2>/dev/null)
    ifneq (,$(findstring M4,$(APPLE_BRAND)))
      ARCH_CFLAGS := -mcpu=apple-m4
    else ifneq (,$(findstring M3,$(APPLE_BRAND)))
      ARCH_CFLAGS := -mcpu=apple-m3
    else ifneq (,$(findstring M2,$(APPLE_BRAND)))
      ARCH_CFLAGS := -mcpu=apple-m2
    else
      # M1 baseline (also covers unknown / future chips if auto-detect fails).
      ARCH_CFLAGS := -mcpu=apple-m1
    endif
  else
    # Intel Mac (x86_64).
    ARCH_CFLAGS := -march=native
  endif
  PLATFORM_LIBS := -framework Accelerate
  DYLIB := librecount.dylib
  LDFLAGS_PLATFORM := -shared
else
  # Linux (or other Unix).
  ARCH_CFLAGS := -march=native -fopenmp
  PLATFORM_LIBS := -lm -fopenmp
  DYLIB := librecount.so
  LDFLAGS_PLATFORM := -shared -Wl,-soname,$(DYLIB)
endif

# Aggressive optimisation flags (verified on Apple clang 17):
#   -O3                 — peephole + scheduling + vectorise
#   -mcpu / -march      — chip-specific instr selection (above)
#   -ffast-math + -fno-finite-math-only — relax FP ordering, keep NaN/Inf
#   -funroll-loops      — unroll small inner loops
#   -fvectorize         — explicit (already on by default at -O3)
#   -flto               — link-time optimisation, cross-file inlining
#   -fno-plt            — avoid PLT trampoline on indirect calls
CFLAGS  ?= -O3 $(ARCH_CFLAGS) -ffast-math -fno-finite-math-only \
           -funroll-loops -fvectorize -flto -fno-plt -fPIC \
           -Wall -Wextra -Wno-unused-parameter \
           -Iinclude
LDFLAGS ?= $(LDFLAGS_PLATFORM) -flto
LIBS    ?= $(PLATFORM_LIBS)

SRCS = src/recount_math.c \
       src/recount_factln.c \
       src/recount_tree.c \
       src/recount_rates.c \
       src/recount_survival.c \
       src/recount_edge.c \
       src/recount_combine.c \
       src/recount_forward.c \
       src/recount_outside.c \
       src/recount_gradient.c \
       src/recount_events.c \
       src/recount_unobserved.c \
       src/recount_batch.c \
       src/recount_bfgs.c \
       src/recount_unobserved_grad.c \
       src/recount_brownian_prior.c \
       src/recount_version.c

OBJS = $(SRCS:.c=.o)

all: $(DYLIB)
	@echo "Built $(DYLIB) for $(UNAME_S) / $(UNAME_M)"

$(DYLIB): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)

%.o: %.c $(wildcard include/*.h)
	$(CC) $(CFLAGS) -c $< -o $@

test: $(DYLIB) test/test_4leaf
	./test/test_4leaf

test/test_4leaf: test/test_4leaf.c $(DYLIB)
	$(CC) -O2 -Iinclude -o $@ $< -L. -lrecount -Wl,-rpath,@executable_path/..

clean:
	rm -f $(OBJS) librecount.dylib librecount.so test/test_4leaf

# Opt-in: rebuild with -mcpu=native for this machine's specific ISA.
# On Apple Silicon: ~3% more than apple-m1 baseline on M4 Max.
# NOT portable across machines.
native:
	$(MAKE) clean
	$(MAKE) ARCH_CFLAGS="-mcpu=native"

.PHONY: all test clean native
