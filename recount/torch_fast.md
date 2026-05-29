# Fast PyTorch backend (`recount.torch_fast`)

A heavily vectorized rewrite of the GLD likelihood for GPU. Same model and
conventions as `recount.gld`, but the forward pass is purely functional
torch ops batched across families, so:

- **Autograd works end-to-end** — `LL.backward()` gives the analytical
  gradient w.r.t. (gain, loss, dup, length).
- **All families process in one batch** — `[F, N, W+1]` tensors with the
  inner loops over copy counts expressed as outer products + `logsumexp`.
- **Designed for A100s** — single-GPU and multi-GPU drivers in
  `validation/`.

## Key algorithmic change vs. `recount.gld`

The destructive log-space in-place update inside `_compute_sibling` is
replaced with a **closed-form multinomial split** that requires no
sequential mutations and so can be vectorized.

For a binary parent of children j₁, j₂ with survival loss params p̃₁, p̃₂
and ε = p̃₁·p̃₂:

```
π₁    = (1 - p̃₁) · p̃₂ / (1 - ε)      # parent copy in j₁ only
π₂    = p̃₁ · (1 - p̃₂) / (1 - ε)      # in j₂ only
π_both = (1 - p̃₁) · (1 - p̃₂) / (1 - ε)# in both
```

For each (ell, a, b) with a, b ∈ [0, W] and a + b ≥ ell ≥ max(a, b):

```
log_split[ell, a, b] = log Mult(ell; ell-b, a+b-ell, ell-a)
                       + (ell-b)·log π₁ + (a+b-ell)·log π_both + (ell-a)·log π₂
```

(invalid entries are −∞). Then

```
C[v][f, ell] = LSE_{a,b} (K[j₁][f, a] + K[j₂][f, b] + log_split[v, ell, a, b])
```

which is a single 4-D add followed by a 2-D `logsumexp` — fully
vectorized over the family batch dim and over the (a, b) grid.

The edge (gain) step is similarly vectorized: gather `C[v][s+t]`, add the
precomputed `log_pmf[v, s, t]`, `logsumexp` over `t`.

## Numerical accuracy

For "typical" parameter ranges (everything bounded comfortably away from
the model boundaries `q→0`, `q→1`, `p→1`) the fast backend matches
`recount.gld` to machine precision on the simple 4-leaf test.

On Williams2017 (60-leaf tree with several internal nodes near the
boundary — `q ≈ 10⁻⁹`, `κ ≈ 10⁹`) the per-family log-likelihood agrees
with the analytical NumPy backend to roughly **1e-4** in the worst
families and ~1e-8 in the median. The cumulative LL on 5378 families
differs by ~1.4 nats out of 125269 (rel. 1.1 × 10⁻⁵), comparable to the
NumPy↔Java agreement on the same dataset. This is acceptable for
gradient-based optimization (the gradient direction and magnitude are
preserved well within ~1%); use the analytical NumPy backend if you need
bit-exact agreement with the Java reference.

## Heavy-tailed profile sums

Williams2017 has profile sums ranging from 4 to 668 with median 6. A
uniform `W = max_sum` would waste ≈10000× memory on the median family.
`recount.torch_fast_bucket.corrected_log_likelihood_bucketed` solves this
by sorting families into buckets of similar size and running the forward
pass at the right `W` per bucket. Default bucket edges (right-inclusive):
`[4, 8, 16, 32, 64, 128, 256, 512, 1024]`.

For Williams2017 on CPU this brings the runtime down to ~2 seconds.

## Single-GPU usage

`recount.torch_fast` runs end-to-end on a single CUDA / MPS device.
Build the per-edge tensors on-device and call
`corrected_log_likelihood_t`; autograd traces straight through. The
multi-GPU SLURM / `torchrun` drivers that previously lived in
`validation/` were retired when the native C backend took over the
production path — the torch backend is now kept primarily as a
reference and an autograd-validated cross-check for the analytical C
gradient. For arc269-scale workloads the native C backend
(`recount.native_backend`, 20–80× faster than Java, ~150× faster than
the NumPy reference) is the production target; the torch fast path
remains useful for forward-only diagnostics on GPU.
