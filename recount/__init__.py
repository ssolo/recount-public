"""recount: phylogenetic gain-loss-duplication likelihood and gradient.

A pure-Python re-implementation of the inside / outside algorithm in M. Csűrös'
Count Java package, exposing both:

  * NumPy backend  — analytical gradient via the inside-outside posterior
                     expectations of Csűrös (2021), arXiv:2107.11440.
  * PyTorch backend — same forward pass written with torch tensors so
                      `torch.autograd` recovers the gradient via reverse-mode AD.

Verified to match the original Java to ~1e-13 (forward log-likelihood) and
~5e-13 (gradient) on the canonical 4-leaf test case.

Public API (NumPy):

    >>> from recount import Tree, GLDRates, log_likelihood, gradient_survival
    >>> tree = Tree(parent=[4, 4, 5, 5, 6, 6, -1])      # leaves first
    >>> rates = GLDRates(tree, gain=..., loss=..., dup=..., length=...)
    >>> ll = corrected_log_likelihood(tree, rates, profiles, min_copies=1)
    >>> g  = gradient_survival(tree, rates, profiles, min_copies=1)
    # g is a flat array indexed 3*v + {GAIN=0, LOSS=1, DUP=2}, in the survival
    # parameterization (p̃, q̃, r̃/κ̃) — same orientation as Java's
    # Gradient.getCorrectedGradient().

PyTorch backend:

    >>> from recount.torch_backend import corrected_log_likelihood_t, gradient_autograd
    >>> grads = gradient_autograd(tree, gain, loss, dup, length, profiles)
    # grads is a dict {'gain', 'loss', 'dup', 'length'} of per-node ∂LL/∂rate.

Reference: Csűrös M (2021). "Gain-loss-duplication models on a phylogeny:
exact algorithms for computing the likelihood and its gradient."
arXiv:2107.11440.
"""

from recount.tree import Tree
from recount.rates import GLDRates, rate_to_p, rate_to_q
from recount.gld import (
    SurvivalParams,
    compute_survival_params,
    log_likelihood,
    corrected_log_likelihood,
    empty_log_likelihood,
    singleton_log_likelihood,
    gradient_survival,
    GAIN,
    LOSS,
    DUP,
)

__all__ = [
    "Tree",
    "GLDRates",
    "SurvivalParams",
    "compute_survival_params",
    "log_likelihood",
    "corrected_log_likelihood",
    "empty_log_likelihood",
    "singleton_log_likelihood",
    "gradient_survival",
    "rate_to_p",
    "rate_to_q",
    "GAIN",
    "LOSS",
    "DUP",
    # Rate variation
    "LogisticShift",
    "mixture_log_likelihood",
    "corrected_mixture_log_likelihood",
]


def __getattr__(name: str):
    # Lazy-import rate_variation so a bare ``import recount`` doesn't drag
    # torch in for users who don't need the mixture path.
    if name in {"LogisticShift", "mixture_log_likelihood",
                "corrected_mixture_log_likelihood"}:
        from recount import rate_variation
        return getattr(rate_variation, name)
    raise AttributeError(name)

__version__ = "0.1.0"
