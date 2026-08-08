"""Type stubs for the PaCal probabilistic calculus library.

PaCal (Probabilistic Calculus) provides stochastic arithmetic for modelling
uncertainty. It overloads Python operators (|, +, -, *, /) on Distribution
objects to represent probabilistic operations and conditional distributions.
"""

from typing import Any, overload

# ---------------------------------------------------------------------------
# Core distribution class
# ---------------------------------------------------------------------------

class Distribution:
    """Base class for all PaCal distributions."""

    mean: float
    std: float
    var: float

    # PaCal's conditional operator: ``distr | Gt(t)`` means "distr given > t".
    # The right-hand side is always a comparison condition object, never a callable.
    def __or__(self, other: Any) -> Any: ...

    # Arithmetic — used extensively in ws3/common.py
    def __add__(self, other: Any) -> Any: ...
    def __radd__(self, other: Any) -> Any: ...
    def __sub__(self, other: Any) -> Any: ...
    def __rsub__(self, other: Any) -> Any: ...
    def __mul__(self, other: Any) -> Any: ...
    def __rmul__(self, other: Any) -> Any: ...
    def __truediv__(self, other: Any) -> Any: ...
    def __rtruediv__(self, other: Any) -> Any: ...
    def __neg__(self) -> "Distribution": ...
    def __pow__(self, other: Any) -> Any: ...

    # Sampling — used in sylv_cred_rv Monte Carlo loop: ``P.rand(n)``
    def rand(self, n: int = 1) -> Any: ...

    # PDF/CDF evaluation
    def __call__(self, x: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Normal distribution — the main distribution used in ws3/common.py
# ---------------------------------------------------------------------------

class NormalDistr(Distribution):
    @overload
    def __init__(self, mu: float, sigma: float) -> None: ...
    @overload
    def __init__(self, mu: float, sigma: float, minval: float, maxval: float) -> None: ...


# ---------------------------------------------------------------------------
# Bounded / conditional distributions
# ---------------------------------------------------------------------------

class BoundedDistr(Distribution):
    def __init__(
        self,
        distribution: Distribution,
        minval: float,
        maxval: float,
    ) -> None: ...


# ---------------------------------------------------------------------------
# Conditional operators — the ``|`` operator target in PaCal
# Usage: ``NormalDistr(mu, sigma) | Gt(threshold)`` → conditioned distribution
# ---------------------------------------------------------------------------

class Gt:
    """``x | Gt(t)`` — condition x > t."""

    def __init__(self, threshold: float) -> None: ...
    def __invert__(self) -> "Lt": ...


class Ge:
    """``x | Ge(t)`` — condition x >= t."""

    def __init__(self, threshold: float) -> None: ...


class Lt:
    """``x | Lt(t)`` — condition x < t."""

    def __init__(self, threshold: float) -> None: ...


class Le:
    """``x | Le(t)`` — condition x <= t."""

    def __init__(self, threshold: float) -> None: ...


class Eq:
    """``x | Eq(t)`` — condition x == t."""

    def __init__(self, value: float) -> None: ...


# ---------------------------------------------------------------------------
# Other distributions used in ws3/common.py or referenced in docstrings
# ---------------------------------------------------------------------------

class UniformDistr(Distribution):
    def __init__(self, a: float, b: float) -> None: ...


class ExponentialDistr(Distribution):
    def __init__(self, lam: float) -> None: ...


class GammaDistr(Distribution):
    def __init__(self, alpha: float, beta: float) -> None: ...


class BetaDistr(Distribution):
    def __init__(self, alpha: float, beta: float) -> None: ...


class BernoulliDistr(Distribution):
    def __init__(self, p: float) -> None: ...


class DiscreteDistr(Distribution):
    def __init__(self, values: Any, probs: Any) -> None: ...


# ---------------------------------------------------------------------------
# Mathematical functions — pacal.log, pacal.exp, etc.
# These are module-level functions that accept Distribution | float and return
# Distribution (for Distribution input) or float (for float input).
# Using Any return type avoids arg-type errors when the result is later used
# in arithmetic with other distributions.
# ---------------------------------------------------------------------------

def log(x: Any) -> Any: ...
def exp(x: Any) -> Any: ...
def sqrt(x: Any) -> Any: ...
def pow(x: Any, y: float) -> Any: ...
def abs(x: Any) -> Any: ...
def max(x: Any, y: Any) -> Any: ...
def min(x: Any, y: Any) -> Any: ...
def cos(x: Any) -> Any: ...
def sin(x: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Numerical integration helpers
# ---------------------------------------------------------------------------

def eval_in_eq(f: Any, distribution: Distribution, n: int = ..., **kwargs: Any) -> float: ...
