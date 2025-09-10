"""Path cost calculation utilities for neural graphs.

This module introduces the :class:`PathCostCalculator` which evaluates the
aggregate loss and latency along a given path and combines them into a single
cost term according to Eq. (1.2).  The implementation avoids module level
imports and records all computed metrics through an optional reporter object.
"""


class PathCostCalculator:
    """Compute cost metrics for paths through a neural graph.

    Parameters
    ----------
    reporter : object, optional
        Object providing a ``report`` method compatible with
        :class:`main.Reporter`.  If supplied, all cost components are reported
        through it.
    zero : object, optional
        Zero-like tensor used for initialisation and type promotion.  Defaults
        to ``0``.
    """

    def __init__(self, reporter=None, zero=0):
        self._reporter = reporter
        self._zero = zero
        self._lambda_0 = zero
        self._lambda_max = zero

    def log_anneal(self, x, T_heat):
        """Return the logarithmically annealed value of ``x``.

        The input ``x`` is interpreted as a latency term.  It is first
        normalised using the ``lambda_0`` and ``lambda_max`` values provided to
        :meth:`compute_cost` and then transformed according to a logarithmic
        annealing schedule controlled by ``T_heat``.
        """
        lambda_0 = getattr(self, "_lambda_0", self._zero)
        lambda_max = getattr(self, "_lambda_max", self._zero)
        if hasattr(x, "__sub__"):
            delta = x - lambda_0
        else:
            delta = x - lambda_0
        if hasattr(delta, "clamp"):
            delta = delta.clamp(min=0)
            denom = (lambda_max - lambda_0) + delta * 0 + 1
            norm = delta / denom
            t_heat = norm * 0 + T_heat
            return (norm * t_heat).log1p() / t_heat.log1p()
        else:
            from math import log1p
            delta = max(delta, 0)
            denom = (lambda_max - lambda_0) + 1
            norm = delta / denom
            return log1p(norm * T_heat) / log1p(T_heat)

    def compute_cost(self, path, lambda_0, lambda_max, alpha, beta, T_heat):
        r"""Compute the total cost of ``path``.

        ``path`` is an iterable of neurons and synapses.  For neurons, the
        attributes ``last_local_loss`` and ``lambda_v`` are used; for synapses
        ``c_e`` and ``lambda_e`` are considered.  The aggregated loss ``L_s`` and
        latency ``\lambda_s`` are combined into ``c_s`` following Eq. (1.2)::

            c_s = \alpha L_s + \beta \cdot A(\lambda_s)

        where ``A`` denotes the logarithmic annealing function implemented by
        :meth:`log_anneal`.
        """
        loss = self._zero
        latency = self._zero
        for element in path:
            if hasattr(element, "last_local_loss"):
                loss = loss + getattr(element, "last_local_loss")
                latency = latency + getattr(element, "lambda_v", self._zero)
            elif hasattr(element, "c_e"):
                loss = loss + getattr(element, "c_e")
                latency = latency + getattr(element, "lambda_e", self._zero)
        zero_like = latency * 0 if hasattr(latency, "__mul__") else 0
        self._lambda_0 = zero_like + lambda_0
        self._lambda_max = zero_like + lambda_max
        alpha_t = zero_like + alpha
        beta_t = zero_like + beta
        annealed = self.log_anneal(latency, T_heat)
        cost = alpha_t * loss + beta_t * annealed
        if self._reporter is not None:
            self._reporter.report("path_loss", "Aggregate loss along path", loss)
            self._reporter.report("path_latency", "Aggregate latency along path", latency)
            self._reporter.report("path_cost", "Total cost of path", cost)
        return cost
