"""Adaptive gate adjustment for soft routing.

This module implements Eq. (3.4) and Eq. (3.5) for automatically updating
activation gates based on gradient influence, latency, and cost statistics.
The implementation is intentionally import-free at module scope in order to
conform with repository guidelines.  All heavy imports are performed lazily
inside methods.
"""


class GateAdjuster:
    """Update neuron activation thresholds using gradient-based influence.

    Parameters
    ----------
    reporter : object, optional
        Metric collector providing a ``report`` method compatible with
        :class:`main.Reporter`.
    alpha : object, optional
        Scaling factor for latency in the influence computation.
    beta : object, optional
        Scaling factor for cost in the influence computation.
    learning_rate : object, optional
        Step size applied to the influence when updating the threshold.
    """

    def __init__(self, reporter=None, alpha=1.0, beta=1.0, learning_rate=0.1):
        import torch  # local import

        self._reporter = reporter
        self._alpha = alpha if hasattr(alpha, "shape") else torch.tensor(alpha)
        self._beta = beta if hasattr(beta, "shape") else torch.tensor(beta)
        self._lr = (
            learning_rate
            if hasattr(learning_rate, "shape")
            else torch.tensor(learning_rate)
        )

    def adjust_gate(self, neuron, stats):
        """Adjust ``neuron``'s activation threshold using ``stats``.

        ``stats`` must provide ``gradient``, ``latency``, and ``cost`` entries.
        The influence :math:`I(g)` is computed as defined in Eq. (3.4)::

            I(g) = g' - (alpha * latency + beta * cost)

        Eq. (3.5) then updates the activation threshold ``theta``::

            theta = theta - lr * I(g)

        All operations use tensors to preserve autograd compatibility.
        """

        import torch  # local import

        grad = stats.get("gradient", torch.tensor(0.0))
        latency = stats.get("latency", torch.tensor(0.0))
        cost = stats.get("cost", torch.tensor(0.0))
        if not hasattr(grad, "shape"):
            grad = torch.tensor(grad)
        if not hasattr(latency, "shape"):
            latency = torch.tensor(latency)
        if not hasattr(cost, "shape"):
            cost = torch.tensor(cost)
        influence = grad - (self._alpha * latency + self._beta * cost)
        threshold = getattr(neuron, "activation_threshold", torch.tensor(0.0))
        new_threshold = threshold - self._lr * influence
        neuron.activation_threshold = new_threshold
        if self._reporter is not None:
            self._reporter.report(
                "routing_adjustment",
                "Influence applied to gate thresholds",
                influence,
            )
            self._reporter.report(
                f"neuron_{id(neuron)}_gate_threshold",
                "Updated activation threshold for neuron",
                new_threshold.detach() if hasattr(new_threshold, "detach") else new_threshold,
            )
        return new_threshold
