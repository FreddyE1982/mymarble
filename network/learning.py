"""Utilities for tracking learning metrics in neural graphs.

The :class:`LossTracker` class maintains per-neuron loss statistics without
introducing any module level imports.  A reporter object is expected to be
provided so that metrics are recorded globally.
"""


class LossTracker:
    """Track rolling loss metrics for neurons.

    Parameters
    ----------
    reporter : object
        Instance providing a ``report`` method compatible with
        :class:`main.Reporter`.  Metrics are emitted through this object.
    zero : object, optional
        Tensor-like object used for initialising averages.  Defaults to ``0``.
    """

    def __init__(self, reporter, zero=0):
        self._reporter = reporter
        self._zero = zero
        self._stats = {}

    def update_loss(self, neuron, new_path_losses):
        """Update loss history for ``neuron``.

        This implements the recurrence from Eq. (0.1) by maintaining both the
        cumulative path count :math:`m_{v\to}^t` and a rolling average of the
        provided path losses over time steps.  The neuron's
        ``last_local_loss`` field is updated with the new rolling average and
        the update is recorded via :func:`Reporter.report`.

        Parameters
        ----------
        neuron : :class:`network.entities.Neuron`
            The neuron whose loss statistics are updated.
        new_path_losses : iterable
            Iterable of loss tensors representing the loss along each outgoing
            path for the current time step.
        """
        losses = list(new_path_losses)
        path_count = len(losses)
        total_loss = self._zero
        for loss in losses:
            total_loss = total_loss + loss
        if path_count:
            mean_loss = total_loss / path_count
        else:
            mean_loss = self._zero
        stats = self._stats.get(neuron)
        if stats is None:
            stats = {"t": 0, "avg": self._zero, "m": 0}
        stats["t"] += 1
        stats["m"] += path_count
        t = stats["t"]
        avg = ((stats["avg"] * (t - 1)) + mean_loss) / t
        avg_detached = avg.detach() if hasattr(avg, "detach") else avg
        stats["avg"] = avg_detached
        self._stats[neuron] = stats
        neuron.record_local_loss(avg_detached)
        count = self._reporter.report("loss_updates") or 0
        self._reporter.report(
            "loss_updates",
            "Number of loss updates performed",
            count + 1,
        )
        self._reporter.report(
            f"neuron_{id(neuron)}_path_count",
            "Total paths processed for neuron",
            stats["m"],
        )
        self._reporter.report(
            f"neuron_{id(neuron)}_rolling_loss",
            "Rolling average of path losses for neuron",
            avg_detached,
        )
        return avg_detached

    def get_stats(self, neuron):
        """Return a dictionary snapshot of statistics for ``neuron``."""
        stats = self._stats.get(neuron)
        if stats is None:
            return {"t": 0, "avg": self._zero, "m": 0}
        return stats.copy()

    def reset(self, neuron=None):
        """Reset statistics for ``neuron`` or all neurons if ``None``."""
        if neuron is None:
            self._stats = {}
        else:
            self._stats.pop(neuron, None)
