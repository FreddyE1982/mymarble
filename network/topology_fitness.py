"""Evaluation of graph topology fitness."""


class TopologyFitness:
    r"""Compute fitness of network topology based on path statistics and complexity.

    The fitness follows Eq. (3.2)::

        F = \sum_s \sum_{P\in S} G(P) - C

    where ``G(P)`` aggregates loss, latency, and cost for a path ``P`` and ``C``
    is the global complexity of the graph. All values are expected to be tensors
    so that gradients can propagate through the computation.

    Parameters
    ----------
    reporter : object, optional
        Object providing a ``report`` method for metric recording.
    """

    def __init__(self, reporter=None):
        self._reporter = reporter

    def evaluate(self, paths_stats, complexity):
        """Return the topology fitness for ``paths_stats`` given ``complexity``.

        ``paths_stats`` should map a source identifier to an iterable of path
        statistic dictionaries. Each statistic dictionary must contain the keys
        ``'loss'``, ``'latency'``, and ``'cost'`` whose values are tensors.
        ``complexity`` is the scalar tensor returned by
        :class:`network.complexity.ComplexityCalculator`.
        """
        zero = complexity * 0 if hasattr(complexity, "__mul__") else 0
        total_score = zero
        total_loss = zero
        total_latency = zero
        total_cost = zero
        if hasattr(paths_stats, "items"):
            sources = paths_stats.items()
        else:
            sources = enumerate(paths_stats)
        for src_key, path_list in sources:
            src_sum = zero
            for p_idx, stats in enumerate(path_list):
                loss = stats.get("loss", zero)
                latency = stats.get("latency", zero)
                cost = stats.get("cost", zero)
                total_loss = total_loss + loss
                total_latency = total_latency + latency
                total_cost = total_cost + cost
                path_value = -(loss + latency + cost)
                src_sum = src_sum + path_value
                if self._reporter is not None:
                    self._reporter.report(
                        f"path_{src_key}_{p_idx}_loss",
                        "Loss contribution for path",
                        loss,
                    )
                    self._reporter.report(
                        f"path_{src_key}_{p_idx}_latency",
                        "Latency contribution for path",
                        latency,
                    )
                    self._reporter.report(
                        f"path_{src_key}_{p_idx}_cost",
                        "Cost contribution for path",
                        cost,
                    )
                    self._reporter.report(
                        f"path_{src_key}_{p_idx}_value",
                        "Aggregated value for path",
                        path_value,
                    )
            total_score = total_score + src_sum
            if self._reporter is not None:
                self._reporter.report(
                    f"source_{src_key}_sum",
                    "Aggregate path value for source",
                    src_sum,
                )
        fitness = total_score - complexity
        if self._reporter is not None:
            self._reporter.report(
                "total_path_loss",
                "Total cumulative path loss",
                total_loss,
            )
            self._reporter.report(
                "total_path_latency",
                "Total cumulative path latency",
                total_latency,
            )
            self._reporter.report(
                "total_path_cost",
                "Total cumulative path cost",
                total_cost,
            )
            self._reporter.report(
                "fitness_value",
                "Overall topology fitness",
                fitness,
            )
        return fitness
