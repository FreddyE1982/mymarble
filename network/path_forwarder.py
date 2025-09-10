r"""Utilities for forwarding along a selected neuron path.

The :class:`PathForwarder` traverses a sequence of neurons, updating their
cumulative losses and recording activation timestamps.  The total traversal
time is computed according to Eq. (2.3)::

    S(P) = \\sum_{k=0}^K (s_{v_k} - s_{v_{k-1}})

All timing metrics are reported through an optional reporter compatible with
:class:`main.Reporter`.
"""


class PathForwarder:
    """Execute neuron updates along a path while measuring timing information.

    Parameters
    ----------
    reporter : object, optional
        Object providing a ``report`` method.  Metrics are written to this
        reporter when supplied.
    time_source : callable, optional
        Function returning a monotonically increasing timestamp.  Defaults to
        :func:`time.perf_counter`.
    """

    def __init__(self, reporter=None, time_source=None):
        self._reporter = reporter
        if time_source is None:
            from time import perf_counter  # local import
            self._time_source = perf_counter
        else:
            self._time_source = time_source

    def run(self, path, losses):
        """Traverse ``path`` applying ``losses`` and record timing metrics.

        The method iterates over ``path`` and ``losses`` simultaneously,
        retrieving a timestamp ``t_curr`` for each hop.  For every neuron the
        corresponding loss is forwarded via
        :meth:`network.entities.Neuron.update_cumulative_loss` and the
        activation is recorded with :meth:`network.entities.Neuron.record_activation`.
        Durations between successive timestamps are accumulated to yield the
        total path time ``S(P)``.

        Parameters
        ----------
        path : iterable
            Sequence of neurons to traverse.
        losses : iterable
            Sequence of loss tensors or numbers, one per neuron in ``path``.

        Returns
        -------
        dict
            Mapping containing ``path_time`` and ``final_loss`` where
            ``final_loss`` corresponds to the cumulative loss of the last
            neuron in ``path``.
        """

        start_time = self._time_source()
        prev_time = start_time
        total_time = 0.0
        last_neuron = None
        for neuron, loss in zip(path, losses):
            t_curr = self._time_source()
            neuron.update_cumulative_loss(loss, self._reporter)
            neuron.record_activation(t_curr, t_curr, self._reporter)
            hop_duration = t_curr - prev_time
            total_time += hop_duration
            prev_time = t_curr
            last_neuron = neuron
        final_loss = getattr(last_neuron, "cumulative_loss", 0)
        if self._reporter is not None:
            self._reporter.report(
                "path_total_time",
                "Total traversal time for processed path",
                total_time,
            )
        return {"path_time": total_time, "final_loss": final_loss}
