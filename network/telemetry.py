"""5. Telemetry updates and definitions

After the backward pass, update stored tensors for neurons on the path (or weighted by g_v in soft routing):

- Timestamps: τ_t ← t_cur (tensor clock).
- Step times: s_t ← EMA of recent measured forward latency for t.
- Losses: ℓ_t ← ℓ_t(y_t, y^*), L_t^cum per (2.1).
- Speed of loss decrease: r_t per (0.1).

These are all tensors retained in the model state and may feed routing costs c_t (1.2), and evolutionary scores (3.4–3.5).
"""


class TelemetryUpdater:
    """Update telemetry tensors for a path of neurons."""

    def __init__(self, reporter=None, time_source=None, ema_alpha=0.5):
        self._reporter = reporter
        if time_source is None:
            from time import perf_counter  # local import
            self._time_source = perf_counter
        else:
            self._time_source = time_source
        self._alpha = ema_alpha

    def update(self, path, losses, g_v=None):
        """Apply telemetry updates for ``path`` using ``losses``.

        Parameters
        ----------
        path : iterable
            Sequence of neurons to update.
        losses : iterable
            Per-neuron loss tensors.
        g_v : dict, optional
            Optional mapping of neurons or identifiers to gate weights.
        """
        g_v = g_v or {}
        final_loss = None
        for neuron, loss in zip(path, losses):
            t_curr = self._time_source()
            key = neuron if neuron in g_v else getattr(neuron, "id", None)
            weight = g_v.get(key, 1.0)
            weighted_loss = loss * weight
            neuron.update_cumulative_loss(weighted_loss, self._reporter)
            neuron.record_activation(t_curr, t_curr, self._reporter)
            latency = getattr(neuron, "lambda_v", 0) * weight
            neuron.update_step_time(latency, self._alpha, self._reporter)
            final_loss = getattr(neuron, "cumulative_loss", final_loss)
        return {"final_loss": final_loss} if final_loss is not None else {}
