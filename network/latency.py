"""Latency estimation utilities for neural graphs.

This module implements the :class:`LatencyEstimator` which tracks activation
latencies for neurons and synapses without introducing any module level
imports.  The estimator maintains the time between consecutive activations of
entities and writes the measured latency to the ``lambda_v`` and ``lambda_e``
fields of :class:`network.entities.Neuron` and
:class:`network.entities.Synapse` respectively.  The per-synapse cost ``c_e`` is
set equal to the measured latency.  All updates are reported through a provided
reporter instance.
"""


class LatencyEstimator:
    """Estimate and record activation latencies.

    Parameters
    ----------
    reporter : object
        Object providing a ``report`` method compatible with
        :class:`main.Reporter`.  Metrics are emitted via this object.  If
        ``None`` is provided no metrics are recorded.
    zero : object, optional
        Zero-like tensor used to initialise latency values.  Defaults to ``0``.
    """

    def __init__(self, reporter=None, zero=0):
        self._reporter = reporter
        self._zero = zero
        self._neuron_times = {}
        self._synapse_times = {}

    def _now(self):
        from time import perf_counter

        return perf_counter()

    def update(self, neuron_id, neuron, synapses):
        """Update latency tensors for ``neuron`` and ``synapses``.

        Parameters
        ----------
        neuron_id : hashable
            Identifier of the neuron being updated.
        neuron : :class:`network.entities.Neuron`
            The neuron whose latency is updated.
        synapses : dict
            Mapping ``{synapse_id: synapse}`` of outgoing synapses for which
            latency should be recorded.
        """
        now = self._now()
        last = self._neuron_times.get(neuron_id, now)
        latency = now - last
        self._neuron_times[neuron_id] = now
        latency_tensor = self._zero + latency
        neuron.update_latency(latency_tensor)
        if self._reporter is not None:
            self._reporter.report(
                f"latency_neuron_{neuron_id}",
                f"Activation latency for neuron {neuron_id}",
                latency_tensor,
            )
        for sid, syn in synapses.items():
            last_s = self._synapse_times.get(sid, now)
            lat_s = now - last_s
            self._synapse_times[sid] = now
            lat_tensor = self._zero + lat_s
            syn.update_latency(lat_tensor)
            syn.update_cost(lat_tensor)
            if self._reporter is not None:
                self._reporter.report(
                    f"latency_synapse_{sid}",
                    f"Activation latency for synapse {sid}",
                    lat_tensor,
                )
        return latency_tensor

    def peek(self, neuron_id, neuron, synapses):
        """Temporarily assign latency values without mutating state.

        Parameters
        ----------
        neuron_id : hashable
            Identifier of the neuron being inspected.
        neuron : :class:`network.entities.Neuron`
            The neuron whose latency is estimated.
        synapses : dict
            Mapping ``{synapse_id: synapse}`` of outgoing synapses for which
            latency should be estimated.

        Returns
        -------
        tuple
            ``(neuron_latency, {sid: (lambda_e, c_e)})`` capturing the original
            latency and cost values so they can be restored using
            :meth:`restore`.
        """
        now = self._now()
        last = self._neuron_times.get(neuron_id, now)
        latency = now - last
        latency_tensor = self._zero + latency
        original_neuron = getattr(neuron, "lambda_v", self._zero)
        neuron.update_latency(latency_tensor)
        syn_backup = {}
        for sid, syn in synapses.items():
            last_s = self._synapse_times.get(sid, now)
            lat_s = now - last_s
            lat_tensor = self._zero + lat_s
            syn_backup[sid] = (
                getattr(syn, "lambda_e", self._zero),
                getattr(syn, "c_e", self._zero),
            )
            syn.update_latency(lat_tensor)
            syn.update_cost(lat_tensor)
        return original_neuron, syn_backup

    def restore(self, neuron, synapses, backup):
        """Restore latency values recorded by :meth:`peek`.

        Parameters
        ----------
        neuron : :class:`network.entities.Neuron`
            The neuron whose latency should be restored.
        synapses : dict
            Mapping ``{synapse_id: synapse}`` of outgoing synapses.
        backup : tuple
            Data returned by :meth:`peek` used to restore original state.
        """
        original_neuron, syn_backup = backup
        neuron.update_latency(original_neuron)
        for sid, syn in synapses.items():
            lambda_e, cost = syn_backup.get(sid, (self._zero, self._zero))
            syn.update_latency(lambda_e)
            syn.update_cost(cost)
