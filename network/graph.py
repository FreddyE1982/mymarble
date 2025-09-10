"""Directed multigraph managing neurons and synapses.

The :class:`Graph` class provides management utilities for registering neurons
and synapses and for navigating the connections between them.  The
implementation is import free and agnostic regarding the concrete tensor
implementation used inside the entities.  Objects of type :class:`Neuron` and
:class:`Synapse` are expected to be created by the caller and supplied to the
graph.
"""


class Graph:
    """A directed multigraph of neurons and synapses."""

    def __init__(self, path_selector=None, latency_estimator=None, reporter=None):
        self.neurons = {}
        self.synapses = {}
        self._outgoing = {}
        self._incoming = {}
        if path_selector is None:
            from .path_selector import PathSelector  # local import to avoid module level dependency
            path_selector = PathSelector(reporter=reporter)
        self._path_selector = path_selector
        if latency_estimator is None:
            from .latency import LatencyEstimator  # local import to avoid module level dependency
            latency_estimator = LatencyEstimator(reporter=reporter)
        self._latency_estimator = latency_estimator

    def add_neuron(self, neuron_id, neuron):
        """Add a neuron to the graph."""
        if neuron_id in self.neurons:
            raise ValueError("Neuron already exists")
        self.neurons[neuron_id] = neuron
        self._outgoing[neuron_id] = {}
        self._incoming[neuron_id] = {}

    def remove_neuron(self, neuron_id):
        if neuron_id not in self.neurons:
            return
        for syn_id in list(self._collect_synapses(neuron_id)):
            self.remove_synapse(syn_id)
        del self.neurons[neuron_id]
        del self._outgoing[neuron_id]
        del self._incoming[neuron_id]

    def add_synapse(self, synapse_id, source_id, target_id, synapse):
        """Add a synapse connecting ``source_id`` to ``target_id``."""
        if synapse_id in self.synapses:
            raise ValueError("Synapse already exists")
        if source_id not in self.neurons or target_id not in self.neurons:
            raise KeyError("Both neurons must exist before adding synapse")
        self.synapses[synapse_id] = (source_id, target_id, synapse)
        self._outgoing[source_id].setdefault(target_id, []).append(synapse_id)
        self._incoming[target_id].setdefault(source_id, []).append(synapse_id)

    def remove_synapse(self, synapse_id):
        if synapse_id not in self.synapses:
            return
        src, tgt, _ = self.synapses.pop(synapse_id)
        outs = self._outgoing.get(src, {}).get(tgt, [])
        if synapse_id in outs:
            outs.remove(synapse_id)
            if not outs:
                del self._outgoing[src][tgt]
        ins = self._incoming.get(tgt, {}).get(src, [])
        if synapse_id in ins:
            ins.remove(synapse_id)
            if not ins:
                del self._incoming[tgt][src]

    def get_neuron(self, neuron_id):
        return self.neurons.get(neuron_id)

    def get_synapse(self, synapse_id):
        meta = self.synapses.get(synapse_id)
        if meta is None:
            return None
        return meta[2]

    def outgoing(self, neuron_id):
        edges = self._outgoing.get(neuron_id, {})
        return [(tgt, [self.synapses[sid][2] for sid in sids]) for tgt, sids in edges.items()]

    def incoming(self, neuron_id):
        edges = self._incoming.get(neuron_id, {})
        return [(src, [self.synapses[sid][2] for sid in sids]) for src, sids in edges.items()]

    def get_synapses(self, source_id, target_id):
        ids = self._outgoing.get(source_id, {}).get(target_id, [])
        return [self.synapses[sid][2] for sid in ids]

    def _collect_synapses(self, neuron_id):
        outs = self._outgoing.get(neuron_id, {})
        ins = self._incoming.get(neuron_id, {})
        for sids in outs.values():
            for sid in sids:
                yield sid
        for sids in ins.values():
            for sid in sids:
                yield sid

    def forward(self, global_loss_target, activations=None):
        """Select paths for all neurons using the configured PathSelector.

        Parameters
        ----------
        global_loss_target : object
            Target loss used by the scoring function.
        activations : dict, optional
            Optional mapping ``{neuron_id: activation_state}`` supplying
            additional per-neuron activation tensors.

        Returns
        -------
        dict
            Mapping of neuron identifiers to the selected outgoing synapse or
            ``None`` if a neuron has no outgoing synapses.
        """
        selections = {}
        if activations is None:
            activations = {}
        for nid, neuron in self.neurons.items():
            outgoing_ids = []
            for sids in self._outgoing.get(nid, {}).values():
                outgoing_ids.extend(sids)
            synapse_map = {sid: self.synapses[sid][2] for sid in outgoing_ids}
            self._latency_estimator.update(nid, neuron, synapse_map)
            synapses = list(synapse_map.values())
            state = {
                "outgoing_synapses": synapses,
                "activation_tensors": activations.get(nid, {}),
                "global_loss_target": global_loss_target,
            }
            selections[nid] = self._path_selector.select_path(neuron, state)
        return selections
