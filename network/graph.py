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

    def __init__(
        self,
        entry_sampler=None,
        path_cost=None,
        path_selector=None,
        reporter=None,
    ):
        self.neurons = {}
        self.synapses = {}
        self._outgoing = {}
        self._incoming = {}
        self._reporter = reporter
        if entry_sampler is None:
            from .entry_sampler import EntrySampler  # local import to avoid module level dependency
            import torch  # local import
            entry_sampler = EntrySampler(temperature=1.0, torch=torch, reporter=reporter)
        self._entry_sampler = entry_sampler
        if path_cost is None:
            from .path_cost import PathCostCalculator  # local import
            path_cost = PathCostCalculator(reporter=reporter)
        self._path_cost = path_cost
        if path_selector is None:
            from .path_selector import PathSelector  # local import
            path_selector = PathSelector(reporter=reporter)
        self._path_selector = path_selector

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

    def _enumerate_paths(self, start_id, visited=None):
        """Return all simple paths starting at ``start_id``.

        Paths are represented as lists alternating between neurons and
        synapses.  Enumeration stops once a terminal neuron (no outgoing
        synapses) is reached or when a cycle would be formed.
        """
        if visited is None:
            visited = set()
        visited.add(start_id)
        neuron = self.neurons[start_id]
        outgoing = self._outgoing.get(start_id, {})
        if not outgoing:
            return [[neuron]]
        paths = []
        for tgt_id, syn_ids in outgoing.items():
            if tgt_id in visited:
                continue
            for sid in syn_ids:
                synapse = self.synapses[sid][2]
                for sub in self._enumerate_paths(tgt_id, visited.copy()):
                    paths.append([neuron, synapse] + sub)
        return paths

    def _aggregate_latency(self, sequence):
        total = 0
        for element in sequence:
            if hasattr(element, "lambda_v"):
                total = total + getattr(element, "lambda_v")
            elif hasattr(element, "lambda_e"):
                total = total + getattr(element, "lambda_e")
        return total

    def forward(self, method="exact", cost_params=None, sample_params=None):
        """Execute one forward selection step through the graph."""
        cost_defaults = {
            "lambda_0": 0,
            "lambda_max": 0,
            "alpha": 1,
            "beta": 1,
            "T_heat": 1,
        }
        if cost_params:
            cost_defaults.update(cost_params)
        sample_defaults = {"R_v_star": 0, "T_sample": 1}
        if sample_params:
            sample_defaults.update(sample_params)
        self._entry_sampler.compute_probabilities(self)
        entry_neuron = self._entry_sampler.sample_entry()
        paths = self._enumerate_paths(entry_neuron.id)
        evaluated = []
        for p in paths:
            cost = self._path_cost.compute_cost(
                p,
                cost_defaults["lambda_0"],
                cost_defaults["lambda_max"],
                cost_defaults["alpha"],
                cost_defaults["beta"],
                cost_defaults["T_heat"],
            )
            latency = self._aggregate_latency(p)
            evaluated.append((p, cost, latency))
        sequences = [(p, c) for p, c, _ in evaluated]
        if method == "soft":
            best, sampled = self._path_selector.select_soft(
                entry_neuron,
                sequences,
                sample_defaults["R_v_star"],
                sample_defaults["T_sample"],
            )
        else:
            best, sampled = self._path_selector.select_exact(entry_neuron, sequences)
        idx = None
        chosen_cost = None
        chosen_latency = None
        for i, (p, c, l) in enumerate(evaluated):
            if p is sampled:
                idx = i
                chosen_cost = c
                chosen_latency = l
                break
        if self._reporter is not None:
            self._reporter.report(
                "entry_id",
                "Identifier of sampled entry neuron",
                entry_neuron.id,
            )
            if idx is not None:
                self._reporter.report(
                    "path_id",
                    "Index of chosen path from entry neuron",
                    idx,
                )
                cost_detached = (
                    chosen_cost.detach() if hasattr(chosen_cost, "detach") else chosen_cost
                )
                self._reporter.report(
                    "path_cost",
                    "Cost of chosen path from entry neuron",
                    cost_detached,
                )
        if sampled:
            for element in sampled:
                if hasattr(element, "update_latency"):
                    element.update_latency(chosen_latency)
                if hasattr(element, "update_cost"):
                    element.update_cost(chosen_cost)
                if hasattr(element, "update_next_min_loss"):
                    element.update_next_min_loss(chosen_cost)
        return sampled
