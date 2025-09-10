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
        path_forwarder=None,
        latency_estimator=None,
        reporter=None,
        routing_adjuster=None,
        complexity_calculator=None,
        topology_fitness=None,
        evolution_operator=None,
    ):
        self.neurons = {}
        self.synapses = {}
        self._outgoing = {}
        self._incoming = {}
        self._reporter = reporter
        if entry_sampler is None:
            from .entry_sampler import EntrySampler  # local import to avoid module level dependency
            import torch  # local import
            entry_sampler = EntrySampler(
                temperature=1.0,
                torch=torch,
                reporter=reporter,
                zero=torch.tensor(0.0),
            )
        self._entry_sampler = entry_sampler
        if path_cost is None:
            from .path_cost import PathCostCalculator  # local import
            path_cost = PathCostCalculator(reporter=reporter)
        self._path_cost = path_cost
        if path_selector is None:
            from .path_selector import PathSelector  # local import
            path_selector = PathSelector(reporter=reporter)
        self._path_selector = path_selector
        if path_forwarder is None:
            from .path_forwarder import PathForwarder  # local import
            path_forwarder = PathForwarder(reporter=reporter)
        self._path_forwarder = path_forwarder
        if latency_estimator is None:
            from .latency import LatencyEstimator  # local import
            latency_estimator = LatencyEstimator(reporter=reporter)
        self._latency_estimator = latency_estimator
        if routing_adjuster is None:
            from routing.improvements import GateAdjuster  # local import
            routing_adjuster = GateAdjuster(reporter=reporter)
        self._routing_adjuster = routing_adjuster
        if complexity_calculator is None:
            from .complexity import ComplexityCalculator  # local import
            import torch  # local import
            complexity_calculator = ComplexityCalculator(
                [],
                [],
                torch.tensor(0.0),
                torch.tensor(0.0),
                [],
                reporter=reporter,
            )
        self._complexity_calculator = complexity_calculator
        if topology_fitness is None:
            from .topology_fitness import TopologyFitness  # local import
            topology_fitness = TopologyFitness(reporter=reporter)
        self._topology_fitness = topology_fitness
        if evolution_operator is None:
            from .evolution import EvolutionOperator  # local import
            import torch  # local import
            evolution_operator = EvolutionOperator(
                self,
                self._complexity_calculator,
                self._topology_fitness,
                torch.tensor(float("inf")),
                torch.tensor(float("inf")),
                torch.tensor(float("inf")),
                torch.tensor(float("inf")),
                torch.tensor(float("inf")),
                reporter=reporter,
            )
        self._evolution_operator = evolution_operator

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

    def forward(
        self,
        method="exact",
        cost_params=None,
        sample_params=None,
        global_loss_target=None,
        activations=None,
        evolution_instructions=None,
    ):
        """Execute one forward selection step through the graph.

        Parameters
        ----------
        method : str, optional
            Path selection strategy. Defaults to "exact".
        cost_params : dict, optional
            Overrides for cost calculation parameters.
        sample_params : dict, optional
            Overrides for sampling parameters when ``method="soft"``.
        global_loss_target : object, optional
            Accepted for backward compatibility. Currently unused.
        activations : dict, optional
            Accepted for backward compatibility. Currently unused.
        """
        if activations is None:
            activations = {}
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
        telemetry = {}
        neuron_sequence = []
        if sampled and self._latency_estimator is not None:
            for i in range(0, len(sampled), 2):
                neuron = sampled[i]
                nid = getattr(neuron, "id", None)
                if nid is None:
                    nid = next(k for k, v in self.neurons.items() if v is neuron)
                syn_map = {}
                if i + 1 < len(sampled):
                    synapse = sampled[i + 1]
                    sid = getattr(synapse, "id", None)
                    if sid is None:
                        sid = next(
                            k for k, meta in self.synapses.items() if meta[2] is synapse
                        )
                    syn_map[sid] = synapse
                self._latency_estimator.update(nid, neuron, syn_map)
        if sampled and self._path_forwarder is not None:
            neuron_sequence = [sampled[i] for i in range(0, len(sampled), 2)]
            step_losses = [getattr(n, "last_local_loss", 0) for n in neuron_sequence]
            telemetry = self._path_forwarder.run(neuron_sequence, step_losses)
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
            if telemetry:
                self._reporter.report(
                    "path_time",
                    "Total traversal time for chosen path",
                    telemetry.get("path_time", 0),
                )
                self._reporter.report(
                    "final_cumulative_loss",
                    "Final cumulative loss after forwarding along path",
                    telemetry.get("final_loss", 0),
                )
        if sampled:
            for element in sampled:
                if hasattr(element, "update_cost"):
                    element.update_cost(chosen_cost)
                if hasattr(element, "update_next_min_loss"):
                    element.update_next_min_loss(chosen_cost)
        import torch  # local import
        paths_stats = {}
        if evaluated:
            zero = chosen_cost * 0 if chosen_cost is not None else torch.tensor(0.0)
            stats_list = []
            for p, c, l in evaluated:
                loss = zero
                for n in p[::2]:
                    val = getattr(n, "last_local_loss", zero)
                    if not hasattr(val, "shape"):
                        val = torch.tensor(val)
                    loss = loss + val
                stats_list.append({"loss": loss, "latency": l, "cost": c})
            if idx is not None and telemetry.get("final_loss") is not None:
                stats_list[idx]["loss"] = telemetry["final_loss"]
            paths_stats[entry_neuron.id] = stats_list
        complexity = self._complexity_calculator.compute(self)
        topology_fitness = self._topology_fitness.evaluate(paths_stats, complexity)
        if self._reporter is not None:
            self._reporter.report(
                "complexity",
                "Global complexity of graph",
                complexity,
            )
            self._reporter.report(
                "topology_fitness",
                "Topology fitness of graph",
                topology_fitness,
            )
        mutation_results = {}
        if evolution_instructions:
            if "add_neuron" in evolution_instructions:
                applied = self._evolution_operator.add_neuron(
                    evolution_instructions["add_neuron"], paths_stats
                )
                mutation_results["add_neuron"] = applied
            if "remove_neuron" in evolution_instructions:
                neuron_id, metrics = evolution_instructions["remove_neuron"]
                applied = self._evolution_operator.remove_neuron(
                    neuron_id, metrics, paths_stats
                )
                mutation_results["remove_neuron"] = applied
            if "add_synapse" in evolution_instructions:
                applied = self._evolution_operator.add_synapse(
                    evolution_instructions["add_synapse"], paths_stats
                )
                mutation_results["add_synapse"] = applied
            if "remove_synapse" in evolution_instructions:
                syn_id, metrics = evolution_instructions["remove_synapse"]
                applied = self._evolution_operator.remove_synapse(
                    syn_id, metrics, paths_stats
                )
                mutation_results["remove_synapse"] = applied
            if "move_synapse" in evolution_instructions:
                syn_id, new_src, new_tgt, metrics = evolution_instructions["move_synapse"]
                applied = self._evolution_operator.move_synapse(
                    syn_id, new_src, new_tgt, metrics, paths_stats
                )
                mutation_results["move_synapse"] = applied
        if self._routing_adjuster is not None and neuron_sequence:
            per_cost = (
                chosen_cost / len(neuron_sequence)
                if chosen_cost is not None and len(neuron_sequence) > 0
                else torch.tensor(0.0)
            )
            for neuron in neuron_sequence:
                threshold = getattr(neuron, "activation_threshold", None)
                if threshold is None:
                    continue
                grad = getattr(threshold, "grad", None)
                if grad is None:
                    grad = torch.zeros_like(threshold)
                latency = getattr(neuron, "lambda_v", torch.tensor(0.0))
                stats = {"gradient": grad, "latency": latency, "cost": per_cost}
                self._routing_adjuster.adjust_gate(neuron, stats)
        result = {"path": sampled, "complexity": complexity, "topology_fitness": topology_fitness}
        if telemetry:
            result["path_time"] = telemetry.get("path_time", 0)
            result["final_cumulative_loss"] = telemetry.get("final_loss", 0)
        if mutation_results:
            result["mutations"] = mutation_results
        return result
