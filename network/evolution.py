"""Evolutionary operators for graph mutations.

This module defines the :class:`EvolutionOperator` responsible for mutating a
neural graph through addition, removal and relocation of neurons and synapses.
Each mutation follows Eq. (3.3) variants and evaluates delta thresholds before
applying changes.  Complexity and fitness values are recorded before and after
mutations to allow detailed analysis of the evolutionary process.
"""


class EvolutionOperator:
    """Perform structural mutations on a neural graph.

    Parameters
    ----------
    graph : object
        Graph instance providing ``add_neuron``, ``remove_neuron``,
        ``add_synapse``, ``remove_synapse`` and related methods.
    complexity_calculator : object
        Instance of :class:`~network.complexity.ComplexityCalculator`.
    fitness_evaluator : object
        Instance of :class:`~network.topology_fitness.TopologyFitness`.
    theta_add : object
        Threshold :math:`\theta_{add}` for neuron insertion.
    theta_remove : object
        Threshold :math:`\theta_{rem}` for neuron removal.
    theta_add_synapse : object
        Threshold :math:`\theta_{add,e}` for synapse insertion.
    theta_remove_synapse : object
        Threshold :math:`\theta_{rem,e}` for synapse deletion.
    theta_move_synapse : object
        Threshold :math:`\theta_{move,e}` for synapse relocation.
    reporter : object, optional
        Metrics collector implementing ``report(name, desc, value)``.
    """

    def __init__(
        self,
        graph,
        complexity_calculator,
        fitness_evaluator,
        theta_add,
        theta_remove,
        theta_add_synapse,
        theta_remove_synapse,
        theta_move_synapse,
        reporter=None,
    ):
        self._graph = graph
        self._complexity = complexity_calculator
        self._fitness = fitness_evaluator
        self._theta_add = theta_add
        self._theta_remove = theta_remove
        self._theta_add_synapse = theta_add_synapse
        self._theta_remove_synapse = theta_remove_synapse
        self._theta_move_synapse = theta_move_synapse
        self._reporter = reporter

    def _log_state(self, pre_c, pre_f, post_c, post_f):
        if self._reporter is not None:
            self._reporter.report(
                "complexity_before",
                "Global complexity prior to mutation",
                pre_c,
            )
            self._reporter.report(
                "fitness_before",
                "Topology fitness prior to mutation",
                pre_f,
            )
            self._reporter.report(
                "complexity_after",
                "Global complexity following mutation",
                post_c,
            )
            self._reporter.report(
                "fitness_after",
                "Topology fitness following mutation",
                post_f,
            )

    def _should_apply(self, delta, threshold):
        if hasattr(delta, "__ge__"):
            decision = delta >= threshold
            if hasattr(decision, "item"):
                return bool(decision.item())
            return bool(decision)
        return False

    def add_neuron(self, neuron_ctx, paths_stats=None):
        pre_c = self._complexity.compute(self._graph)
        pre_f = self._fitness.evaluate(paths_stats or {}, pre_c)
        zero = pre_c * 0 if hasattr(pre_c, "__mul__") else 0
        b = neuron_ctx.get("b_vn", zero)
        g = neuron_ctx.get("g_vn", zero)
        latency = neuron_ctx.get("latency", zero)
        cost = neuron_ctx.get("cost", zero)
        delta = b + g - latency - cost
        if self._reporter is not None:
            self._reporter.report(
                "add_neuron_delta", "Delta for neuron addition", delta
            )
        applied = False
        if self._should_apply(delta, self._theta_add):
            neuron_id = neuron_ctx.get("neuron_id")
            neuron = neuron_ctx.get("neuron")
            src_id = neuron_ctx.get("source_id")
            tgt_id = neuron_ctx.get("target_id")
            syn_in_id = neuron_ctx.get("synapse_in_id")
            syn_in = neuron_ctx.get("synapse_in")
            syn_out_id = neuron_ctx.get("synapse_out_id")
            syn_out = neuron_ctx.get("synapse_out")
            self._graph.add_neuron(neuron_id, neuron)
            self._graph.add_synapse(syn_in_id, src_id, neuron_id, syn_in)
            self._graph.add_synapse(syn_out_id, neuron_id, tgt_id, syn_out)
            applied = True
        post_c = self._complexity.compute(self._graph)
        post_f = self._fitness.evaluate(paths_stats or {}, post_c)
        self._log_state(pre_c, pre_f, post_c, post_f)
        if self._reporter is not None:
            self._reporter.report(
                "add_neuron_applied", "Neuron addition executed", int(applied)
            )
        return applied

    def remove_neuron(self, neuron_id, metrics, paths_stats=None):
        pre_c = self._complexity.compute(self._graph)
        pre_f = self._fitness.evaluate(paths_stats or {}, pre_c)
        neuron = self._graph.get_neuron(neuron_id)
        zero = pre_c * 0 if hasattr(pre_c, "__mul__") else 0
        b = metrics.get("b_vn", getattr(neuron, "b_vn", zero))
        g = metrics.get("g_vn", getattr(neuron, "g_vn", zero))
        latency = metrics.get("latency", getattr(neuron, "lambda_v", zero))
        cost = metrics.get("cost", getattr(neuron, "c_v", zero))
        delta = b - g - latency - cost
        if self._reporter is not None:
            self._reporter.report(
                "remove_neuron_delta", "Delta for neuron removal", delta
            )
        applied = False
        if self._should_apply(delta, self._theta_remove):
            self._graph.remove_neuron(neuron_id)
            applied = True
        post_c = self._complexity.compute(self._graph)
        post_f = self._fitness.evaluate(paths_stats or {}, post_c)
        self._log_state(pre_c, pre_f, post_c, post_f)
        if self._reporter is not None:
            self._reporter.report(
                "remove_neuron_applied", "Neuron removal executed", int(applied)
            )
        return applied

    def add_synapse(self, syn_ctx, paths_stats=None):
        pre_c = self._complexity.compute(self._graph)
        pre_f = self._fitness.evaluate(paths_stats or {}, pre_c)
        zero = pre_c * 0 if hasattr(pre_c, "__mul__") else 0
        b = syn_ctx.get("b_e", zero)
        g = syn_ctx.get("g_e", zero)
        latency = syn_ctx.get("latency", zero)
        cost = syn_ctx.get("cost", zero)
        delta = b + g - latency - cost
        if self._reporter is not None:
            self._reporter.report(
                "add_synapse_delta", "Delta for synapse addition", delta
            )
        applied = False
        if self._should_apply(delta, self._theta_add_synapse):
            syn_id = syn_ctx.get("synapse_id")
            src = syn_ctx.get("source_id")
            tgt = syn_ctx.get("target_id")
            syn = syn_ctx.get("synapse")
            self._graph.add_synapse(syn_id, src, tgt, syn)
            applied = True
        post_c = self._complexity.compute(self._graph)
        post_f = self._fitness.evaluate(paths_stats or {}, post_c)
        self._log_state(pre_c, pre_f, post_c, post_f)
        if self._reporter is not None:
            self._reporter.report(
                "add_synapse_applied", "Synapse addition executed", int(applied)
            )
        return applied

    def remove_synapse(self, synapse_id, metrics, paths_stats=None):
        pre_c = self._complexity.compute(self._graph)
        pre_f = self._fitness.evaluate(paths_stats or {}, pre_c)
        synapse = self._graph.get_synapse(synapse_id)
        zero = pre_c * 0 if hasattr(pre_c, "__mul__") else 0
        b = metrics.get("b_e", getattr(synapse, "b_e", zero))
        g = metrics.get("g_e", getattr(synapse, "g_e", zero))
        latency = metrics.get("latency", getattr(synapse, "lambda_e", zero))
        cost = metrics.get("cost", getattr(synapse, "c_e", zero))
        delta = b - g - latency - cost
        if self._reporter is not None:
            self._reporter.report(
                "remove_synapse_delta", "Delta for synapse removal", delta
            )
        applied = False
        if self._should_apply(delta, self._theta_remove_synapse):
            self._graph.remove_synapse(synapse_id)
            applied = True
        post_c = self._complexity.compute(self._graph)
        post_f = self._fitness.evaluate(paths_stats or {}, post_c)
        self._log_state(pre_c, pre_f, post_c, post_f)
        if self._reporter is not None:
            self._reporter.report(
                "remove_synapse_applied", "Synapse removal executed", int(applied)
            )
        return applied

    def move_synapse(
        self,
        synapse_id,
        new_source_id,
        new_target_id,
        metrics,
        paths_stats=None,
    ):
        pre_c = self._complexity.compute(self._graph)
        pre_f = self._fitness.evaluate(paths_stats or {}, pre_c)
        synapse = self._graph.get_synapse(synapse_id)
        zero = pre_c * 0 if hasattr(pre_c, "__mul__") else 0
        b = metrics.get("b_e", getattr(synapse, "b_e", zero))
        g = metrics.get("g_e", getattr(synapse, "g_e", zero))
        latency = metrics.get("latency", getattr(synapse, "lambda_e", zero))
        cost = metrics.get("cost", getattr(synapse, "c_e", zero))
        delta = b + g - latency - cost
        if self._reporter is not None:
            self._reporter.report(
                "move_synapse_delta", "Delta for synapse move", delta
            )
        applied = False
        if self._should_apply(delta, self._theta_move_synapse):
            src_old, tgt_old, syn = self._graph.synapses.get(synapse_id)
            self._graph.remove_synapse(synapse_id)
            self._graph.add_synapse(synapse_id, new_source_id, new_target_id, syn)
            applied = True
        post_c = self._complexity.compute(self._graph)
        post_f = self._fitness.evaluate(paths_stats or {}, post_c)
        self._log_state(pre_c, pre_f, post_c, post_f)
        if self._reporter is not None:
            self._reporter.report(
                "move_synapse_applied", "Synapse move executed", int(applied)
            )
        return applied
