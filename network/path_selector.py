class PathSelector:
    r"""Select outgoing synapses based on a configurable scoring function.

    The selector evaluates each candidate synapse using a latency weighted
    scoring function as defined by Eq. (0.2).  The default implementation of
    this equation is::

        f = w(\lambda_e) * \lambda_e + |(L_v + c_e) - L^*| + \sum_i h_i

    where ``w(\lambda_e)`` is a latency weighting function, ``\lambda_e`` the
    synapse latency, ``L_v`` the neuron's last recorded local loss, ``c_e`` a
    synapse cost term, ``L^*`` the global loss target, and ``h_i`` additional
    penalties supplied via hooks.  The synapse (or sequence) with the minimum
    score is returned.

    Parameters
    ----------
    latency_weight : float or callable, optional
        Weight applied to latency.  If a callable is provided it is invoked with
        the synapse and should return a numeric weight.  Defaults to ``1.0``.
    loss_hooks : iterable of callables, optional
        Additional scoring terms.  Each hook is called as
        ``hook(neuron, synapse, state)`` and should return a numeric penalty.
    reporter : object, optional
        Object providing a ``report`` method compatible with
        :class:`main.Reporter`.  If supplied, selection metrics are recorded
        through it.
    """

    def __init__(self, latency_weight=1.0, loss_hooks=None, reporter=None):
        if callable(latency_weight):
            self._latency_weight = latency_weight
        else:
            self._latency_weight = lambda synapse: latency_weight
        self._loss_hooks = list(loss_hooks) if loss_hooks else []
        self._reporter = reporter
        self._selection_count = 0

    def add_hook(self, hook):
        """Register an additional loss hook."""
        self._loss_hooks.append(hook)

    def _score(self, neuron, synapse, state):
        latency = getattr(synapse, "lambda_e", 0)
        weight = self._latency_weight(synapse)
        local_loss = getattr(neuron, "last_local_loss", 0)
        syn_cost = getattr(synapse, "c_e", 0)
        target = state.get("global_loss_target", 0)
        loss_term = abs((local_loss + syn_cost) - target)
        score = weight * latency + loss_term
        for hook in self._loss_hooks:
            score += hook(neuron, synapse, state)
        return score

    def select_path(self, neuron, graph_state):
        """Return the outgoing synapse minimising the scoring function."""
        outgoing = graph_state.get("outgoing_synapses") or graph_state.get("outgoing") or []
        best_synapse = None
        best_score = None
        for synapse in outgoing:
            score = self._score(neuron, synapse, graph_state)
            if best_score is None or score < best_score:
                best_score = score
                best_synapse = synapse
        if self._reporter is not None:
            self._selection_count += 1
            count = self._reporter.report("path_selector_calls") or 0
            self._reporter.report(
                "path_selector_calls",
                "Number of path selections performed",
                count + 1,
            )
            if best_score is not None:
                self._reporter.report(
                    "path_selector_last_score",
                    "Score of last selected path",
                    best_score,
                )
        return best_synapse
