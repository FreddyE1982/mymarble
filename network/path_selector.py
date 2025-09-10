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

    def _extract_cost(self, path):
        """Return ``(sequence, cost)`` for ``path``.

        ``path`` may either be a plain iterable of graph elements with an
        associated ``cost`` attribute or a two-item iterable ``(sequence,
        cost)``.  When no explicit cost is available the method falls back to
        computing the cost using :class:`network.path_cost.PathCostCalculator`
        with default parameters.  Imports are performed lazily inside the
        method to honour the repository guidelines.
        """

        sequence = path
        cost = None
        if isinstance(path, (list, tuple)) and len(path) == 2:
            sequence, cost = path
        else:
            cost = getattr(path, "cost", None)
        if cost is None:
            from .path_cost import PathCostCalculator  # local import
            calculator = PathCostCalculator(self._reporter)
            cost = calculator.compute_cost(sequence, 0, 0, 1, 1, 1)
        return sequence, cost

    def select_exact(self, neuron, paths):
        r"""Return the minimal cost path and its sampled counterpart.

        The method implements Eq. (1.3.1) by selecting the path ``S`` with the
        lowest cost :math:`C(P)`.  As no stochastic sampling is performed, the
        sampled path ``P`` is identical to ``S``.

        Parameters
        ----------
        neuron : object
            Neuron for which the selection is performed.  Used only for metric
            reporting.
        paths : iterable
            Iterable of paths.  Each path may provide an explicit cost or will
            be evaluated on demand.
        """

        best_index = None
        best_cost = None
        best_path = None
        for idx, raw in enumerate(paths):
            sequence, cost = self._extract_cost(raw)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_index = idx
                best_path = sequence
        sampled_path = best_path
        if self._reporter is not None and best_cost is not None:
            self._reporter.report(
                f"neuron_{id(neuron)}_selected_path",
                "Index of selected path for neuron",
                best_index,
            )
            cost_detached = best_cost.detach() if hasattr(best_cost, "detach") else best_cost
            self._reporter.report(
                f"neuron_{id(neuron)}_sampled_path_cost",
                "Cost of sampled path for neuron",
                cost_detached,
            )
        return best_path, sampled_path

    def select_soft(self, neuron, paths, R_v_star, T_sample):
        r"""Stochastically select a path using a Gumbel-softmax scheme.

        For each feasible path ``P`` the score
        ``R(P) = exp((R_v_star - C(P)) / T_sample)`` is computed and
        normalised into a categorical distribution.  A path is sampled from
        this distribution using a Gumbel-softmax reparameterisation.  The
        method returns both the best path ``S`` (minimal cost) and the sampled
        full path ``P``.

        Parameters
        ----------
        neuron : object
            Neuron for which the selection is performed.
        paths : iterable
            Iterable of candidate paths.
        R_v_star : tensor-like
            Reference reward used in the exponent.
        T_sample : tensor-like
            Sampling temperature controlling exploration.
        """

        sequences = []
        costs = []
        for raw in paths:
            sequence, cost = self._extract_cost(raw)
            sequences.append(sequence)
            costs.append(cost)
        if not costs:
            return None, None
        import torch  # local import
        from torch.nn.functional import gumbel_softmax  # local import

        cost_tensors = [c if hasattr(c, "shape") else torch.tensor(c) for c in costs]
        stacked = torch.stack(cost_tensors)
        R_v_star_t = R_v_star if hasattr(R_v_star, "shape") else torch.tensor(R_v_star)
        T_sample_t = T_sample if hasattr(T_sample, "shape") else torch.tensor(T_sample)
        rewards = torch.exp((R_v_star_t - stacked) / T_sample_t)
        probs = rewards / rewards.sum()
        sample_one_hot = gumbel_softmax(torch.log(probs), tau=1.0, hard=True)
        sampled_index = int(sample_one_hot.argmax().item())
        best_index = int(torch.argmin(stacked).item())
        sampled_path = sequences[sampled_index]
        best_path = sequences[best_index]
        sampled_cost = cost_tensors[sampled_index]
        if self._reporter is not None:
            self._reporter.report(
                f"neuron_{id(neuron)}_selected_path",
                "Index of selected path for neuron",
                best_index,
            )
            cost_detached = sampled_cost.detach() if hasattr(sampled_cost, "detach") else sampled_cost
            self._reporter.report(
                f"neuron_{id(neuron)}_sampled_path_cost",
                "Cost of sampled path for neuron",
                cost_detached,
            )
        return best_path, sampled_path
