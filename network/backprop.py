"""Backward pass utilities with active subgraph gating."""


class Backpropagator:
    """Perform backward computations over an active subgraph.

    The class builds gate masks for vertices and edges according to the
    provided routing mode and evaluates the sample loss using these masks.
    All heavy imports are executed lazily inside the respective methods to
    comply with repository constraints.
    """

    def __init__(self, reporter=None):
        self._reporter = reporter

    def build_active_subgraph(self, graph, path, routing_mode):
        """Create gate tensors for the active subgraph.

        Parameters
        ----------
        graph : network.graph.Graph
            Graph containing neurons and synapses.
        path : sequence
            Alternating sequence of neurons and synapses describing the
            selected path.
        routing_mode : str
            Either ``"hard"`` or ``"soft"``.  Hard routing activates only
            elements present in ``path`` while soft routing activates all
            elements.

        Returns
        -------
        dict
            Dictionary with ``g_v`` and ``g_e`` entries mapping neuron and
            synapse identifiers to their respective gate tensors.
        """
        import torch  # local import

        path_set = set(path) if path is not None else set()
        g_v = {}
        g_e = {}
        active_vertices = 0
        active_edges = 0
        for nid, neuron in graph.neurons.items():
            gamma_v = getattr(neuron, "gate", torch.tensor(0.0))
            zero_like = gamma_v * 0 if hasattr(gamma_v, "__mul__") else torch.tensor(0.0)
            if routing_mode == "hard" and neuron not in path_set:
                gate = zero_like
            else:
                gate = gamma_v
            g_v[nid] = gate
            value = gate.detach() if hasattr(gate, "detach") else gate
            if hasattr(value, "abs"):
                nonzero = bool(value.abs().sum().item())
            else:
                nonzero = bool(value)
            if nonzero:
                active_vertices += 1
        for sid, (_, _, synapse) in graph.synapses.items():
            gamma_e = getattr(synapse, "gate", torch.tensor(0.0))
            zero_like = gamma_e * 0 if hasattr(gamma_e, "__mul__") else torch.tensor(0.0)
            if routing_mode == "hard" and synapse not in path_set:
                gate = zero_like
            else:
                gate = gamma_e
            g_e[sid] = gate
            value = gate.detach() if hasattr(gate, "detach") else gate
            if hasattr(value, "abs"):
                nonzero = bool(value.abs().sum().item())
            else:
                nonzero = bool(value)
            if nonzero:
                active_edges += 1
        if self._reporter is not None:
            self._reporter.report(
                "active_vertices",
                "Number of active vertices in subgraph",
                active_vertices,
            )
            self._reporter.report(
                "active_edges",
                "Number of active edges in subgraph",
                active_edges,
            )
        return {"g_v": g_v, "g_e": g_e}

    def compute_sample_loss(self, graph, gates):
        """Compute sample loss using gated sums over nodes and edges."""
        import torch  # local import

        g_v = gates.get("g_v", {})
        g_e = gates.get("g_e", {})
        zero = torch.tensor(0.0)
        loss = zero
        for nid, neuron in graph.neurons.items():
            gate = g_v.get(nid, zero)
            l_v = getattr(neuron, "last_local_loss", zero)
            if not hasattr(l_v, "shape"):
                l_v = torch.tensor(l_v)
            loss = loss + gate * l_v
        for sid, (_, _, synapse) in graph.synapses.items():
            gate = g_e.get(sid, zero)
            c_e = getattr(synapse, "c_e", zero)
            if not hasattr(c_e, "shape"):
                c_e = torch.tensor(c_e)
            loss = loss + gate * c_e
        if self._reporter is not None:
            self._reporter.report(
                "sample_loss",
                "Loss computed over active subgraph",
                loss,
            )
        return loss

    def compute_gradients(self, graph, gates, loss=None):
        """Compute gradients for weights with explicit regularisation terms.

        Parameters
        ----------
        graph : network.graph.Graph
            Graph containing neurons and synapses.
        gates : dict
            Dictionary with ``g_v`` and ``g_e`` entries as produced by
            :meth:`build_active_subgraph`.
        loss : torch.Tensor, optional
            Precomputed sample loss.  When ``None`` the loss is recomputed
            using :meth:`compute_sample_loss`.

        Returns
        -------
        dict
            Mapping with ``neurons`` and ``synapses`` entries that associate
            identifiers to their respective gradient tensors.
        """
        import torch  # local import

        if loss is None:
            loss = self.compute_sample_loss(graph, gates)

        weights = []
        meta = []
        g_v = gates.get("g_v", {})
        g_e = gates.get("g_e", {})
        zero = torch.tensor(0.0)

        for nid, neuron in graph.neurons.items():
            w = getattr(neuron, "weight", None)
            if w is None or not hasattr(w, "requires_grad"):
                continue
            weights.append(w)
            lam = getattr(neuron, "lambda_v", zero)
            gate = g_v.get(nid, zero)
            meta.append(("neurons", nid, neuron, w, gate, lam))

        for sid, (_, _, synapse) in graph.synapses.items():
            w = getattr(synapse, "weight", None)
            if w is None or not hasattr(w, "requires_grad"):
                continue
            weights.append(w)
            lam = getattr(synapse, "lambda_e", zero)
            gate = g_e.get(sid, zero)
            meta.append(("synapses", sid, synapse, w, gate, lam))

        grads = []
        if weights:
            grads = list(
                torch.autograd.grad(
                    loss, weights, allow_unused=True, retain_graph=True
                )
            )

        result = {"neurons": {}, "synapses": {}}
        for (kind, identifier, obj, w, gate, lam), grad in zip(meta, grads):
            if grad is None:
                grad = torch.zeros_like(w)
            lam_t = lam if hasattr(lam, "shape") else torch.tensor(lam, dtype=w.dtype, device=w.device)
            if kind == "neurons":
                l1 = gate * lam_t * torch.sign(w)
                l2 = gate * lam_t * w
                grad = grad + l1 + l2
                result["neurons"][identifier] = grad
                if self._reporter is not None:
                    self._reporter.report(
                        f"neuron_{id(obj)}_grad_norm",
                        "Gradient norm for neuron weight",
                        torch.norm(grad).detach(),
                    )
            else:
                l2 = gate * lam_t * w
                grad = grad + l2
                result["synapses"][identifier] = grad
                if self._reporter is not None:
                    self._reporter.report(
                        f"synapse_{id(obj)}_grad_norm",
                        "Gradient norm for synapse weight",
                        torch.norm(grad).detach(),
                    )
        return result
