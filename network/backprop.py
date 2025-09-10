"""Backward pass utilities with active subgraph gating."""


class Backpropagator:
    """Perform backward computations over an active subgraph.

    The class builds gate masks for vertices and edges according to the
    provided routing mode and evaluates the sample loss using these masks.
    All heavy imports are executed lazily inside the respective methods to
    comply with repository constraints.
    """

    def __init__(self, reporter=None, optimizer=None):
        self._reporter = reporter
        self._optimizer = optimizer

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

    def compute_gradients(self, active_subgraph):
        """Compute gradients for all weights in the active subgraph.

        Parameters
        ----------
        active_subgraph : dict
            Dictionary containing at least ``graph`` alongside ``g_v`` and
            ``g_e`` entries describing the active vertices and edges.  A
            precomputed ``loss`` term can optionally be supplied.

        Returns
        -------
        dict
            Mapping with ``neurons`` and ``synapses`` entries where each
            identifier is associated with its gradient tensor.
        """
        import torch  # local import

        graph = active_subgraph.get("graph")
        g_v = active_subgraph.get("g_v", {})
        g_e = active_subgraph.get("g_e", {})
        loss = active_subgraph.get("loss")

        if graph is None:
            return {"neurons": {}, "synapses": {}}

        if loss is None:
            loss = self.compute_sample_loss(graph, active_subgraph)

        zero = torch.tensor(0.0)
        result = {"neurons": {}, "synapses": {}}

        for nid, neuron in graph.neurons.items():
            gate = g_v.get(nid, zero)
            value = gate.detach() if hasattr(gate, "detach") else gate
            if hasattr(value, "abs"):
                active = bool(value.abs().sum().item())
            else:
                active = bool(value)
            if not active:
                continue

            w_v = getattr(neuron, "weight", None)
            if w_v is None or not getattr(w_v, "requires_grad", False):
                continue
            l_v = getattr(neuron, "last_local_loss", zero)
            lam = getattr(neuron, "lambda_v", zero)

            dC_dy = torch.autograd.grad(
                loss,
                l_v,
                grad_outputs=torch.ones_like(loss),
                retain_graph=True,
                allow_unused=True,
            )[0]
            if dC_dy is None:
                dC_dy = torch.zeros_like(l_v)
            dC_dw = torch.autograd.grad(
                l_v,
                w_v,
                grad_outputs=dC_dy,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if dC_dw is None:
                dC_dw = torch.zeros_like(w_v)
            first_term = dC_dw

            norm1 = w_v.abs().sum()
            grad_norm1 = torch.autograd.grad(
                norm1, w_v, retain_graph=True, allow_unused=True
            )[0]
            if grad_norm1 is None:
                grad_norm1 = torch.zeros_like(w_v)
            l1_term = lam * grad_norm1

            norm2 = 0.5 * w_v.pow(2).sum()
            grad_norm2 = torch.autograd.grad(
                norm2, w_v, retain_graph=True, allow_unused=True
            )[0]
            if grad_norm2 is None:
                grad_norm2 = torch.zeros_like(w_v)
            l2_term = lam * grad_norm2

            grad = first_term + l1_term + l2_term
            result["neurons"][nid] = grad
            if self._reporter is not None:
                self._reporter.report(
                    f"neuron_{id(neuron)}_grad_norm",
                    "Gradient norm for neuron weight",
                    torch.norm(grad).detach(),
                )

        for sid, (_, _, synapse) in graph.synapses.items():
            gate = g_e.get(sid, zero)
            value = gate.detach() if hasattr(gate, "detach") else gate
            if hasattr(value, "abs"):
                active = bool(value.abs().sum().item())
            else:
                active = bool(value)
            if not active:
                continue

            w_e = getattr(synapse, "weight", None)
            if w_e is None or not getattr(w_e, "requires_grad", False):
                continue
            c_e = getattr(synapse, "c_e", zero)
            lam = getattr(synapse, "lambda_e", zero)

            dC_dy = torch.autograd.grad(
                loss,
                c_e,
                grad_outputs=torch.ones_like(loss),
                retain_graph=True,
                allow_unused=True,
            )[0]
            if dC_dy is None:
                dC_dy = torch.zeros_like(c_e)
            dC_dw = torch.autograd.grad(
                c_e,
                w_e,
                grad_outputs=dC_dy,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if dC_dw is None:
                dC_dw = torch.zeros_like(w_e)
            first_term = dC_dw

            norm2 = 0.5 * w_e.pow(2).sum()
            grad_norm2 = torch.autograd.grad(
                norm2, w_e, retain_graph=True, allow_unused=True
            )[0]
            if grad_norm2 is None:
                grad_norm2 = torch.zeros_like(w_e)
            l2_term = lam * grad_norm2

            grad = first_term + l2_term
            result["synapses"][sid] = grad
            if self._reporter is not None:
                self._reporter.report(
                    f"synapse_{id(synapse)}_grad_norm",
                    "Gradient norm for synapse weight",
                    torch.norm(grad).detach(),
                )

        return result

    def apply_updates(self, active_subgraph, lr_v, lr_e, optimizer=None):
        """Apply gradient updates to weights in the active subgraph.

        Parameters
        ----------
        active_subgraph : dict
            Dictionary containing ``graph`` and ``grads`` mappings.
        lr_v : object
            Learning rate for neuron weights.
        lr_e : object
            Learning rate for synapse weights.
        optimizer : object, optional
            Optimizer instance from :mod:`torch.optim`.  When provided, it is
            used for the update step; otherwise manual SGD is executed.
        """
        import torch  # local import

        graph = active_subgraph.get("graph")
        grads = active_subgraph.get("grads", {})
        if graph is None or not grads:
            return

        opt = optimizer if optimizer is not None else self._optimizer
        neuron_grads = grads.get("neurons", {})
        synapse_grads = grads.get("synapses", {})

        if opt is not None:
            for nid, grad in neuron_grads.items():
                neuron = graph.neurons.get(nid)
                w = getattr(neuron, "weight", None)
                if w is None:
                    continue
                w.grad = grad
            for sid, grad in synapse_grads.items():
                synapse = graph.synapses.get(sid)[2]
                w = getattr(synapse, "weight", None)
                if w is None:
                    continue
                w.grad = grad
            opt.step()
            opt.zero_grad()
            if self._reporter is not None:
                for idx, group in enumerate(opt.param_groups):
                    lr = group.get("lr")
                    self._reporter.report(
                        f"optimizer_lr_{idx}",
                        f"Learning rate for optimizer param group {idx}",
                        lr,
                    )
                for nid, neuron in graph.neurons.items():
                    w = getattr(neuron, "weight", None)
                    self._reporter.report(
                        f"neuron_{id(neuron)}_weight_norm_post",
                        "Post-update weight norm for neuron",
                        torch.norm(w).detach(),
                    )
                for sid, (_, _, synapse) in graph.synapses.items():
                    w = getattr(synapse, "weight", None)
                    self._reporter.report(
                        f"synapse_{id(synapse)}_weight_norm_post",
                        "Post-update weight norm for synapse",
                        torch.norm(w).detach(),
                    )
            return

        if self._reporter is not None:
            self._reporter.report(
                "lr_v", "Learning rate for neuron weights", lr_v
            )
            self._reporter.report(
                "lr_e", "Learning rate for synapse weights", lr_e
            )

        for nid, grad in neuron_grads.items():
            neuron = graph.neurons.get(nid)
            w = getattr(neuron, "weight", None)
            if w is None:
                continue
            new_w = (w - lr_v * grad).detach().requires_grad_(True)
            neuron.update_weight(new_w, reporter=self._reporter)
            if self._reporter is not None:
                self._reporter.report(
                    f"neuron_{id(neuron)}_weight_norm_post",
                    "Post-update weight norm for neuron",
                    torch.norm(new_w).detach(),
                )

        for sid, grad in synapse_grads.items():
            synapse = graph.synapses.get(sid)[2]
            w = getattr(synapse, "weight", None)
            if w is None:
                continue
            new_w = (w - lr_e * grad).detach().requires_grad_(True)
            synapse.update_weight(new_w, reporter=self._reporter)
            if self._reporter is not None:
                self._reporter.report(
                    f"synapse_{id(synapse)}_weight_norm_post",
                    "Post-update weight norm for synapse",
                    torch.norm(new_w).detach(),
                )
