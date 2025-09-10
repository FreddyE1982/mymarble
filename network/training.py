"""Orchestrator for one training iteration on a neural graph.

This module defines :class:`TrainingKernel` which coordinates the forward pass,
backward pass and telemetry updates for a single training iteration as outlined
in the repository documentation.  The implementation purposefully avoids any
module level imports in order to comply with repository constraints.
"""


class TrainingKernel:
    """Execute a full training iteration over a graph.

    Parameters
    ----------
    graph : object
        Instance of :class:`network.graph.Graph` representing the current
        neural topology.
    backprop : object
        Instance of :class:`network.backprop.Backpropagator` responsible for
        gradient computations and parameter updates.
    telemetry : object
        Instance of :class:`network.telemetry.TelemetryUpdater` used to refresh
        telemetry tensors after parameter updates.
    loss_tracker : object, optional
        Optional :class:`network.learning.LossTracker` to maintain rolling loss
        statistics for neurons.
    reporter : object, optional
        Reporter used for metric emission.
    """

    def __init__(self, graph, backprop, telemetry, loss_tracker=None, reporter=None):
        self._graph = graph
        self._backprop = backprop
        self._telemetry = telemetry
        self._loss_tracker = loss_tracker
        self._reporter = reporter

    def run_iteration(
        self,
        sample=None,
        routing_mode="hard",
        cost_params=None,
        sample_params=None,
        lr_v=None,
        lr_e=None,
        optimizer=None,
        evolution_instructions=None,
    ):
        """Perform one full training iteration.

        The procedure follows the eight step kernel described in the project
        documentation.  Existing components such as :meth:`graph.forward` and
        :class:`Backpropagator` methods are orchestrated without reimplementing
        their internal utilities.

        Parameters
        ----------
        sample : object, optional
            Training sample forwarded through :meth:`graph.forward`.
        routing_mode : {"hard", "soft"}, optional
            Determines whether routing is restricted to the chosen path or
            weighted across the entire graph.
        cost_params : dict, optional
            Parameters forwarded to :meth:`graph.forward` for cost calculation.
        sample_params : dict, optional
            Sampling parameters forwarded to :meth:`graph.forward`.
        lr_v : object, optional
            Learning rate for neuron weights when manual updates are used.
        lr_e : object, optional
            Learning rate for synapse weights when manual updates are used.
        optimizer : object, optional
            ``torch.optim`` compatible optimizer.  When supplied, it is used for
            the parameter update step.
        evolution_instructions : dict, optional
            Optional instructions passed to :meth:`graph.forward` to invoke the
            evolutionary operator.

        Returns
        -------
        dict
            Combination of the results returned by :meth:`graph.forward` along
            with ``loss`` and ``grads`` entries produced during the backward
            pass.
        """

        method = "soft" if routing_mode == "soft" else "exact"
        forward_kwargs = {"method": method}
        if sample is not None:
            forward_kwargs["sample"] = sample
        if cost_params is not None:
            forward_kwargs["cost_params"] = cost_params
        if sample_params is not None:
            forward_kwargs["sample_params"] = sample_params
        if evolution_instructions is not None:
            forward_kwargs["evolution_instructions"] = evolution_instructions

        forward_result = self._graph.forward(**forward_kwargs)
        path = forward_result.get("path")

        gates = self._backprop.build_active_subgraph(self._graph, path, routing_mode)
        loss = self._backprop.compute_sample_loss(self._graph, gates)
        active = {"graph": self._graph, "g_v": gates["g_v"], "g_e": gates["g_e"], "loss": loss}
        grads = self._backprop.compute_gradients(active)
        active["grads"] = grads
        self._backprop.apply_updates(active, lr_v, lr_e, optimizer)

        if path:
            neuron_path = [path[i] for i in range(0, len(path), 2)]
            losses = [getattr(n, "last_local_loss", 0) for n in neuron_path]
            self._telemetry.update(neuron_path, losses, gates.get("g_v"))
            if self._loss_tracker is not None:
                for neuron, loss in zip(neuron_path, losses):
                    self._loss_tracker.update_loss(neuron, [loss])

        if self._reporter is not None:
            count = self._reporter.report("training_iterations") or 0
            self._reporter.report(
                "training_iterations",
                "Number of completed training iterations",
                count + 1,
            )

        forward_result["loss"] = loss
        forward_result["grads"] = grads
        return forward_result
