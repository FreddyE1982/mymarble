"""Computation of global network complexity metrics."""


class ComplexityCalculator:
    """Compute the global complexity of a neural graph.

    The complexity is calculated according to Eq. (3.1)::

        C = |V| + sum_i(A_i * |N_i| + B_i * |E_i| +
                       Γ * ||w_i||_1 + Λ * ||w_i||_2) + sum_k |Ω_k|

    Parameters
    ----------
    A : sequence
        Sequence of tensors weighting neuron counts for each component.
    B : sequence
        Sequence of tensors weighting edge counts for each component.
    gamma : tensor
        Global weight for the L1 norms of ``w_i``.
    lambda_ : tensor
        Global weight for the L2 norms of ``w_i``.
    weights : sequence
        Sequence of tensors ``w_i`` whose norms contribute to the complexity.
    attribute_sets : sequence, optional
        Iterable of sets ``Ω_k`` representing additional attributes.
    reporter : object, optional
        Object providing a ``report`` method for metric recording.
    """

    def __init__(self, A, B, gamma, lambda_, weights, attribute_sets=None, reporter=None):
        self._A = A
        self._B = B
        self._gamma = gamma
        self._lambda = lambda_
        self._weights = weights
        self._attribute_sets = [] if attribute_sets is None else list(attribute_sets)
        self._reporter = reporter

    def compute(self, graph_state):
        """Return the global complexity of ``graph_state``.

        The ``graph_state`` is expected to provide ``neurons`` and ``synapses``
        mappings, similar to :class:`network.graph.Graph`.
        """
        neurons = getattr(graph_state, "neurons", {})
        synapses = getattr(graph_state, "synapses", {})
        num_neurons = len(neurons)
        num_edges = len(synapses)
        zero_like = self._gamma * 0 if hasattr(self._gamma, "__mul__") else 0
        complexity = zero_like + num_neurons
        if self._reporter is not None:
            self._reporter.report("num_neurons", "Total number of neurons", num_neurons)
            self._reporter.report("num_edges", "Total number of synapses", num_edges)
        edge_count_t = zero_like + num_edges
        total_neuron_penalty = zero_like
        total_edge_penalty = zero_like
        total_l1 = zero_like
        total_l2 = zero_like
        for idx, weight in enumerate(self._weights):
            A_i = self._A[idx]
            B_i = self._B[idx]
            n_penalty = A_i * (zero_like + num_neurons)
            e_penalty = B_i * edge_count_t
            l1 = weight.abs().sum() if hasattr(weight, "abs") else weight
            if hasattr(weight, "pow"):
                l2 = weight.pow(2).sum().sqrt()
            else:
                from math import sqrt
                l2 = sqrt(weight * weight)
            total_neuron_penalty = total_neuron_penalty + n_penalty
            total_edge_penalty = total_edge_penalty + e_penalty
            total_l1 = total_l1 + l1
            total_l2 = total_l2 + l2
            if self._reporter is not None:
                self._reporter.report(
                    f"neuron_penalty_{idx}",
                    "Weighted neuron count for component",
                    n_penalty,
                )
                self._reporter.report(
                    f"edge_penalty_{idx}",
                    "Weighted edge count for component",
                    e_penalty,
                )
                self._reporter.report(
                    f"l1_norm_{idx}",
                    "L1 norm of weight tensor",
                    l1,
                )
                self._reporter.report(
                    f"l2_norm_{idx}",
                    "L2 norm of weight tensor",
                    l2,
                )
        attr_total = zero_like
        for i, attrs in enumerate(self._attribute_sets):
            size_t = zero_like + len(attrs)
            attr_total = attr_total + size_t
            complexity = complexity + size_t
            if self._reporter is not None:
                self._reporter.report(
                    f"attr_set_{i}",
                    "Size of attribute set",
                    size_t,
                )
        complexity = (
            complexity
            + total_neuron_penalty
            + total_edge_penalty
            + self._gamma * total_l1
            + self._lambda * total_l2
        )
        if self._reporter is not None:
            self._reporter.report("neuron_penalty", "Total neuron penalty", total_neuron_penalty)
            self._reporter.report("edge_penalty", "Total edge penalty", total_edge_penalty)
            self._reporter.report("l1_norm", "Aggregate L1 norm", total_l1)
            self._reporter.report("l2_norm", "Aggregate L2 norm", total_l2)
            self._reporter.report(
                "attribute_size",
                "Total size over attribute sets",
                attr_total,
            )
        return complexity
