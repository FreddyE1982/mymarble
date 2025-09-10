"""Entry sampling for neural graphs.

This module implements the :class:`EntrySampler` which computes a softmax
distribution over neurons and allows drawing entry neurons according to that
probability.  All tensor operations are fully injected through the constructor
and no module level imports are performed.
"""


class EntrySampler:
    """Sample neurons for graph entry based on feature scores.

    The sampler evaluates each neuron ``v`` using its feature tensor
    ``phi_v`` and computes probabilities according to Eq. (1.1)::

        R_in(v) = exp(phi_v / T) / sum_u exp(phi_u / T)

    where ``T`` is the sampling temperature.  Probabilities are computed in a
    numerically stable fashion using ``logsumexp`` and sampling is performed via
    ``torch.multinomial`` from the injected tensor library.

    Parameters
    ----------
    temperature : object
        Temperature ``T`` used during sampling.  Can be a tensor or scalar.
    torch : module
        Tensor library providing ``stack``, ``logsumexp``, ``exp`` and
        ``multinomial`` functions.
    reporter : object, optional
        Object providing a ``report`` method compatible with
        :class:`main.Reporter`.  Every sampled neuron identifier is reported via
        ``report("entry_sample", v.id)``.
    zero : object, optional
        Tensor used when a neuron lacks ``phi_v``.  Defaults to ``0``.
    """

    def __init__(self, temperature, torch, reporter=None, zero=0):
        self._temperature = temperature
        self._torch = torch
        self._reporter = reporter
        self._zero = zero
        self._prob_tensor = None
        self._neurons = []
        self._ids = []

    def compute_probabilities(self, graph):
        """Compute entry probabilities for all neurons in ``graph``.

        Parameters
        ----------
        graph : :class:`network.graph.Graph`
            Graph providing the neurons over which probabilities are computed.

        Returns
        -------
        dict
            Mapping of ``{neuron_id: probability}`` tensors.
        """
        ids = []
        neurons = []
        features = []
        for nid, neuron in graph.neurons.items():
            setattr(neuron, "id", nid)
            phi = getattr(neuron, "phi_v", self._zero)
            features.append(phi)
            ids.append(nid)
            neurons.append(neuron)
        if not features:
            self._prob_tensor = self._torch.zeros(0) if hasattr(self._torch, "zeros") else []
            self._neurons = []
            self._ids = []
            return {}
        logits = self._torch.stack(features) / self._temperature
        log_probs = logits - self._torch.logsumexp(logits, dim=0)
        probs = self._torch.exp(log_probs)
        self._prob_tensor = probs
        self._neurons = neurons
        self._ids = ids
        return {nid: prob for nid, prob in zip(ids, probs)}

    def sample_entry(self):
        """Draw and return a neuron according to ``R_in(v)``.

        Returns
        -------
        :class:`network.entities.Neuron`
            The sampled neuron.
        """
        if not self._neurons or self._prob_tensor is None:
            raise RuntimeError("Probabilities have not been computed")
        idx_t = self._torch.multinomial(self._prob_tensor, 1)
        idx = int(idx_t.item()) if hasattr(idx_t, "item") else int(idx_t[0])
        neuron = self._neurons[idx]
        if self._reporter is not None:
            self._reporter.report("entry_sample", neuron.id)
        return neuron
