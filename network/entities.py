"""Core neural network entities.

This module defines the :class:`Neuron` and :class:`Synapse` classes used by
:mod:`network.graph`.  Both classes are intentionally import free and expect
any tensor like objects to be provided by the caller.  Scalar fields default to
zero valued 0-D tensors (represented as ``0`` when no tensor factory is
injected).
"""


class Neuron:
    """Represents a single neuron in a neural graph.

    Parameters can be provided during construction.  If omitted, the neuron
    initializes all tensor attributes with a zero valued 0-D tensor.  The
    concrete tensor object can be supplied through the ``zero`` argument.
    """

    def __init__(
        self,
        reset_state=None,
        payload=None,
        activation=None,
        path_preactivation=None,
        last_local_loss=None,
        next_min_loss=None,
        lambda_v=None,
        zero=None,
    ):
        if zero is None:
            zero = 0
        self._zero = zero
        self.reset_state = zero if reset_state is None else reset_state
        self.payload = zero if payload is None else payload
        self.activation = zero if activation is None else activation
        self.path_preactivation = zero if path_preactivation is None else path_preactivation
        self.last_local_loss = zero if last_local_loss is None else last_local_loss
        self.next_min_loss = zero if next_min_loss is None else next_min_loss
        self.lambda_v = zero if lambda_v is None else lambda_v

    def reset(self):
        """Reset all dynamic tensors to the configured zero value."""
        zero = self._zero
        self.reset_state = zero
        self.payload = zero
        self.activation = zero
        self.path_preactivation = zero
        self.last_local_loss = zero
        self.next_min_loss = zero
        self.lambda_v = zero

    def update_reset_state(self, tensor):
        self.reset_state = tensor

    def update_payload(self, tensor):
        self.payload = tensor

    def update_activation(self, tensor):
        self.activation = tensor

    def update_path_preactivation(self, tensor):
        self.path_preactivation = tensor

    def record_local_loss(self, tensor):
        self.last_local_loss = tensor

    def update_next_min_loss(self, tensor):
        self.next_min_loss = tensor

    def update_latency(self, tensor):
        self.lambda_v = tensor

    def to_dict(self):
        """Return a dictionary snapshot of the neuron state."""
        return {
            "reset_state": self.reset_state,
            "payload": self.payload,
            "activation": self.activation,
            "path_preactivation": self.path_preactivation,
            "last_local_loss": self.last_local_loss,
            "next_min_loss": self.next_min_loss,
            "lambda_v": self.lambda_v,
        }


class Synapse:
    """Represents a directed connection between two neurons."""

    def __init__(
        self,
        preactivation=None,
        activation=None,
        lambda_e=None,
        c_e=None,
        zero=None,
    ):
        if zero is None:
            zero = 0
        self._zero = zero
        self.preactivation = zero if preactivation is None else preactivation
        self.activation = zero if activation is None else activation
        self.lambda_e = zero if lambda_e is None else lambda_e
        self.c_e = zero if c_e is None else c_e

    def reset(self):
        zero = self._zero
        self.preactivation = zero
        self.activation = zero
        self.lambda_e = zero
        self.c_e = zero

    def update_preactivation(self, tensor):
        self.preactivation = tensor

    def update_activation(self, tensor):
        self.activation = tensor

    def update_latency(self, tensor):
        self.lambda_e = tensor

    def update_cost(self, tensor):
        self.c_e = tensor

    def to_dict(self):
        return {
            "preactivation": self.preactivation,
            "activation": self.activation,
            "lambda_e": self.lambda_e,
            "c_e": self.c_e,
        }
