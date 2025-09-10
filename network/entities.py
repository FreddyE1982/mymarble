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
        cumulative_loss=None,
        step_loss=None,
        timestamp=None,
        measured_time=None,
        loss_decrease_speed=None,
        prev_cumulative_loss=None,
        gate=None,
        activation_threshold=None,
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
        self.cumulative_loss = zero if cumulative_loss is None else cumulative_loss
        self.step_loss = zero if step_loss is None else step_loss
        self.timestamp = zero if timestamp is None else timestamp
        self.measured_time = zero if measured_time is None else measured_time
        self.loss_decrease_speed = zero if loss_decrease_speed is None else loss_decrease_speed
        self.prev_cumulative_loss = (
            zero if prev_cumulative_loss is None else prev_cumulative_loss
        )
        self.gate = zero if gate is None else gate
        self.activation_threshold = (
            zero if activation_threshold is None else activation_threshold
        )

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
        self.cumulative_loss = zero
        self.step_loss = zero
        self.timestamp = zero
        self.measured_time = zero
        self.loss_decrease_speed = zero
        self.prev_cumulative_loss = zero
        self.gate = zero
        self.activation_threshold = zero

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

    def update_gate(self, tensor):
        self.gate = tensor

    def update_activation_threshold(self, tensor):
        self.activation_threshold = tensor

    def update_cumulative_loss(self, loss_tensor, reporter=None):
        """Add ``loss_tensor`` to the neuron's cumulative loss.

        Parameters
        ----------
        loss_tensor : object
            Loss value to record for the current step.
        reporter : object, optional
            Instance providing a ``report`` method. Metrics are written to this
            reporter when supplied.
        """
        detached_loss = (
            loss_tensor.detach() if hasattr(loss_tensor, "detach") else loss_tensor
        )
        current = (
            self.cumulative_loss.detach()
            if hasattr(self.cumulative_loss, "detach")
            else self.cumulative_loss
        )
        self.step_loss = detached_loss
        self.cumulative_loss = current + detached_loss
        if reporter is not None:
            reporter.report(
                f"neuron_{id(self)}_step_loss",
                "Loss recorded for neuron at current step",
                self.step_loss,
            )
            reporter.report(
                f"neuron_{id(self)}_cumulative_loss",
                "Cumulative loss recorded for neuron",
                self.cumulative_loss,
            )

    def record_activation(self, time_tensor, measured_time_tensor, reporter=None):
        """Record an activation event and update timing metrics.

        Parameters
        ----------
        time_tensor : object
            Expected timestamp of the activation :math:`\tau_n`.
        measured_time_tensor : object
            Observed time :math:`s_n` at which the activation was measured.
        reporter : object, optional
            Instance providing a ``report`` method. Metrics are written to this
            reporter when supplied.

        The loss decrease speed :math:`r_n` is calculated according to
        Eq. (0.1) using the previously recorded timestamp as
        :math:`t_{k-1}`.  Metrics are reported via the optional ``reporter``.
        """
        prev_timestamp = self.timestamp
        prev_cumulative = self.prev_cumulative_loss
        time_delta = measured_time_tensor - prev_timestamp
        time_delta = time_delta.detach() if hasattr(time_delta, "detach") else time_delta
        loss_delta = self.cumulative_loss - prev_cumulative
        loss_delta = loss_delta.detach() if hasattr(loss_delta, "detach") else loss_delta
        if hasattr(time_delta, "item"):
            zero_time = time_delta.item() == 0
        else:
            zero_time = time_delta == 0
        speed = self._zero if zero_time else loss_delta / time_delta
        self.loss_decrease_speed = speed
        self.timestamp = time_tensor
        self.measured_time = measured_time_tensor
        self.prev_cumulative_loss = self.cumulative_loss
        if reporter is not None:
            reporter.report(
                f"neuron_{id(self)}_step_loss",
                "Loss recorded for neuron at current step",
                self.step_loss,
            )
            reporter.report(
                f"neuron_{id(self)}_cumulative_loss",
                "Cumulative loss recorded for neuron",
                self.cumulative_loss,
            )
            reporter.report(
                f"neuron_{id(self)}_loss_decrease_speed",
                "Loss decrease speed for neuron",
                self.loss_decrease_speed,
            )

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
            "cumulative_loss": self.cumulative_loss,
            "step_loss": self.step_loss,
            "timestamp": self.timestamp,
            "measured_time": self.measured_time,
            "loss_decrease_speed": self.loss_decrease_speed,
            "gate": self.gate,
            "activation_threshold": self.activation_threshold,
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
