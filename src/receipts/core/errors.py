"""The base of the core lane's model-call errors.

Every error a core client raises is a contracts.ModelCallError with its reason, and the cost of what it spent goes in
its `cost` once the call began. This base also makes each one cross a process pool whole: its message, reason, cost
and any attribute a subclass adds (an HTTP status, an attempt count) survive a pickle round trip.
"""

from __future__ import annotations

from typing import ClassVar

from receipts.contracts import ModelCallError, UncheckedReason


class CoreCallError(ModelCallError):
    """A model call that produced no response. `reason` is what the caller records as unchecked."""

    default_reason: ClassVar[UncheckedReason] = UncheckedReason.RUNTIME

    def __init__(self, message: str, *, reason: UncheckedReason | None = None) -> None:
        super().__init__(message, reason if reason is not None else self.default_reason)

    def __reduce__(self) -> tuple[object, ...]:
        # The message rebuilds the exception; every attribute (reason, cost, status, attempts) comes back as state.
        return (_rebuild, (type(self), str(self)), dict(vars(self)))


def _rebuild(cls: type[CoreCallError], message: str) -> CoreCallError:
    error = cls.__new__(cls)
    Exception.__init__(error, message)
    return error
