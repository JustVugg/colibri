"""Small Qt bridges that keep launcher work off the GUI thread."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from .domain import ProcessEvent


class WorkerSignals(QObject):
    result = Signal(str, object)
    failed = Signal(str, str)


class FunctionWorker(QRunnable):
    """Run one callable and return its value with an opaque request token."""

    def __init__(self, token: str, function: Callable[..., Any], *args: Any):
        super().__init__()
        self.token = token
        self.function = function
        self.args = args
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.function(*self.args)
        except Exception as exc:  # surfaced as friendly text on the GUI thread
            message = str(exc).strip() or type(exc).__name__
            self.signals.failed.emit(self.token, message)
        else:
            self.signals.result.emit(self.token, result)


class ProcessEventBridge(QObject):
    """Marshal supervisor callbacks from monitor threads to Qt's GUI thread."""

    received = Signal(object)

    @Slot(object)
    def publish(self, event: ProcessEvent) -> None:
        self.received.emit(event)
