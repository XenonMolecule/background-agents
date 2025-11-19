# src/precursor/testing/telemetry.py
"""
Lightweight telemetry system for testing and observability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TelemetryEvent:
    """A single telemetry event with name and payload."""
    name: str
    payload: Dict[str, Any] = field(default_factory=dict)


class TelemetrySink(ABC):
    """Abstract interface for telemetry sinks."""
    
    @abstractmethod
    def emit(self, event_name: str, payload: Dict[str, Any] = None) -> None:
        """Emit a telemetry event."""
        pass


class MockTelemetrySink(TelemetrySink):
    """Mock telemetry sink that captures events for testing."""
    
    def __init__(self) -> None:
        self.events: List[TelemetryEvent] = []
    
    def emit(self, event_name: str, payload: Dict[str, Any] = None) -> None:
        """Capture a telemetry event."""
        event = TelemetryEvent(name=event_name, payload=payload or {})
        self.events.append(event)
        logger.debug("telemetry: %s %s", event_name, payload)
    
    def get_events(self, event_name: str = None) -> List[TelemetryEvent]:
        """Get all events, optionally filtered by name."""
        if event_name is None:
            return self.events.copy()
        return [e for e in self.events if e.name == event_name]
    
    def clear(self) -> None:
        """Clear all captured events."""
        self.events.clear()


class LoggingTelemetrySink(TelemetrySink):
    """Telemetry sink that logs events."""
    
    def emit(self, event_name: str, payload: Dict[str, Any] = None) -> None:
        """Log a telemetry event."""
        logger.info("telemetry: %s %s", event_name, payload or {})


class NotificationSink(ABC):
    """Abstract interface for notification sinks."""
    
    @abstractmethod
    def send_notification(self, project: str, message: str, **kwargs) -> None:
        """Send a notification."""
        pass


class MockNotificationSink(NotificationSink):
    """Mock notification sink that captures notifications for testing."""
    
    def __init__(self) -> None:
        self.notifications: List[Dict[str, Any]] = []
    
    def send_notification(self, project: str, message: str, **kwargs) -> None:
        """Capture a notification."""
        notification = {
            "project": project,
            "message": message,
            **kwargs
        }
        self.notifications.append(notification)
        logger.debug("notification: %s", notification)
    
    def get_notifications(self, project: str = None) -> List[Dict[str, Any]]:
        """Get all notifications, optionally filtered by project."""
        if project is None:
            return self.notifications.copy()
        return [n for n in self.notifications if n.get("project") == project]
    
    def clear(self) -> None:
        """Clear all captured notifications."""
        self.notifications.clear()


# Global default sinks (can be overridden for testing)
_default_telemetry_sink: TelemetrySink = LoggingTelemetrySink()
_default_notification_sink: NotificationSink = None


def get_telemetry_sink() -> TelemetrySink:
    """Get the current telemetry sink."""
    return _default_telemetry_sink


def set_telemetry_sink(sink: TelemetrySink) -> None:
    """Set the global telemetry sink."""
    global _default_telemetry_sink
    _default_telemetry_sink = sink


def get_notification_sink() -> NotificationSink:
    """Get the current notification sink."""
    return _default_notification_sink


def set_notification_sink(sink: NotificationSink) -> None:
    """Set the global notification sink."""
    global _default_notification_sink
    _default_notification_sink = sink


def emit_telemetry(event_name: str, payload: Dict[str, Any] = None) -> None:
    """Emit a telemetry event using the current sink."""
    sink = get_telemetry_sink()
    if sink:
        sink.emit(event_name, payload)


def send_notification(project: str, message: str, **kwargs) -> None:
    """Send a notification using the current sink."""
    sink = get_notification_sink()
    if sink:
        sink.send_notification(project, message, **kwargs)