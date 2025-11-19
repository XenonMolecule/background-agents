# src/precursor/testing/sinks.py
"""Mock sinks for testing telemetry and notifications."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import dataclass


@dataclass
class TelemetryEvent:
    """A telemetry event with name and payload."""
    name: str
    payload: Dict[str, Any]


@dataclass
class NotificationEvent:
    """A notification event with project and details."""
    project: str
    notification_type: str
    message: str
    sent: bool  # True if sent, False if skipped


class TelemetrySink(ABC):
    """Abstract interface for telemetry sinks."""
    
    @abstractmethod
    def emit(self, event_name: str, payload: Dict[str, Any]) -> None:
        """Emit a telemetry event."""
        pass


class NotificationSink(ABC):
    """Abstract interface for notification sinks."""
    
    @abstractmethod
    def send_notification(self, project: str, notification_type: str, message: str) -> None:
        """Send a notification."""
        pass
    
    @abstractmethod
    def skip_notification(self, project: str, reason: str) -> None:
        """Skip a notification with a reason."""
        pass


class MockTelemetrySink(TelemetrySink):
    """Mock telemetry sink that captures events for testing."""
    
    def __init__(self) -> None:
        self.events: List[TelemetryEvent] = []
    
    def emit(self, event_name: str, payload: Dict[str, Any]) -> None:
        """Emit a telemetry event."""
        self.events.append(TelemetryEvent(name=event_name, payload=payload))
    
    def get_events(self, event_name: str = None) -> List[TelemetryEvent]:
        """Get all events, optionally filtered by name."""
        if event_name is None:
            return self.events[:]
        return [e for e in self.events if e.name == event_name]
    
    def clear(self) -> None:
        """Clear all captured events."""
        self.events.clear()


class MockNotificationSink(NotificationSink):
    """Mock notification sink that captures notifications for testing."""
    
    def __init__(self) -> None:
        self.notifications: List[NotificationEvent] = []
    
    def send_notification(self, project: str, notification_type: str, message: str) -> None:
        """Send a notification."""
        self.notifications.append(NotificationEvent(
            project=project,
            notification_type=notification_type,
            message=message,
            sent=True
        ))
    
    def skip_notification(self, project: str, reason: str) -> None:
        """Skip a notification with a reason."""
        self.notifications.append(NotificationEvent(
            project=project,
            notification_type="skipped",
            message=f"Skipped: {reason}",
            sent=False
        ))
    
    def get_notifications(self, project: str = None, sent: bool = None) -> List[NotificationEvent]:
        """Get all notifications, optionally filtered by project and/or sent status."""
        notifications = self.notifications[:]
        if project is not None:
            notifications = [n for n in notifications if n.project == project]
        if sent is not None:
            notifications = [n for n in notifications if n.sent == sent]
        return notifications
    
    def clear(self) -> None:
        """Clear all captured notifications."""
        self.notifications.clear()