# src/precursor/testing/harness.py
"""
Deterministic test harness for project transition and notification testing.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

from precursor.context.events import ContextEvent
from precursor.context.project_history import ProjectHistory
from precursor.managers.state_manager import StateManager
from precursor.managers.ui_manager import UIManager
from precursor.managers.agent_manager import AgentManager
from precursor.observers.project_transition import ProjectActivityObserver
from precursor.testing.telemetry import (
    MockTelemetrySink, 
    MockNotificationSink, 
    set_telemetry_sink, 
    set_notification_sink
)


@dataclass
class TestEvent:
    """A simplified test event for deterministic testing."""
    timestamp: datetime
    project: str
    context_update: str
    user_name: str = "Test User"
    screenshot_path: Optional[str] = None
    
    # Prevent pytest from collecting this as a test class
    __test__ = False


class DeterministicTestHarness:
    """
    Test harness that provides deterministic, controllable event processing
    for testing project transitions and notifications.
    """
    
    def __init__(self, *, enable_scratchpad: bool = False):
        """
        Initialize the test harness.
        
        Args:
            enable_scratchpad: If True, use real scratchpad operations (slower).
                              If False, use mock managers for faster testing.
        """
        self.enable_scratchpad = enable_scratchpad
        
        # Set up telemetry and notification sinks
        self.telemetry_sink = MockTelemetrySink()
        self.notification_sink = MockNotificationSink()
        set_telemetry_sink(self.telemetry_sink)
        set_notification_sink(self.notification_sink)
        
        # Initialize core components
        self.history = ProjectHistory()
        
        if enable_scratchpad:
            # Use real managers (slower but more realistic)
            self.state_manager = StateManager(history=self.history)
            self.agent_manager = AgentManager(deploy_enabled=False)
            self.ui_manager = UIManager()
        else:
            # Use mock managers for faster testing
            self.state_manager = MockStateManager(history=self.history)
            self.agent_manager = MockAgentManager()
            self.ui_manager = MockUIManager()
        
        # Set up observers
        self.transition_observer = ProjectActivityObserver(
            history=self.history,
            agent_manager=self.agent_manager,
            mode="departure",
            window_size=20,
            min_entries_previous_segment=3,
            time_threshold=timedelta(minutes=10),
        )
        
        self.arrival_observer = ProjectActivityObserver(
            history=self.history,
            agent_manager=self.ui_manager,
            mode="arrival",
            window_size=20,
            min_entries_current_segment=1,
            time_threshold=timedelta(minutes=15),
        )
        
        # Track processed events
        self.processed_events: List[ContextEvent] = []
    
    def process_events(self, events: List[TestEvent]) -> None:
        """
        Process a batch of events deterministically.
        
        This simulates the batch processing behavior seen in the main pipeline
        where multiple events are processed and then observers are triggered.
        """
        # Convert test events to context events and process them
        for test_event in events:
            context_event = self._test_event_to_context_event(test_event)
            
            # Process the event through the state manager
            result = self.state_manager.process_event(context_event)
            self.processed_events.append(context_event)
            
            # Emit telemetry for queued observation
            from precursor.testing.telemetry import emit_telemetry
            emit_telemetry("observation_queued", {
                "project": result.get("project", "unknown"),
                "queue_size": len(self.processed_events)
            })
        
        # After processing all events, trigger observers (simulates batch processing)
        self.transition_observer.handle_processed()
        self.arrival_observer.handle_processed()
        
        # Emit batch processed telemetry
        from precursor.testing.telemetry import emit_telemetry
        emit_telemetry("batch_processed", {
            "count": len(events)
        })
    
    def _test_event_to_context_event(self, test_event: TestEvent) -> ContextEvent:
        """Convert a TestEvent to a ContextEvent."""
        return ContextEvent(
            timestamp=test_event.timestamp,
            context_update=test_event.context_update,
            screenshot=None,  # No screenshots in basic tests
            user_name=test_event.user_name,
            user_description="Test user description",
            user_agent_goals="Test agent goals",
            recent_propositions=[{"text": f"Working on {test_event.project}", "confidence": 8}],
            calendar_events=None,
            raw={"project": test_event.project}
        )
    
    def create_csv_fixture(self, tmp_path: Path, events: List[TestEvent]) -> Path:
        """Create a CSV fixture file for testing with csv_simulator."""
        csv_path = tmp_path / "test_events.csv"
        
        fieldnames = [
            "timestamp",
            "screenshot_path", 
            "user_name",
            "user_details",
            "calendar_events",
            "recent_observations",
            "context_update",
            "goals",
            "reasoning"
        ]
        
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for event in events:
                writer.writerow({
                    "timestamp": event.timestamp.strftime("%Y%m%d_%H%M%S"),
                    "screenshot_path": event.screenshot_path or "",
                    "user_name": event.user_name,
                    "user_details": json.dumps([{"text": f"Working on {event.project}", "confidence": 8}]),
                    "calendar_events": "",
                    "recent_observations": "[]",
                    "context_update": event.context_update,
                    "goals": "[]",
                    "reasoning": ""
                })
        
        return csv_path
    
    def get_telemetry_events(self, event_name: str = None):
        """Get telemetry events, optionally filtered by name."""
        return self.telemetry_sink.get_events(event_name)
    
    def get_notifications(self, project: str = None):
        """Get notifications, optionally filtered by project."""
        return self.notification_sink.get_notifications(project)
    
    def clear_telemetry(self):
        """Clear all telemetry events."""
        self.telemetry_sink.clear()
    
    def clear_notifications(self):
        """Clear all notifications."""
        self.notification_sink.clear()


class MockStateManager:
    """Mock state manager for faster testing."""
    
    def __init__(self, history: ProjectHistory):
        self.history = history
    
    def process_event(self, event: ContextEvent) -> Dict[str, Any]:
        """Mock event processing that just extracts project from raw data."""
        # Emit telemetry for observation processed (like the real state manager)
        from precursor.testing.telemetry import emit_telemetry
        emit_telemetry("observation_processed", {
            "timestamp": event.timestamp.isoformat(),
            "has_screenshot": event.screenshot is not None,
            "context_length": len(event.context_update) if event.context_update else 0
        })
        
        # Extract project from the raw data (set by test harness)
        project = event.raw.get("project", "Unknown Project")
        
        # Add to history
        self.history.append(
            timestamp=event.timestamp,
            project=project,
            objectives=[f"Working on {project}"]
        )
        
        return {
            "project": project,
            "induced_goals": [],
            "induction_reasoning": "Mock reasoning",
            "scratchpad_edits_summary": "Mock edits",
            "scratchpad_text": f"Mock scratchpad for {project}"
        }


class MockAgentManager:
    """Mock agent manager for testing."""
    
    def __init__(self):
        self.calls = []
    
    def run_for_project(self, project_name: str) -> Dict[str, Any]:
        """Mock agent manager run."""
        self.calls.append(project_name)
        return {
            "project": project_name,
            "candidates": [
                {
                    "task_description": f"Mock task for {project_name}",
                    "value_score": 8,
                    "feasibility_score": 7,
                    "safety_score": 9
                }
            ]
        }


class MockUIManager:
    """Mock UI manager for testing."""
    
    def __init__(self):
        self.calls = []
        self._pending_tasks = {}  # project -> bool
    
    def set_pending_tasks(self, project: str, has_pending: bool):
        """Set whether a project has pending tasks (for testing)."""
        self._pending_tasks[project] = has_pending
    
    def run_for_project(self, project_name: str, **kwargs) -> Dict[str, Any]:
        """Mock UI manager run that simulates the notification logic."""
        self.calls.append(project_name)
        
        # Simulate the pending tasks check
        has_pending = self._pending_tasks.get(project_name, False)
        
        if has_pending:
            from precursor.testing.telemetry import emit_telemetry, send_notification
            emit_telemetry("notification_sent", {
                "project": project_name,
                "reason": "pending_agent_tasks"
            })
            send_notification(
                project=project_name,
                message=f"Welcome back to {project_name}.",
                reason="pending_agent_tasks"
            )
        else:
            from precursor.testing.telemetry import emit_telemetry
            emit_telemetry("notification_skipped", {
                "project": project_name,
                "reason": "no_pending_agent_tasks"
            })
        
        return {
            "project": project_name,
            "notification": {
                "type": "project_return_if_pending",
                "message": f"Welcome back to {project_name}.",
            },
        }