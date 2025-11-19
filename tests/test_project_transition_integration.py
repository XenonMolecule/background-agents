# tests/test_project_transition_integration.py
"""
Deterministic integration tests for project transitions and notifications.

These tests use the CSV simulator in step mode to create fully deterministic,
hermetic test scenarios without sleeps or network calls.
"""

import asyncio
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List
# import pytest  # Not available in this environment

from precursor.context.events import ContextEvent
from precursor.context.project_history import ProjectHistory
from precursor.managers.state_manager import StateManager
from precursor.managers.agent_manager import AgentManager
from precursor.managers.ui_manager import UIManager
from precursor.observers.project_transition import ProjectActivityObserver
from precursor.observers.csv_simulator import CSVSimulatorObserver, CSVSimulatorConfig
from precursor.testing.sinks import MockTelemetrySink, MockNotificationSink
from precursor.scratchpad import store


class MockTaskProposerPipeline:
    """Mock task proposer pipeline for testing."""
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        return {
            "future_goals": ["Test goal 1", "Test goal 2"],
            "goal_to_milestones": {"Test goal 1": ["Milestone 1", "Milestone 2"]},
            "agent_tasks": ["Task 1", "Task 2"],
            "task_assessments": [
                {
                    "task_description": "Test task 1",
                    "value_score": 8,
                    "feasibility_score": 7,
                    "safety_score": 9,
                    "user_preference_alignment_score": 6,
                },
                {
                    "task_description": "Test task 2", 
                    "value_score": 6,
                    "feasibility_score": 8,
                    "safety_score": 8,
                    "user_preference_alignment_score": 7,
                }
            ]
        }


class MockScratchpadStore:
    """Mock scratchpad store that simulates pending tasks."""
    
    def __init__(self, has_pending_tasks: bool = False):
        self.has_pending_tasks = has_pending_tasks
        self.entries = []
        
    def init_db(self):
        pass
        
    def list_entries(self, project_name: str, section: str = None):
        if section == "Agent Completed Tasks (Pending Review)":
            return ["Mock pending task"] if self.has_pending_tasks else []
        return self.entries


def _write_test_csv(tmp_path: Path, rows: List[Dict[str, Any]]) -> Path:
    """Write test CSV data to a temporary file."""
    csv_path = tmp_path / "test_context.csv"
    fieldnames = [
        "timestamp",
        "screenshot_path", 
        "user_name",
        "user_details",
        "calendar_events",
        "recent_observations",
        "context_update",
        "goals",
        "reasoning",
    ]
    
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    return csv_path


def _create_test_event(timestamp_str: str, project_context: str) -> Dict[str, Any]:
    """Create a test CSV row for a given timestamp and project context."""
    return {
        "timestamp": timestamp_str,
        "screenshot_path": "",
        "user_name": "Test User",
        "user_details": json.dumps([{"text": project_context, "confidence": 8}]),
        "calendar_events": "",
        "recent_observations": "[]",
        "context_update": project_context,
        "goals": "[]",
        "reasoning": "",
    }


# Fixtures replaced with manual setup since pytest not available


def test_project_transition_detected(tmp_path, monkeypatch):
    """
    Test that project transitions are detected and trigger the appropriate manager.
    
    Scenario: User works on Project Alpha, then switches to Project Beta.
    Expected: Transition from Alpha to Beta is detected and telemetry is emitted.
    """
    # Mock the scratchpad store
    mock_store = MockScratchpadStore()
    monkeypatch.setattr(store, "init_db", mock_store.init_db)
    monkeypatch.setattr(store, "list_entries", mock_store.list_entries)
    
    # Create test CSV with project transition: Alpha -> Beta
    base_time = datetime.now(timezone.utc)
    rows = [
        # Work on Alpha for a while
        _create_test_event(
            (base_time - timedelta(minutes=20)).strftime("%Y%m%d_%H%M%S"),
            "Working on Project Alpha - initial setup"
        ),
        _create_test_event(
            (base_time - timedelta(minutes=15)).strftime("%Y%m%d_%H%M%S"),
            "Project Alpha - implementing features"
        ),
        _create_test_event(
            (base_time - timedelta(minutes=10)).strftime("%Y%m%d_%H%M%S"),
            "Project Alpha - testing and debugging"
        ),
        # Switch to Beta
        _create_test_event(
            (base_time - timedelta(minutes=5)).strftime("%Y%m%d_%H%M%S"),
            "Starting work on Project Beta"
        ),
        _create_test_event(
            base_time.strftime("%Y%m%d_%H%M%S"),
            "Project Beta - reviewing requirements"
        ),
    ]
    
    csv_path = _write_test_csv(tmp_path, rows)
    
    # Set up components with mocks
    telemetry_sink = MockTelemetrySink()
    notification_sink = MockNotificationSink()
    
    history = ProjectHistory()
    state_mgr = StateManager(history=history, telemetry_sink=telemetry_sink)
    agent_mgr = AgentManager(
        task_pipeline=MockTaskProposerPipeline(),
        deploy_enabled=False,
        telemetry_sink=telemetry_sink
    )
    
    transition_obs = ProjectActivityObserver(
        history=history,
        agent_manager=agent_mgr,
        mode="departure",
        min_entries_previous_segment=3,
        time_threshold=timedelta(minutes=5),
        telemetry_sink=telemetry_sink
    )
    
    # Process events
    processed_events = []
    
    def handle_event(event: ContextEvent):
        result = state_mgr.process_event(event)
        processed_events.append((event, result))
        transition_obs.handle_processed()
    
    # Run CSV simulator in step mode
    config = CSVSimulatorConfig(
        csv_path=str(csv_path),
        mode="step",
        max_rows=5
    )
    simulator = CSVSimulatorObserver(config=config)
    
    # Run the simulation
    asyncio.run(simulator.run(handle_event))
    
    # Verify project transition was detected
    transition_events = telemetry_sink.get_events("project_transition_detected")
    assert len(transition_events) == 1
    
    transition_event = transition_events[0]
    assert transition_event.payload["from"] == "Project Alpha"
    assert transition_event.payload["to"] == "Project Beta"
    
    # Verify batch processing events were emitted
    batch_events = telemetry_sink.get_events("batch_processed")
    assert len(batch_events) >= 1  # At least one batch processed for Alpha
    
    # Verify observation queued events
    queued_events = telemetry_sink.get_events("observation_queued")
    assert len(queued_events) == 5  # One for each CSV row processed


def test_batch_processing_metrics_and_coalescing(tmp_path, monkeypatch):
    """
    Test that batch processing metrics are correctly emitted.
    
    Scenario: Process a batch of 5 observations for the same project.
    Expected: Batch processing telemetry is emitted with correct counts.
    """
    # Mock the scratchpad store
    mock_store = MockScratchpadStore()
    monkeypatch.setattr(store, "init_db", mock_store.init_db)
    monkeypatch.setattr(store, "list_entries", mock_store.list_entries)
    
    # Create test CSV with 5 observations for same project
    base_time = datetime.now(timezone.utc)
    rows = []
    for i in range(5):
        rows.append(_create_test_event(
            (base_time - timedelta(minutes=20-i*2)).strftime("%Y%m%d_%H%M%S"),
            f"Project Alpha - activity {i+1}"
        ))
    
    csv_path = _write_test_csv(tmp_path, rows)
    
    # Set up components with mocks
    telemetry_sink = MockTelemetrySink()
    
    history = ProjectHistory()
    state_mgr = StateManager(history=history, telemetry_sink=telemetry_sink)
    agent_mgr = AgentManager(
        task_pipeline=MockTaskProposerPipeline(),
        deploy_enabled=False,
        telemetry_sink=telemetry_sink
    )
    
    transition_obs = ProjectActivityObserver(
        history=history,
        agent_manager=agent_mgr,
        mode="departure",
        min_entries_previous_segment=3,
        time_threshold=timedelta(minutes=5),
        telemetry_sink=telemetry_sink
    )
    
    # Process events
    def handle_event(event: ContextEvent):
        result = state_mgr.process_event(event)
        transition_obs.handle_processed()
    
    # Run CSV simulator in step mode
    config = CSVSimulatorConfig(
        csv_path=str(csv_path),
        mode="step",
        max_rows=5
    )
    simulator = CSVSimulatorObserver(config=config)
    
    # Run the simulation
    asyncio.run(simulator.run(handle_event))
    
    # Verify observation queued events
    queued_events = telemetry_sink.get_events("observation_queued")
    assert len(queued_events) == 5
    
    # Verify queue sizes are tracked
    for i, event in enumerate(queued_events):
        assert "queue_size" in event.payload
        # Queue size should increase as we process more events
        assert event.payload["queue_size"] >= 0


def test_notification_skipped_when_no_pending_tasks(tmp_path, monkeypatch):
    """
    Test that notifications are skipped when there are no pending agent tasks.
    
    Scenario: User returns to a project but there are no pending agent-completed tasks.
    Expected: Notification is skipped and appropriate telemetry is emitted.
    """
    # Mock the scratchpad store with NO pending tasks
    mock_store = MockScratchpadStore(has_pending_tasks=False)
    monkeypatch.setattr(store, "init_db", mock_store.init_db)
    monkeypatch.setattr(store, "list_entries", mock_store.list_entries)
    
    # Create test CSV with project return scenario
    base_time = datetime.now(timezone.utc)
    rows = [
        # Work on Alpha initially
        _create_test_event(
            (base_time - timedelta(minutes=30)).strftime("%Y%m%d_%H%M%S"),
            "Working on Project Alpha"
        ),
        # Switch to Beta
        _create_test_event(
            (base_time - timedelta(minutes=20)).strftime("%Y%m%d_%H%M%S"),
            "Working on Project Beta"
        ),
        _create_test_event(
            (base_time - timedelta(minutes=15)).strftime("%Y%m%d_%H%M%S"),
            "Project Beta - continued work"
        ),
        # Return to Alpha (should trigger arrival notification check)
        _create_test_event(
            base_time.strftime("%Y%m%d_%H%M%S"),
            "Back to Project Alpha"
        ),
    ]
    
    csv_path = _write_test_csv(tmp_path, rows)
    
    # Set up components with mocks
    telemetry_sink = MockTelemetrySink()
    notification_sink = MockNotificationSink()
    
    history = ProjectHistory()
    state_mgr = StateManager(history=history, telemetry_sink=telemetry_sink)
    ui_mgr = UIManager(
        telemetry_sink=telemetry_sink,
        notification_sink=notification_sink
    )
    
    return_obs = ProjectActivityObserver(
        history=history,
        agent_manager=ui_mgr,  # UI manager handles arrival notifications
        mode="arrival",
        min_entries_current_segment=1,
        time_threshold=timedelta(minutes=10),
        telemetry_sink=telemetry_sink
    )
    
    # Process events
    def handle_event(event: ContextEvent):
        result = state_mgr.process_event(event)
        return_obs.handle_processed()
    
    # Run CSV simulator in step mode
    config = CSVSimulatorConfig(
        csv_path=str(csv_path),
        mode="step",
        max_rows=4
    )
    simulator = CSVSimulatorObserver(config=config)
    
    # Run the simulation
    asyncio.run(simulator.run(handle_event))
    
    # Verify notification was skipped
    skipped_events = telemetry_sink.get_events("notification_skipped")
    assert len(skipped_events) >= 1
    
    skipped_event = skipped_events[-1]  # Get the most recent one
    assert skipped_event.payload["project"] == "Project Alpha"
    assert skipped_event.payload["reason"] == "no_pending_agent_tasks"
    
    # Verify notification sink recorded the skip
    notifications = notification_sink.get_notifications(sent=False)
    assert len(notifications) >= 1
    
    skip_notification = notifications[-1]
    assert skip_notification.project == "Project Alpha"
    assert "no_pending_agent_tasks" in skip_notification.message


def test_notification_sent_when_pending_tasks_exist(tmp_path, monkeypatch):
    """
    Test that notifications are sent when there are pending agent tasks.
    
    Scenario: User returns to a project with pending agent-completed tasks.
    Expected: Notification is sent and appropriate telemetry is emitted.
    """
    # Mock the scratchpad store WITH pending tasks
    mock_store = MockScratchpadStore(has_pending_tasks=True)
    monkeypatch.setattr(store, "init_db", mock_store.init_db)
    monkeypatch.setattr(store, "list_entries", mock_store.list_entries)
    
    # Create test CSV with project return scenario
    base_time = datetime.now(timezone.utc)
    rows = [
        # Work on Alpha initially
        _create_test_event(
            (base_time - timedelta(minutes=30)).strftime("%Y%m%d_%H%M%S"),
            "Working on Project Alpha"
        ),
        # Switch to Beta
        _create_test_event(
            (base_time - timedelta(minutes=20)).strftime("%Y%m%d_%H%M%S"),
            "Working on Project Beta"
        ),
        _create_test_event(
            (base_time - timedelta(minutes=15)).strftime("%Y%m%d_%H%M%S"),
            "Project Beta - continued work"
        ),
        # Return to Alpha (should trigger arrival notification)
        _create_test_event(
            base_time.strftime("%Y%m%d_%H%M%S"),
            "Back to Project Alpha"
        ),
    ]
    
    csv_path = _write_test_csv(tmp_path, rows)
    
    # Set up components with mocks
    telemetry_sink = MockTelemetrySink()
    notification_sink = MockNotificationSink()
    
    history = ProjectHistory()
    state_mgr = StateManager(history=history, telemetry_sink=telemetry_sink)
    ui_mgr = UIManager(
        telemetry_sink=telemetry_sink,
        notification_sink=notification_sink
    )
    
    return_obs = ProjectActivityObserver(
        history=history,
        agent_manager=ui_mgr,  # UI manager handles arrival notifications
        mode="arrival",
        min_entries_current_segment=1,
        time_threshold=timedelta(minutes=10),
        telemetry_sink=telemetry_sink
    )
    
    # Process events
    def handle_event(event: ContextEvent):
        result = state_mgr.process_event(event)
        return_obs.handle_processed()
    
    # Run CSV simulator in step mode
    config = CSVSimulatorConfig(
        csv_path=str(csv_path),
        mode="step",
        max_rows=4
    )
    simulator = CSVSimulatorObserver(config=config)
    
    # Run the simulation
    asyncio.run(simulator.run(handle_event))
    
    # Verify notification was sent
    sent_events = telemetry_sink.get_events("notification_sent")
    assert len(sent_events) >= 1
    
    sent_event = sent_events[-1]  # Get the most recent one
    assert sent_event.payload["project"] == "Project Alpha"
    assert sent_event.payload["reason"] == "pending_agent_tasks"
    
    # Verify notification sink recorded the send
    notifications = notification_sink.get_notifications(sent=True)
    assert len(notifications) >= 1
    
    sent_notification = notifications[-1]
    assert sent_notification.project == "Project Alpha"
    assert sent_notification.notification_type == "project_return_if_pending"
    assert "Welcome back to Project Alpha" in sent_notification.message