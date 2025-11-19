# tests/observers/test_project_transition_integration.py
"""
Integration tests for project transition + notifications using deterministic scenarios.

These tests use the DeterministicTestHarness to reproduce:
1. Project transition detection
2. Batch processing with coalescing/queue updates  
3. The "skipping notification" case when there are no pending agent-completed tasks
4. Notification sent when pending tasks exist

All tests are deterministic and use telemetry events for assertions.
"""

import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from precursor.testing.harness import DeterministicTestHarness, TestEvent
from precursor.observers.csv_simulator import CSVSimulatorObserver, CSVSimulatorConfig


class TestProjectTransitionIntegration:
    """Integration tests for project transition detection and notifications."""
    
    def test_project_transition_detected(self):
        """
        Test that project transition detection works correctly.
        
        Scenario: Feed a sequence that starts with Project A observations 
        and then switches to Project B. Assert that project transition logic 
        triggers exactly one transition event.
        """
        harness = DeterministicTestHarness(enable_scratchpad=False)
        
        # Create events: 4 events for Project Alpha, then switch to Project Beta
        base_time = datetime.now(timezone.utc)
        events = [
            # Project Alpha segment (4 events over 12 minutes)
            TestEvent(
                timestamp=base_time - timedelta(minutes=15),
                project="Project Alpha",
                context_update="Starting work on Alpha project"
            ),
            TestEvent(
                timestamp=base_time - timedelta(minutes=12),
                project="Project Alpha", 
                context_update="Continuing Alpha development"
            ),
            TestEvent(
                timestamp=base_time - timedelta(minutes=9),
                project="Project Alpha",
                context_update="Alpha feature implementation"
            ),
            TestEvent(
                timestamp=base_time - timedelta(minutes=6),
                project="Project Alpha",
                context_update="Finalizing Alpha work"
            ),
            # Switch to Project Beta (3 events)
            TestEvent(
                timestamp=base_time - timedelta(minutes=3),
                project="Project Beta",
                context_update="Starting Beta project work"
            ),
            TestEvent(
                timestamp=base_time - timedelta(minutes=1),
                project="Project Beta",
                context_update="Beta development continues"
            ),
            TestEvent(
                timestamp=base_time,
                project="Project Beta",
                context_update="Current Beta work"
            ),
        ]
        
        # Process events
        harness.process_events(events)
        
        # Assert project transition was detected
        transition_events = harness.get_telemetry_events("project_transition_detected")
        assert len(transition_events) == 1
        
        transition_event = transition_events[0]
        assert transition_event.payload["from"] == "Project Alpha"
        assert transition_event.payload["to"] == "Project Beta"
        assert transition_event.payload["entries"] == 4
        assert transition_event.payload["duration_seconds"] >= 540  # At least 9 minutes
        
        # Assert agent manager was called for the departed project
        assert "Project Alpha" in harness.agent_manager.calls
        
        # Assert batch processing telemetry
        batch_events = harness.get_telemetry_events("batch_processed")
        assert len(batch_events) == 1
        assert batch_events[0].payload["count"] == 7
    
    def test_batch_processing_metrics_and_notifications(self):
        """
        Test batch processing with queue metrics and notification behavior.
        
        Scenario: Feed a batch of 5 observations. Assert we emit telemetry 
        for queue growth and a batch_processed event with count=5.
        """
        harness = DeterministicTestHarness(enable_scratchpad=False)
        
        base_time = datetime.now(timezone.utc)
        events = [
            TestEvent(
                timestamp=base_time - timedelta(minutes=i*2),
                project="Project Gamma",
                context_update=f"Gamma work step {i+1}"
            )
            for i in range(5)
        ]
        
        # Process the batch
        harness.process_events(events)
        
        # Assert observation queued events
        queued_events = harness.get_telemetry_events("observation_queued")
        assert len(queued_events) == 5
        
        # Check queue size progression
        queue_sizes = [event.payload["queue_size"] for event in queued_events]
        assert queue_sizes == [1, 2, 3, 4, 5]
        
        # Assert batch processed event
        batch_events = harness.get_telemetry_events("batch_processed")
        assert len(batch_events) == 1
        assert batch_events[0].payload["count"] == 5
        
        # Assert observation processed events
        processed_events = harness.get_telemetry_events("observation_processed")
        assert len(processed_events) == 5
    
    def test_notification_skipped_when_no_pending_tasks(self):
        """
        Test the case where notifications are skipped due to no pending tasks.
        
        Scenario: Reproduce the case where the system would display 
        "skipping notification for Background Agents (Precursor) (no pending agent-completed tasks)".
        Assert we emit a telemetry event notification_skipped with reason="no_pending_agent_tasks".
        """
        harness = DeterministicTestHarness(enable_scratchpad=False)
        
        # Set up scenario where user returns to a project but there are no pending tasks
        base_time = datetime.now(timezone.utc)
        events = [
            # Previous work on Project Delta
            TestEvent(
                timestamp=base_time - timedelta(minutes=30),
                project="Project Delta",
                context_update="Previous Delta work"
            ),
            TestEvent(
                timestamp=base_time - timedelta(minutes=27),
                project="Project Delta",
                context_update="More Delta work"
            ),
            # Switch to different project
            TestEvent(
                timestamp=base_time - timedelta(minutes=20),
                project="Project Epsilon",
                context_update="Working on Epsilon"
            ),
            TestEvent(
                timestamp=base_time - timedelta(minutes=17),
                project="Project Epsilon", 
                context_update="Epsilon development"
            ),
            # Return to Project Delta (should trigger arrival observer)
            TestEvent(
                timestamp=base_time,
                project="Project Delta",
                context_update="Returning to Delta work"
            ),
        ]
        
        # Ensure no pending tasks for Project Delta
        harness.ui_manager.set_pending_tasks("Project Delta", False)
        
        # Process events
        harness.process_events(events)
        
        # Assert notification was skipped
        skipped_events = harness.get_telemetry_events("notification_skipped")
        assert len(skipped_events) == 1
        
        skipped_event = skipped_events[0]
        assert skipped_event.payload["project"] == "Project Delta"
        assert skipped_event.payload["reason"] == "no_pending_agent_tasks"
        
        # Assert no notifications were sent
        notifications = harness.get_notifications()
        assert len(notifications) == 0
        
        # Assert UI manager was still called (but skipped notification)
        assert "Project Delta" in harness.ui_manager.calls
    
    def test_notification_sent_when_pending_tasks_exist(self):
        """
        Test that notifications are sent when pending agent-completed tasks exist.
        
        Scenario: Ensure that when pending agent-completed tasks are present, 
        a notification is sent and telemetry notification_sent is emitted 
        with the correct project and reason.
        """
        harness = DeterministicTestHarness(enable_scratchpad=False)
        
        # Set up scenario where user returns to a project with pending tasks
        base_time = datetime.now(timezone.utc)
        events = [
            # Previous work on Project Zeta
            TestEvent(
                timestamp=base_time - timedelta(minutes=25),
                project="Project Zeta",
                context_update="Previous Zeta work"
            ),
            TestEvent(
                timestamp=base_time - timedelta(minutes=22),
                project="Project Zeta",
                context_update="More Zeta development"
            ),
            # Switch to different project
            TestEvent(
                timestamp=base_time - timedelta(minutes=18),
                project="Project Eta",
                context_update="Working on Eta"
            ),
            TestEvent(
                timestamp=base_time - timedelta(minutes=15),
                project="Project Eta",
                context_update="Eta implementation"
            ),
            # Return to Project Zeta (should trigger arrival observer)
            TestEvent(
                timestamp=base_time,
                project="Project Zeta",
                context_update="Returning to Zeta work"
            ),
        ]
        
        # Set up pending tasks for Project Zeta
        harness.ui_manager.set_pending_tasks("Project Zeta", True)
        
        # Process events
        harness.process_events(events)
        
        # Assert notification was sent
        sent_events = harness.get_telemetry_events("notification_sent")
        assert len(sent_events) == 1
        
        sent_event = sent_events[0]
        assert sent_event.payload["project"] == "Project Zeta"
        assert sent_event.payload["reason"] == "pending_agent_tasks"
        
        # Assert notification was actually sent via notification sink
        notifications = harness.get_notifications("Project Zeta")
        assert len(notifications) == 1
        
        notification = notifications[0]
        assert notification["project"] == "Project Zeta"
        assert notification["message"] == "Welcome back to Project Zeta."
        assert notification["reason"] == "pending_agent_tasks"
        
        # Assert UI manager was called
        assert "Project Zeta" in harness.ui_manager.calls


class TestCSVSimulatorIntegration:
    """Integration tests using CSV simulator for deterministic replay."""
    
    def test_csv_simulator_deterministic_project_transition(self, tmp_path):
        """
        Test project transition detection using CSV simulator for deterministic replay.
        
        This test demonstrates how to use csv_simulator.py to drive deterministic 
        input without real time.sleep calls.
        """
        harness = DeterministicTestHarness(enable_scratchpad=False)
        
        # Create test events
        base_time = datetime.now(timezone.utc)
        events = [
            TestEvent(
                timestamp=base_time - timedelta(minutes=20),
                project="CSV Project A",
                context_update="CSV-driven Project A work"
            ),
            TestEvent(
                timestamp=base_time - timedelta(minutes=17),
                project="CSV Project A",
                context_update="More CSV Project A work"
            ),
            TestEvent(
                timestamp=base_time - timedelta(minutes=14),
                project="CSV Project A",
                context_update="Final CSV Project A work"
            ),
            TestEvent(
                timestamp=base_time - timedelta(minutes=10),
                project="CSV Project B",
                context_update="Switching to CSV Project B"
            ),
            TestEvent(
                timestamp=base_time - timedelta(minutes=7),
                project="CSV Project B",
                context_update="CSV Project B development"
            ),
        ]
        
        # Create CSV fixture
        csv_path = harness.create_csv_fixture(tmp_path, events)
        
        # Use CSV simulator to replay events
        config = CSVSimulatorConfig(
            csv_path=str(csv_path),
            mode="asap",  # No delays for deterministic testing
            user_name="CSV Test User"
        )
        
        simulator = CSVSimulatorObserver(config=config)
        processed_events = []
        
        def event_handler(event):
            # Extract project from user_details (which contains recent_propositions)
            # and add it to raw field for our mock state manager
            project = "Unknown Project"
            if event.recent_propositions:
                for prop in event.recent_propositions:
                    if isinstance(prop, dict) and "text" in prop:
                        text = prop["text"]
                        if "CSV Project A" in text:
                            project = "CSV Project A"
                        elif "CSV Project B" in text:
                            project = "CSV Project B"
                        break
            
            # Add project to raw field for mock state manager
            if not hasattr(event, 'raw') or event.raw is None:
                event.raw = {}
            event.raw["project"] = project
            
            # Process through our harness state manager
            result = harness.state_manager.process_event(event)
            processed_events.append(event)
        
        # Run the simulator synchronously
        import asyncio
        asyncio.run(simulator.run(event_handler))
        
        # Trigger observers after all events processed (simulates batch processing)
        harness.transition_observer.handle_processed()
        harness.arrival_observer.handle_processed()
        
        # Emit batch processed telemetry
        from precursor.testing.telemetry import emit_telemetry
        emit_telemetry("batch_processed", {
            "count": len(processed_events)
        })
        
        # Assert events were processed
        assert len(processed_events) == 5
        
        # Assert project transition was detected
        transition_events = harness.get_telemetry_events("project_transition_detected")
        assert len(transition_events) == 1
        
        transition_event = transition_events[0]
        assert transition_event.payload["from"] == "CSV Project A"
        assert transition_event.payload["to"] == "CSV Project B"
    
    def test_csv_simulator_batch_processing_coalescing(self, tmp_path):
        """
        Test batch processing and coalescing behavior using CSV simulator.
        
        This test simulates rapid events that would be coalesced in real usage,
        ensuring the system handles batch processing correctly.
        """
        harness = DeterministicTestHarness(enable_scratchpad=False)
        
        # Create rapid sequence of events (simulating coalescing scenario)
        base_time = datetime.now(timezone.utc)
        events = []
        
        # Rapid sequence of 8 events over 2 minutes (would normally be coalesced)
        for i in range(8):
            events.append(TestEvent(
                timestamp=base_time - timedelta(seconds=i*15),  # Every 15 seconds
                project="Rapid Project",
                context_update=f"Rapid update {i+1}"
            ))
        
        # Create CSV and process
        csv_path = harness.create_csv_fixture(tmp_path, events)
        
        config = CSVSimulatorConfig(
            csv_path=str(csv_path),
            mode="asap"
        )
        
        simulator = CSVSimulatorObserver(config=config)
        processed_count = 0
        
        def event_handler(event):
            nonlocal processed_count
            # Add project to raw field for mock state manager
            if not hasattr(event, 'raw') or event.raw is None:
                event.raw = {}
            event.raw["project"] = "Rapid Project"
            
            harness.state_manager.process_event(event)
            processed_count += 1
        
        # Process all events
        import asyncio
        asyncio.run(simulator.run(event_handler))
        
        # Trigger observers (simulates end-of-batch processing)
        harness.transition_observer.handle_processed()
        harness.arrival_observer.handle_processed()
        
        # Emit batch processed telemetry
        from precursor.testing.telemetry import emit_telemetry
        emit_telemetry("batch_processed", {
            "count": processed_count
        })
        
        # Assert all events were processed
        assert processed_count == 8
        
        # Assert observation processing telemetry
        processed_events = harness.get_telemetry_events("observation_processed")
        assert len(processed_events) == 8
        
        # In a real scenario, these rapid events might be coalesced,
        # but our deterministic test processes them all individually
        # This demonstrates the system can handle high-frequency events


def test_integration_readme_instructions():
    """
    Test that demonstrates how to run these integration tests locally.
    
    This test serves as documentation for running the integration tests.
    Run with: pytest -k project_transition_integration
    """
    # This test always passes and serves as documentation
    assert True
    
    # Instructions for running these tests:
    # 1. Run all integration tests:
    #    pytest tests/observers/test_project_transition_integration.py
    #
    # 2. Run specific test scenarios:
    #    pytest -k test_project_transition_detected
    #    pytest -k test_notification_skipped
    #    pytest -k test_csv_simulator
    #
    # 3. Run with verbose output to see telemetry events:
    #    pytest -v -s tests/observers/test_project_transition_integration.py
    #
    # 4. Run integration tests only:
    #    pytest -k project_transition_integration