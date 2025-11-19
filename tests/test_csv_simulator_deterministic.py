# tests/test_csv_simulator_deterministic.py
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pytest

from precursor.observers.csv_simulator import (
    CSVSimulatorObserver,
    CSVSimulatorConfig,
    FakeClock,
    RealClock,
    Clock,
)
from precursor.context.events import ContextEvent


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    """Helper to write CSV test data."""
    csv_path = tmp_path / "test_context_log.csv"
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
        for r in rows:
            writer.writerow(r)
    return csv_path


def _create_test_rows() -> List[dict]:
    """Create test rows with known timestamps for deterministic testing."""
    return [
        {
            "timestamp": "20251119_100000",  # 10:00:00
            "screenshot_path": "",
            "user_name": "Test User",
            "user_details": json.dumps([{"text": "first task", "confidence": 8}]),
            "calendar_events": "",
            "recent_observations": "[]",
            "context_update": "Starting first task",
            "goals": "[]",
            "reasoning": "",
        },
        {
            "timestamp": "20251119_100300",  # 10:03:00 (3 minutes later)
            "screenshot_path": "",
            "user_name": "Test User",
            "user_details": json.dumps([{"text": "second task", "confidence": 9}]),
            "calendar_events": "",
            "recent_observations": "[]",
            "context_update": "Working on second task",
            "goals": "[]",
            "reasoning": "",
        },
        {
            "timestamp": "20251119_100600",  # 10:06:00 (6 minutes from start)
            "screenshot_path": "",
            "user_name": "Test User",
            "user_details": json.dumps([{"text": "third task", "confidence": 7}]),
            "calendar_events": "",
            "recent_observations": "[]",
            "context_update": "Finishing third task",
            "goals": "[]",
            "reasoning": "",
        },
    ]


class TestClockAbstraction:
    """Test the clock abstraction classes."""
    
    def test_real_clock_returns_time(self):
        """Test that RealClock returns actual time."""
        clock = RealClock()
        import time
        before = time.time()
        now = clock.now()
        after = time.time()
        
        assert before <= now <= after
    
    def test_fake_clock_initial_time(self):
        """Test FakeClock initialization."""
        clock = FakeClock(100.0)
        assert clock.now() == 100.0
        
        clock_default = FakeClock()
        assert clock_default.now() == 0.0
    
    def test_fake_clock_advance(self):
        """Test FakeClock advance functionality."""
        clock = FakeClock(100.0)
        clock.advance(50.0)
        assert clock.now() == 150.0
        
        clock.advance(25.5)
        assert clock.now() == 175.5


class TestDeterministicMode:
    """Test deterministic mode functionality."""
    
    def test_deterministic_mode_requires_fake_clock_for_step(self, tmp_path):
        """Test that step() requires FakeClock."""
        rows = _create_test_rows()
        csv_path = _write_csv(tmp_path, rows)
        
        # Using RealClock should raise error for step()
        config = CSVSimulatorConfig(
            csv_path=str(csv_path),
            mode="deterministic",
            clock=RealClock()
        )
        sim = CSVSimulatorObserver(config=config)
        
        with pytest.raises(ValueError, match="step\\(\\) requires a FakeClock instance"):
            sim.step(100.0)
    
    def test_run_once_processes_single_eligible_row(self, tmp_path):
        """Test run_once processes at most one eligible row."""
        rows = _create_test_rows()
        csv_path = _write_csv(tmp_path, rows)
        
        # Start clock at first row's timestamp
        first_timestamp = datetime.strptime("20251119_100000", "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc).timestamp()
        clock = FakeClock(first_timestamp)
        
        config = CSVSimulatorConfig(
            csv_path=str(csv_path),
            mode="deterministic",
            clock=clock
        )
        sim = CSVSimulatorObserver(config=config)
        
        events = []
        
        # First call should process first row
        result = sim.run_once(lambda event: events.append(event))
        assert result is True
        assert len(events) == 1
        assert events[0].context_update == "Starting first task"
        
        # Second call with same time should return False (no more eligible rows)
        result = sim.run_once(lambda event: events.append(event))
        assert result is False
        assert len(events) == 1  # No new events
    
    def test_run_once_respects_timing(self, tmp_path):
        """Test run_once respects row timestamps."""
        rows = _create_test_rows()
        csv_path = _write_csv(tmp_path, rows)
        
        # Start clock before first row's timestamp
        first_timestamp = datetime.strptime("20251119_100000", "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc).timestamp()
        clock = FakeClock(first_timestamp - 60)  # 1 minute before
        
        config = CSVSimulatorConfig(
            csv_path=str(csv_path),
            mode="deterministic",
            clock=clock
        )
        sim = CSVSimulatorObserver(config=config)
        
        events = []
        
        # Should not process any rows yet
        result = sim.run_once(lambda event: events.append(event))
        assert result is False
        assert len(events) == 0
        
        # Advance clock to first row's time
        clock.advance(60)
        result = sim.run_once(lambda event: events.append(event))
        assert result is True
        assert len(events) == 1
        assert events[0].context_update == "Starting first task"
    
    def test_step_processes_all_eligible_rows(self, tmp_path):
        """Test step processes all rows that become eligible."""
        rows = _create_test_rows()
        csv_path = _write_csv(tmp_path, rows)
        
        # Start clock before all rows
        first_timestamp = datetime.strptime("20251119_100000", "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc).timestamp()
        clock = FakeClock(first_timestamp - 60)
        
        config = CSVSimulatorConfig(
            csv_path=str(csv_path),
            mode="deterministic",
            clock=clock
        )
        sim = CSVSimulatorObserver(config=config)
        
        # Step forward to make first row eligible
        events = sim.step(60)  # Advance 1 minute
        assert len(events) == 1
        assert events[0].context_update == "Starting first task"
        
        # Step forward to make second row eligible (3 minutes from first)
        events = sim.step(180)  # Advance 3 minutes
        assert len(events) == 1
        assert events[0].context_update == "Working on second task"
        
        # Step forward to make third row eligible (3 more minutes)
        events = sim.step(180)  # Advance 3 minutes
        assert len(events) == 1
        assert events[0].context_update == "Finishing third task"
        
        # No more rows
        events = sim.step(180)
        assert len(events) == 0
    
    def test_step_processes_multiple_rows_in_single_step(self, tmp_path):
        """Test step processes multiple rows if they all become eligible."""
        rows = _create_test_rows()
        csv_path = _write_csv(tmp_path, rows)
        
        # Start clock before all rows
        first_timestamp = datetime.strptime("20251119_100000", "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc).timestamp()
        clock = FakeClock(first_timestamp - 60)
        
        config = CSVSimulatorConfig(
            csv_path=str(csv_path),
            mode="deterministic",
            clock=clock
        )
        sim = CSVSimulatorObserver(config=config)
        
        # Step forward past all rows (7 minutes total + 1 minute buffer)
        events = sim.step(480)  # 8 minutes
        assert len(events) == 3
        assert events[0].context_update == "Starting first task"
        assert events[1].context_update == "Working on second task"
        assert events[2].context_update == "Finishing third task"


class TestDeterministicOrdering:
    """Test deterministic ordering behavior."""
    
    def test_row_ordering_determinism(self, tmp_path):
        """Test that row emission order is deterministic across runs."""
        # Create rows with same timestamp to test stable ordering
        rows = [
            {
                "timestamp": "20251119_100000",
                "screenshot_path": "",
                "user_name": "Test User",
                "user_details": json.dumps([{"text": "task A"}]),
                "calendar_events": "",
                "recent_observations": "[]",
                "context_update": "Task A",
                "goals": "[]",
                "reasoning": "",
            },
            {
                "timestamp": "20251119_100000",  # Same timestamp
                "screenshot_path": "",
                "user_name": "Test User",
                "user_details": json.dumps([{"text": "task B"}]),
                "calendar_events": "",
                "recent_observations": "[]",
                "context_update": "Task B",
                "goals": "[]",
                "reasoning": "",
            },
            {
                "timestamp": "20251119_100000",  # Same timestamp
                "screenshot_path": "",
                "user_name": "Test User",
                "user_details": json.dumps([{"text": "task C"}]),
                "calendar_events": "",
                "recent_observations": "[]",
                "context_update": "Task C",
                "goals": "[]",
                "reasoning": "",
            },
        ]
        csv_path = _write_csv(tmp_path, rows)
        
        # Run simulation twice and verify same order
        results1 = []
        results2 = []
        
        for results in [results1, results2]:
            timestamp = datetime.strptime("20251119_100000", "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc).timestamp()
            clock = FakeClock(timestamp)
            
            config = CSVSimulatorConfig(
                csv_path=str(csv_path),
                mode="deterministic",
                clock=clock
            )
            sim = CSVSimulatorObserver(config=config)
            
            events = sim.step(0)  # Process all eligible rows
            results.extend([event.context_update for event in events])
        
        assert results1 == results2
        assert len(results1) == 3
        # Should maintain CSV input order for same timestamps
        assert results1 == ["Task A", "Task B", "Task C"]
    
    def test_timing_determinism(self, tmp_path):
        """Test that timing behavior is deterministic across runs."""
        rows = _create_test_rows()
        csv_path = _write_csv(tmp_path, rows)
        
        # Run simulation twice with identical clock progression
        results1 = []
        results2 = []
        
        for results in [results1, results2]:
            first_timestamp = datetime.strptime("20251119_100000", "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc).timestamp()
            clock = FakeClock(first_timestamp - 60)
            
            config = CSVSimulatorConfig(
                csv_path=str(csv_path),
                mode="deterministic",
                clock=clock
            )
            sim = CSVSimulatorObserver(config=config)
            
            # Step through with fixed increments
            for step_size in [60, 180, 180, 180]:  # 1min, 3min, 3min, 3min
                events = sim.step(step_size)
                results.append(len(events))
        
        assert results1 == results2
        assert results1 == [1, 1, 1, 0]  # 1 event, 1 event, 1 event, 0 events


class TestBackwardCompatibility:
    """Test that existing behavior remains unchanged."""
    
    def test_default_mode_unchanged(self, tmp_path):
        """Test that default mode behavior is unchanged."""
        rows = _create_test_rows()
        csv_path = _write_csv(tmp_path, rows)
        
        # Default config should work as before
        config = CSVSimulatorConfig(csv_path=str(csv_path), mode="asap")
        sim = CSVSimulatorObserver(config=config)
        
        events = []
        
        async def run_test():
            await sim.run(lambda event: events.append(event))
        
        import asyncio
        asyncio.run(run_test())
        
        assert len(events) == 3
        assert events[0].context_update == "Starting first task"
        assert events[1].context_update == "Working on second task"
        assert events[2].context_update == "Finishing third task"
    
    def test_run_once_and_step_only_in_deterministic_mode(self, tmp_path):
        """Test that run_once and step only work in deterministic mode."""
        rows = _create_test_rows()
        csv_path = _write_csv(tmp_path, rows)
        
        # Test with interval mode
        config = CSVSimulatorConfig(csv_path=str(csv_path), mode="interval")
        sim = CSVSimulatorObserver(config=config)
        
        with pytest.raises(ValueError, match="run_once\\(\\) only available in deterministic mode"):
            sim.run_once(lambda event: None)
        
        with pytest.raises(ValueError, match="step\\(\\) only available in deterministic mode"):
            sim.step(100.0)
        
        # Test with asap mode
        config = CSVSimulatorConfig(csv_path=str(csv_path), mode="asap")
        sim = CSVSimulatorObserver(config=config)
        
        with pytest.raises(ValueError, match="run_once\\(\\) only available in deterministic mode"):
            sim.run_once(lambda event: None)
        
        with pytest.raises(ValueError, match="step\\(\\) only available in deterministic mode"):
            sim.step(100.0)
    
    def test_default_clock_is_real_clock(self, tmp_path):
        """Test that default clock is RealClock."""
        rows = _create_test_rows()
        csv_path = _write_csv(tmp_path, rows)
        
        config = CSVSimulatorConfig(csv_path=str(csv_path))
        sim = CSVSimulatorObserver(config=config)
        
        assert isinstance(sim.clock, RealClock)
    
    def test_injectable_clock_works(self, tmp_path):
        """Test that injectable clock works correctly."""
        rows = _create_test_rows()
        csv_path = _write_csv(tmp_path, rows)
        
        fake_clock = FakeClock(1000.0)
        config = CSVSimulatorConfig(csv_path=str(csv_path), clock=fake_clock)
        sim = CSVSimulatorObserver(config=config)
        
        assert sim.clock is fake_clock
        assert sim.clock.now() == 1000.0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_csv_file(self, tmp_path):
        """Test behavior with empty CSV file."""
        csv_path = _write_csv(tmp_path, [])
        
        clock = FakeClock(0.0)
        config = CSVSimulatorConfig(
            csv_path=str(csv_path),
            mode="deterministic",
            clock=clock
        )
        sim = CSVSimulatorObserver(config=config)
        
        # Should handle empty file gracefully
        events = sim.step(1000.0)
        assert len(events) == 0
        
        result = sim.run_once(lambda event: None)
        assert result is False
    
    def test_rows_without_timestamps(self, tmp_path):
        """Test behavior with rows that have no timestamp."""
        rows = [
            {
                "timestamp": "",  # Empty timestamp
                "screenshot_path": "",
                "user_name": "Test User",
                "user_details": json.dumps([{"text": "no timestamp"}]),
                "calendar_events": "",
                "recent_observations": "[]",
                "context_update": "No timestamp task",
                "goals": "[]",
                "reasoning": "",
            }
        ]
        csv_path = _write_csv(tmp_path, rows)
        
        clock = FakeClock(0.0)
        config = CSVSimulatorConfig(
            csv_path=str(csv_path),
            mode="deterministic",
            clock=clock
        )
        sim = CSVSimulatorObserver(config=config)
        
        # Should still work, using current time for missing timestamps
        events = sim.step(float('inf'))
        assert len(events) == 1
        assert events[0].context_update == "No timestamp task"