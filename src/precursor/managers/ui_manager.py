from __future__ import annotations

import subprocess
from pathlib import Path
import logging
from typing import Dict, Any, Optional

import precursor
from precursor.scratchpad import store

logger = logging.getLogger(__name__)

# Import sinks for optional injection
try:
    from precursor.testing.sinks import TelemetrySink, NotificationSink
except ImportError:
    # Define minimal interfaces if testing module not available
    from abc import ABC, abstractmethod
    
    class TelemetrySink(ABC):
        @abstractmethod
        def emit(self, event_name: str, payload: Dict[str, Any]) -> None:
            pass
    
    class NotificationSink(ABC):
        @abstractmethod
        def send_notification(self, project: str, notification_type: str, message: str) -> None:
            pass
        
        @abstractmethod
        def skip_notification(self, project: str, reason: str) -> None:
            pass


class UIManager:
    """
    Simple UI-oriented manager that can be triggered by observers.
    This version intentionally uses a 'janky dev mode' approach:
      - Always kills existing PrecursorApp processes
      - Always launches a fresh `swift run PrecursorApp` instance

    TODO (Long-Term, Real Solution):
        • Build PrecursorApp as a proper macOS `.app` bundle
        • Register a custom URL scheme (e.g., precursor://open?project=X)
        • Add `onOpenURL` or event handler in Swift to route to the correct project
        • Use:
              open "precursor://open?project=X"
          OR:
              open -a PrecursorApp --args --project X
          to reliably reuse/open/focus the same app instance.

        Once bundled:
        • Remove `pkill`
        • Replace this entire launcher with `open -a PrecursorApp ...`
        • Use Apple Event bridging to focus + reopen windows
    """

    def __init__(
        self,
        *,
        telemetry_sink: Optional[TelemetrySink] = None,
        notification_sink: Optional[NotificationSink] = None,
    ) -> None:
        """
        Initialize UIManager with optional sinks for testing.
        
        Args:
            telemetry_sink: Optional sink for telemetry events (for testing)
            notification_sink: Optional sink for notifications (for testing)
        """
        self.telemetry_sink = telemetry_sink
        self.notification_sink = notification_sink

    def _resolve_precursor_swift_root(self) -> Path:
        """
        Resolve the Swift PrecursorApp package root based on the installed Python package layout.
        """
        pkg_dir = Path(precursor.__file__).resolve().parent
        src_dir = pkg_dir.parent
        swift_root = src_dir / "interface" / "PrecursorApp"
        if not (swift_root / "Package.swift").exists():
            raise RuntimeError(f"PrecursorApp Swift package not found at {swift_root}")
        return swift_root

    def _has_pending_agent_tasks(self, project_name: str) -> bool:
        """
        Check whether the scratchpad has entries in
        'Agent Completed Tasks (Pending Review)'.
        """
        try:
            store.init_db()
            rows = store.list_entries(
                project_name, section="Agent Completed Tasks (Pending Review)"
            )
            return bool(rows)
        except Exception:
            logger.exception("ui_manager: failed to read scratchpad entries")
            return False

    def _notify_precursor_for_project(self, project_name: str) -> None:
        """
        Show a macOS notification which, when clicked, will:

          1) Kill any existing PrecursorApp processes (dev-only behavior)
          2) Launch a new `swift run PrecursorApp --project "<name>"`

        This avoids the complexity of reopening/focusing an existing GUI
        process created via `swift run`, which macOS does not treat as a
        real application bundle.

        TODO (Real App Flow):
            When a real `.app` exists:
              - Do NOT kill the running app
              - Instead call:
                    open -a PrecursorApp --args --project "<name>"
              - OR trigger:
                    open precursor://open?project=<name>
              - Then handle routing in Swift via onOpenURL.
        """

        swift_root = self._resolve_precursor_swift_root()

        execute_cmd = (
            f'/usr/bin/env zsh -lc "'
            f'pkill -x PrecursorApp || true; '  # ignore error if not running
            f'cd \\"{swift_root}\\" && '
            f'swift run PrecursorApp --project \\"{project_name}\\" '
            f'>> /tmp/precursor_app.log 2>&1 &"'
        )

        logger.info("ui_manager: sending macOS notification for project %s", project_name)
        subprocess.run(
            [
                "terminal-notifier",
                "-title",
                f"{project_name}",
                "-message",
                f"Click to check out what I worked on for \"{project_name}\" while you were gone.",
                "-execute",
                execute_cmd,
            ],
            check=True,
        )

    def run_for_project(
        self,
        project_name: str,
        *,
        user_profile: str = "",
        project_description: Optional[str] = None,
        user_agent_goals: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        For UIManager, interpret as a project-return notification and also
        trigger a macOS notification that can launch the Swift UI.

        (This preserves the Manager protocol and existing return shape.)
        """
        logger.info("ui_manager: run_for_project (notify project return) → %s", project_name)

        # Only notify if there are pending agent-completed tasks to review.
        if self._has_pending_agent_tasks(project_name):
            try:
                self._notify_precursor_for_project(project_name)
                # Emit telemetry event
                if self.telemetry_sink:
                    self.telemetry_sink.emit("notification_sent", {
                        "project": project_name,
                        "reason": "pending_agent_tasks"
                    })
                # Send notification event
                if self.notification_sink:
                    self.notification_sink.send_notification(
                        project_name,
                        "project_return_if_pending",
                        f"Welcome back to {project_name}."
                    )
            except Exception:
                logger.exception("ui_manager: failed to send macOS notification")
        else:
            logger.info(
                "ui_manager: skipping notification for %s (no pending agent-completed tasks)",
                project_name,
            )
            # Emit telemetry event for skipped notification
            if self.telemetry_sink:
                self.telemetry_sink.emit("notification_skipped", {
                    "project": project_name,
                    "reason": "no_pending_agent_tasks"
                })
            # Send notification skip event
            if self.notification_sink:
                self.notification_sink.skip_notification(project_name, "no_pending_agent_tasks")

        return {
            "project": project_name,
            "notification": {
                "type": "project_return_if_pending",
                "message": f"Welcome back to {project_name}.",
            },
        }