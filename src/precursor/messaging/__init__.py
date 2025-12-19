"""
Messaging / conversation history persistence for Precursor.

This package stores per-project conversation messages (user <-> agent) in the same
SQLite DB file as the scratchpad (controlled by PRECURSOR_SCRATCHPAD_DB), but in
its own table.
"""

from __future__ import annotations


