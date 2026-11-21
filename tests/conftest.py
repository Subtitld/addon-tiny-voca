"""Shared fixtures: the request handler with a stand-in for the engine.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import math
import sys
from array import array
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiny_voca_addon import __main__ as addon  # noqa: E402
from tiny_voca_addon import engine  # noqa: E402


class FakeEngine:
    """Records each line and returns a short tone."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []

    def synthesize(self, text, voice='F2', **kwargs):
        self.calls.append((text, voice, kwargs))
        return array('f', (0.25 * math.sin(i / 10.0) for i in range(2400))), 24000


@pytest.fixture
def run(monkeypatch, tmp_path):
    """run(params) -> (frames, fake): one tts.synthesize request through the handler."""
    frames: list[dict] = []
    fake = FakeEngine()
    monkeypatch.setattr(addon, '_emit', frames.append)
    monkeypatch.setattr(engine, 'synthesize', fake.synthesize)
    for name in ('TINY_VOCA_SPEED', 'TINY_VOCA_THREADS'):
        monkeypatch.delenv(name, raising=False)

    def _run(params: dict, cancelled: bool = False):
        state = addon._Request('req-1')
        if cancelled:
            state.cancel.set()
        frames.clear()
        addon._handle_synthesize(state, params)
        return list(frames), fake
    return _run
