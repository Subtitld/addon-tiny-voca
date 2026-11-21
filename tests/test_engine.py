"""llama-server as built by tools/build_native.py, with the bundled model.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Needs build/assets/ (run tools/build_native.py first).
"""

from __future__ import annotations

import pytest

from tiny_voca_addon import engine


@pytest.fixture(scope='module', autouse=True)
def server():
    yield
    engine.stop()


def test_the_runtime_is_there():
    assert engine.server_path().is_file()
    assert all(path.is_file() for path in engine.model_paths())
    assert engine.runtime_version().startswith('llama.cpp ')


def test_speaks_in_both_voices_and_at_a_pace():
    lengths = {}
    for voice in engine.VOICES:
        samples, rate = engine.synthesize('The meeting moved to 3:30 pm on March 14th.', voice)
        assert rate == 24000
        lengths[voice] = len(samples) / rate
        assert 1.5 < lengths[voice] < 8
        assert max(abs(x) for x in samples) > 0.1
    fast, rate = engine.synthesize('The meeting moved to 3:30 pm on March 14th.', 'F2', speed=1.5)
    assert len(fast) / rate < lengths['F2'] * 0.85


def test_the_server_is_reused_and_restarts_after_it_stops():
    first = engine.start()
    assert engine.start() is first
    first.process.kill()
    first.process.wait()
    samples, _rate = engine.synthesize('Back again.', 'M2')
    assert engine.start() is not first and len(samples) > 0


def test_an_unknown_voice_is_refused():
    with pytest.raises(engine.EngineError):
        engine.synthesize('Hi.', 'Z9')


def test_a_server_without_permission_bits_is_made_executable(tmp_path, monkeypatch):
    import os
    import shutil
    import sys
    if sys.platform == 'win32':
        pytest.skip('no permission bits on Windows')
    engine.stop()
    copy = tmp_path / 'assets'
    shutil.copytree(engine.assets_dir(), copy)
    server = copy / 'llama' / engine.SERVER
    server.chmod(0o644)
    monkeypatch.setenv('TINY_VOCA_ASSETS', str(copy))
    samples, _rate = engine.synthesize('Permissions restored.', 'F2')
    assert len(samples) > 0 and os.access(server, os.X_OK)
    engine.stop()


def test_a_server_started_from_a_finished_thread_keeps_running():
    """Linux's parent-death signal follows the thread that started the server, not the process."""
    import threading
    import time
    engine.stop()
    worker = threading.Thread(target=lambda: engine.synthesize('From a worker thread.', 'F2'))
    worker.start()
    worker.join()
    time.sleep(0.5)
    samples, _rate = engine.synthesize('Still answering.', 'M2')
    assert len(samples) > 0
