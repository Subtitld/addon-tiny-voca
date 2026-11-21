"""The frozen binary, unpacked from its release archive the way Subtitld installs it.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

$TINY_VOCA_TEST_BINARY names the binary; without it these tests skip. The
release workflow sets it on every platform. The model is part of the add-on,
so every test here speaks for real: the archive is unpacked as Subtitld does
it (no permission bits but the add-on's own), and its llama-server must still
start.

    TINY_VOCA_TEST_BINARY=... pytest tests/test_installed.py
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import wave
from pathlib import Path

import pytest

BINARY = os.environ.get('TINY_VOCA_TEST_BINARY', '')
pytestmark = pytest.mark.skipif(not BINARY, reason='TINY_VOCA_TEST_BINARY is not set')


class Addon:
    """The binary on the protocol, with a watchdog so a hang fails instead of stalling CI."""

    def __init__(self, tmp_path: Path, timeout: float):
        env = dict(os.environ, SUBTITLD_TEMP_DIR=str(tmp_path))
        self.proc = subprocess.Popen([BINARY], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, env=env)
        self.timer = threading.Timer(timeout, self.proc.kill)
        self.timer.start()

    def send(self, frame: dict) -> None:
        self.proc.stdin.write((json.dumps(frame) + '\n').encode())
        self.proc.stdin.flush()

    def read(self) -> dict:
        line = self.proc.stdout.readline()
        assert line, 'the add-on closed its output (crashed or timed out)'
        return json.loads(line)

    def until_terminal(self, req_id: str, on_progress=None) -> dict:
        while True:
            frame = self.read()
            if frame.get('id') != req_id:
                continue
            if frame['type'] in ('result', 'error'):
                return frame
            if on_progress is not None:
                on_progress(frame)

    def close(self) -> int:
        try:
            self.send({'type': 'shutdown'})
        except (BrokenPipeError, OSError):
            pass
        try:
            return self.proc.wait(timeout=15)
        finally:
            self.timer.cancel()
            self.proc.kill()


def test_version():
    out = subprocess.run([BINARY, '--version'], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.startswith('tiny-voca-addon ')


def test_self_test_speaks_both_voices():
    out = subprocess.run([BINARY, '--self-test'], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    report = json.loads(out.stdout.strip().splitlines()[-1])
    assert report['ok'] is True and report['runtime'].startswith('llama.cpp ')
    assert set(report['voices']) == {'tiny-voca-f2', 'tiny-voca-m2'}


def test_handshake_and_refusals(tmp_path):
    addon = Addon(tmp_path, timeout=120)
    try:
        hello = addon.read()
        assert hello['type'] == 'hello' and hello['addon'] == 'tiny-voca'
        addon.send({'type': 'ready', 'host': 'test'})
        addon.send({'id': 'a', 'type': 'tts.synthesize',
                    'params': {'text': 'Hi', 'voice': 'nope', 'output_path': str(tmp_path / 'a.wav')}})
        assert addon.until_terminal('a')['code'] == 'unsupported_voice'
    finally:
        assert addon.close() == 0


def test_speaks_lines_in_both_voices(tmp_path):
    addon = Addon(tmp_path, timeout=300)
    try:
        addon.read()
        lengths = {}
        for voice, rate in (('tiny-voca-f2', 0), ('tiny-voca-m2', 0), ('tiny-voca-f2', 50)):
            out = tmp_path / f'{voice}-{rate}.wav'
            addon.send({'id': out.stem, 'type': 'tts.synthesize', 'params': {
                'text': 'The meeting moved to 3:30 pm on March 14th, so we have about 45 minutes left.',
                'voice': voice, 'rate': rate, 'output_path': str(out)}})
            result = addon.until_terminal(out.stem)
            assert result['type'] == 'result', result
            assert result['data']['sample_rate'] == 24000
            with wave.open(str(out)) as clip:
                assert clip.getframerate() == 24000
                lengths[out.stem] = clip.getnframes() / 24000
        assert all(2.5 < seconds < 9 for seconds in lengths.values()), lengths
        assert lengths['tiny-voca-f2-50'] < lengths['tiny-voca-f2-0'] * 0.85
    finally:
        assert addon.close() == 0


def test_no_server_is_left_running_after_shutdown(tmp_path):
    addon = Addon(tmp_path, timeout=120)
    try:
        addon.read()
        addon.send({'id': 'a', 'type': 'tts.synthesize', 'params': {
            'text': 'One line, to start the server.', 'voice': 'tiny-voca-m2', 'output_path': str(tmp_path / 'a.wav')}})
        assert addon.until_terminal('a')['type'] == 'result'
    finally:
        assert addon.close() == 0
    import time
    server = str(Path(BINARY).parent / '_internal' / 'llama')
    for _ in range(30):
        if os.name == 'nt':
            running = 'llama-server.exe' in subprocess.run(['tasklist'], capture_output=True, text=True).stdout
        else:
            running = server in subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True).stdout
        if not running:
            break
        time.sleep(0.2)
    assert not running, 'llama-server outlived the add-on'
