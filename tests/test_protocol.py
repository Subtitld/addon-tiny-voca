"""The tts.synthesize handler, with a stand-in for the engine.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path

from tiny_voca_addon import voices

REPO = Path(__file__).resolve().parents[1]


def _terminal(frames):
    terminal = [f for f in frames if f.get('type') in ('result', 'error')]
    assert len(terminal) == 1, frames
    return terminal[0]


def test_a_line_in_each_voice(run, tmp_path):
    for voice in voices.VOICES:
        out = tmp_path / f'{voice.id}.wav'
        frames, fake = run({'text': '  Hello,   world ', 'voice': voice.id, 'output_path': str(out), 'seed': 7})
        result = _terminal(frames)
        assert result['type'] == 'result', result
        text, model_voice, kwargs = fake.calls[-1]
        assert text == 'Hello, world' and model_voice == voice.model_voice
        assert kwargs['speed'] == 1.0 and kwargs['seed'] == 7
        data = result['data']
        assert data['sample_rate'] == 24000 and data['channels'] == 1 and data['voice'] == voice.id
        with wave.open(str(out)) as clip:
            assert (clip.getnchannels(), clip.getsampwidth(), clip.getframerate(), clip.getnframes()) == (1, 2, 24000, 2400)
        values = [f['value'] for f in frames if f['type'] == 'progress']
        assert values == sorted(values) and values[-1] == 1.0


def test_speed_setting_and_rate(run, tmp_path, monkeypatch):
    base = {'text': 'Hi', 'voice': 'tiny-voca-f2', 'output_path': str(tmp_path / 'a.wav')}
    monkeypatch.setenv('TINY_VOCA_SPEED', '1.2')
    _frames, fake = run(dict(base, rate=50))
    assert abs(fake.calls[-1][2]['speed'] - 1.8) < 1e-9
    _frames, fake = run(dict(base, rate=-300))
    assert fake.calls[-1][2]['speed'] == 0.25
    monkeypatch.setenv('TINY_VOCA_SPEED', 'fast')
    _frames, fake = run(dict(base, rate='slow'))
    assert fake.calls[-1][2]['speed'] == 1.0


def test_threads_reach_the_engine(run, tmp_path, monkeypatch):
    monkeypatch.setenv('TINY_VOCA_THREADS', '2')
    _frames, fake = run({'text': 'Hi', 'voice': 'tiny-voca-m2', 'output_path': str(tmp_path / 'a.wav')})
    assert fake.calls[-1][2]['threads'] == 2


def test_unknown_voice_and_missing_fields(run, tmp_path):
    error = _terminal(run({'text': 'Hi', 'voice': 'qwen3-ryan', 'output_path': str(tmp_path / 'a.wav')})[0])
    assert error['code'] == 'unsupported_voice'
    assert _terminal(run({'text': '   ', 'voice': 'tiny-voca-f2', 'output_path': 'x.wav'})[0])['code'] == 'bad_params'
    assert _terminal(run({'text': 'Hi', 'voice': 'tiny-voca-f2'})[0])['code'] == 'bad_params'


def test_a_cancelled_request_speaks_nothing(run, tmp_path):
    out = tmp_path / 'a.wav'
    frames, fake = run({'text': 'Hi', 'voice': 'tiny-voca-f2', 'output_path': str(out)}, cancelled=True)
    assert _terminal(frames)['code'] == 'cancelled'
    assert not fake.calls and not out.exists()


def test_an_engine_failure_is_reported(run, tmp_path, monkeypatch):
    from tiny_voca_addon import engine

    def broken(*args, **kwargs):
        raise engine.EngineError('llama-server stopped (exit 1)')

    monkeypatch.setattr(engine, 'synthesize', broken)
    error = _terminal(run({'text': 'Hi', 'voice': 'tiny-voca-f2', 'output_path': str(tmp_path / 'a.wav')})[0])
    assert error['code'] == 'internal' and 'stopped' in error['message'] and error['retryable'] is True


def test_the_process_says_hello_and_leaves_on_shutdown():
    env = dict(os.environ, PYTHONPATH=str(REPO))
    proc = subprocess.Popen([sys.executable, '-m', 'tiny_voca_addon'], cwd=REPO, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        hello = json.loads(proc.stdout.readline())
        proc.stdin.write(b'{"type":"ready","host":"test"}\n')
        proc.stdin.write(b'{"id":"x","type":"asr.transcribe","params":{}}\n')
        proc.stdin.flush()
        error = json.loads(proc.stdout.readline())
        proc.stdin.write(b'{"type":"shutdown"}\n')
        proc.stdin.flush()
        assert proc.wait(timeout=20) == 0
    finally:
        proc.kill()
    assert hello['type'] == 'hello' and hello['addon'] == 'tiny-voca'
    capability = hello['capabilities'][0]
    assert capability['task'] == 'tts.synthesize' and capability['voice_clone'] is False
    assert capability['languages'] == ['en']
    assert {v['id'] for v in capability['voices']} == set(voices.BY_ID)
    assert error['code'] == 'bad_params'
