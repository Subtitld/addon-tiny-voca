"""tiny-voca for the add-on: llama.cpp's server, started once, spoken to over HTTP.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Runtime
-------
tiny-voca (4.5M parameters, Apache-2.0) runs in llama.cpp (MIT), which also
turns the text into phonemes with the model's `tts` template, a port of
tiny-voca's phonemizer: numbers, dates, times, money and units are read out
as words. tools/build_native.py builds llama.cpp's server, `llama-server`,
for the CPU only; it is in build/assets/llama/ from source and `llama/`
inside the frozen bundle. The model, two GGUF files (26 MB), is shipped
beside it in `models/`: nothing is downloaded.

The server
----------
The first line starts `llama-server` on 127.0.0.1, on a free port, and waits
for it to answer; every line after that is one request to its `/tts`
endpoint, which returns 24 kHz 16-bit PCM. A line takes a few hundredths of a
second, so a cancel waits for it rather than interrupting it. The server is
stopped when the add-on exits; on Linux it also dies with the add-on (the
parent-death signal), and on Windows with the job object it is assigned to,
so a crashed add-on does not leave it running.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from array import array
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger('tiny_voca_addon.engine')

_PACKAGE = Path(__file__).resolve().parent

MODEL_FILE = 'tiny-voca-F32.gguf'
MMPROJ_FILE = 'mmproj-tiny-voca-F32.gguf'
SAMPLE_RATE = 24000
SERVER = 'llama-server.exe' if sys.platform == 'win32' else 'llama-server'

# The model's voices, by the name the server knows them by.
VOICES = ('F2', 'M2')

START_TIMEOUT_SEC = 60
REQUEST_TIMEOUT_SEC = 120
# Threads: tiny-voca is small, and more than four rarely makes a line faster.
MAX_THREADS = 4


class EngineError(Exception):
    """The server could not be started, or failed a line."""


class Cancelled(Exception):
    """The request was cancelled."""


def assets_dir() -> Path:
    """Where llama/ and models/ are: $TINY_VOCA_ASSETS, the frozen bundle's, or build/assets/."""
    override = (os.environ.get('TINY_VOCA_ASSETS') or '').strip()
    if override:
        return Path(os.path.expanduser(override))
    if getattr(sys, 'frozen', False):
        return Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent))
    return _PACKAGE.parent / 'build' / 'assets'


def server_path() -> Path:
    return assets_dir() / 'llama' / SERVER


def model_paths() -> tuple[Path, Path]:
    models = assets_dir() / 'models'
    return models / MODEL_FILE, models / MMPROJ_FILE


def default_threads() -> int:
    return max(1, min(MAX_THREADS, os.cpu_count() or 2))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def _child_env() -> dict:
    """Our environment, without the frozen bootloader's library path (the server finds its own)."""
    env = dict(os.environ)
    if getattr(sys, 'frozen', False) and sys.platform.startswith('linux'):
        original = env.get('LD_LIBRARY_PATH_ORIG')
        if original is not None:
            env['LD_LIBRARY_PATH'] = original
        else:
            env.pop('LD_LIBRARY_PATH', None)
    return env


def _die_with_parent() -> None:
    """Linux: have the kernel send the server SIGTERM when the add-on goes away.

    The signal comes when the *thread* that started the server ends, not the
    process: so the server is started from _Launcher's thread, which lives as
    long as the add-on, never from a request's worker thread.
    """
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(1, 15)  # PR_SET_PDEATHSIG, SIGTERM
    except (OSError, AttributeError):
        pass


class _WindowsJob:
    """A job object that kills the processes in it when the add-on's last handle to it closes."""

    def __init__(self):
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

        class _Basic(ctypes.Structure):
            _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64),
                        ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                        ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                        ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD),
                        ('SchedulingClass', wintypes.DWORD)]

        class _Io(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                'ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
                'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]

        class _Extended(ctypes.Structure):
            _fields_ = [('BasicLimitInformation', _Basic), ('IoInfo', _Io),
                        ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                        ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]

        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel32 = kernel32
        self.handle = kernel32.CreateJobObjectW(None, None)
        info = _Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        kernel32.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info))

    def add(self, process: subprocess.Popen) -> None:
        self.kernel32.AssignProcessToJobObject(self.handle, int(process._handle))


class _Launcher:
    """One thread that lives as long as the add-on, and starts every server process."""

    def __init__(self):
        import queue
        self.requests: queue.Queue = queue.Queue()
        threading.Thread(target=self._serve, name='tiny-voca-launcher', daemon=True).start()

    def _serve(self) -> None:
        while True:
            args, kwargs, done = self.requests.get()
            try:
                done['process'] = subprocess.Popen(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 — handed back to the caller
                done['error'] = exc
            done['event'].set()

    def popen(self, *args, **kwargs) -> subprocess.Popen:
        done: dict = {'event': threading.Event()}
        self.requests.put((args, kwargs, done))
        done['event'].wait()
        if 'error' in done:
            raise done['error']
        return done['process']


_launcher: _Launcher | None = None


@dataclass
class Server:
    process: subprocess.Popen
    port: int
    threads: int
    log_path: Path

    @property
    def url(self) -> str:
        return f'http://127.0.0.1:{self.port}'


_server: Server | None = None
_server_lock = threading.Lock()
_job = None


def _log_tail(path: Path, lines: int = 8) -> str:
    try:
        return ' / '.join(path.read_text(encoding='utf-8', errors='replace').strip().splitlines()[-lines:])
    except OSError:
        return ''


def _get(url: str, timeout: float = 2.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def start(threads: int = 0) -> Server:
    """The running server, started on the first call."""
    global _server, _job
    with _server_lock:
        if _server is not None and _server.process.poll() is None:
            return _server
        binary = server_path()
        model, mmproj = model_paths()
        missing = [str(p) for p in (binary, model, mmproj) if not p.is_file()]
        if missing:
            raise EngineError(f'not installed completely: {", ".join(missing)} missing')
        # Subtitld unpacks an add-on without permission bits and marks only the
        # add-on's own executable as one; the server needs the same.
        if sys.platform != 'win32' and not os.access(binary, os.X_OK):
            try:
                os.chmod(binary, binary.stat().st_mode | 0o755)
            except OSError as exc:
                raise EngineError(f'{binary} is not executable and could not be made so: {exc}') from exc
        threads = threads if threads > 0 else default_threads()
        port = _free_port()
        log_path = Path(tempfile.gettempdir()) / f'tiny-voca-server-{os.getpid()}.log'
        command = [str(binary), '-m', str(model), '-mm', str(mmproj), '--host', '127.0.0.1', '--port', str(port),
                   '-t', str(threads), '--no-webui', '-np', '1']
        kwargs: dict = {}
        if sys.platform.startswith('linux'):
            kwargs['preexec_fn'] = _die_with_parent
        if sys.platform == 'win32':
            kwargs['creationflags'] = 0x08000000  # CREATE_NO_WINDOW
        log.info('starting %s on port %d with %d threads', binary.name, port, threads)
        global _launcher
        if _launcher is None:
            _launcher = _Launcher()
        with log_path.open('wb') as log_file:
            process = _launcher.popen(command, stdin=subprocess.DEVNULL, stdout=log_file, stderr=log_file,
                                      env=_child_env(), cwd=str(binary.parent), **kwargs)
        if sys.platform == 'win32':
            try:
                if _job is None:
                    _job = _WindowsJob()
                _job.add(process)
            except OSError as exc:
                log.warning('could not tie the server to the add-on: %s', exc)
        started = time.monotonic()
        while time.monotonic() - started < START_TIMEOUT_SEC:
            if process.poll() is not None:
                raise EngineError(f'llama-server stopped as it started (exit {process.returncode}): '
                                  f'{_log_tail(log_path)}')
            try:
                if json.loads(_get(f'http://127.0.0.1:{port}/health')).get('status') == 'ok':
                    break
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(0.1)
        else:
            process.kill()
            raise EngineError(f'llama-server did not answer within {START_TIMEOUT_SEC} s: {_log_tail(log_path)}')
        log.info('llama-server ready in %.1fs', time.monotonic() - started)
        _server = Server(process=process, port=port, threads=threads, log_path=log_path)
        return _server


def stop() -> None:
    """Stop the server, if it runs."""
    global _server
    with _server_lock:
        server, _server = _server, None
    if server is None:
        return
    server.process.terminate()
    try:
        server.process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        server.process.kill()
    try:
        server.log_path.unlink()
    except OSError:
        pass


def synthesize(text: str, voice: str = 'F2', speed: float = 1.0, seed: int = 0, threads: int = 0) -> tuple[array, int]:
    """(float32 mono samples, their sample rate) of `text` spoken in `voice` ('F2' or 'M2')."""
    if voice not in VOICES:
        raise EngineError(f'unknown voice {voice!r}')
    server = start(threads)
    body = json.dumps({'input': text, 'voice': voice, 'speed': round(float(speed), 3),
                       'seed': int(seed) & 0x7FFFFFFF, 'response_format': 'pcm16'}).encode('utf-8')
    request = urllib.request.Request(f'{server.url}/tts', data=body, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
            data = response.read()
            content_type = response.headers.get('Content-Type', '')
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', 'replace')[:300]
        raise EngineError(f'llama-server refused the line (HTTP {exc.code}): {detail}') from exc
    except (urllib.error.URLError, OSError) as exc:
        if server.process.poll() is not None:
            raise EngineError(f'llama-server stopped (exit {server.process.returncode}): '
                              f'{_log_tail(server.log_path)}') from exc
        raise EngineError(f'llama-server did not answer: {exc}') from exc
    rate = SAMPLE_RATE
    for part in content_type.split(';'):
        key, _, value = part.strip().partition('=')
        if key == 'rate' and value.isdigit():
            rate = int(value)
    pcm = array('h')
    pcm.frombytes(data[:len(data) - len(data) % 2])
    if sys.byteorder != 'little':
        pcm.byteswap()
    # The model can peak a little above full scale; the server clips it to 16 bits.
    return array('f', (x / 32768.0 for x in pcm)), rate


def runtime_version() -> str:
    """llama.cpp's build, from native.json beside the runtime."""
    try:
        record = json.loads((assets_dir() / 'native.json').read_text(encoding='utf-8'))
        return record.get('runtime', 'llama.cpp')
    except (OSError, ValueError):
        return 'llama.cpp (unknown build)'
