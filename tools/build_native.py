"""Build llama.cpp's server for tiny-voca and put it, with the model, into build/assets/.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

tiny-voca runs in llama.cpp, which reads its GGUF model and turns text into
phonemes with the model's own `tts` template, a port of tiny-voca's phonemizer
(dizione). The add-on drives llama.cpp's server, `llama-server`, over HTTP on
127.0.0.1 (its `/tts` endpoint). This script checks out ggml-org/llama.cpp at
a pinned commit, builds `llama-server` with its libraries, fetches the pinned
model files, and collects:

- build/assets/llama/: llama-server and, on Linux and macOS, the libraries it
  loads (llama, mtmd, ggml and, on Linux, one ggml CPU backend per x86-64
  instruction set, of which ggml loads the best the processor runs), each
  finding the others in its own folder. On Windows, one llama-server.exe with
  all of it linked in (see WINDOWS_OPTIONS).
- build/assets/models/: tiny-voca's two GGUF files, checked by sha256.
- build/assets/native.json: where it all came from, and each file's SHA-256.
- build/licences/llama/: the licences of what is compiled in.

The server is built without its chat UI (nothing downloaded at build time)
and without HTTPS: it only ever listens on 127.0.0.1. ggml runs on its own
thread pool, without OpenMP, and on the CPU only: tiny-voca is small enough
that a GPU would not make a line noticeably faster.

Linux builds run in the AlmaLinux 8 container (tools/build_native_linux.sh),
so the libraries load on glibc 2.28. macOS builds are for Apple Silicon and
macOS 13.3 or newer. Windows builds need Visual Studio's C++ tools, and link
Microsoft's C runtime statically. Haiku builds run on Haiku, against the
system's libroot and libstdc++ (on every install), with the same libraries
as on Linux.

    python tools/build_native.py [--jobs N]
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / 'build'
ASSETS = BUILD / 'assets'
RUNTIME = ASSETS / 'llama'
MODELS = ASSETS / 'models'
LICENCES = BUILD / 'licences' / 'llama'
SRC = BUILD / 'llama-src'
NATIVE_BUILD = BUILD / 'native'
DOWNLOADS = BUILD / 'downloads'

REPO = 'https://github.com/ggml-org/llama.cpp'
# main after the v1.0.0-b12359 release: the commit that added tiny-voca
# (#32427, 2026-11-17); no release has it yet.
COMMIT = '3574df84b776611928d9b227bf614be71fbb9849'

MODEL_REPO = 'ggml-org/tiny-voca-GGUF'
MODEL_REVISION = '8a785351a6dac0b9168bc14bc5de635a98a7b809'
MODEL_FILES = (
    ('tiny-voca-F32.gguf', 12169280, '6cc38fe1a32ac312d906da6549a142de8cb6a2d0d1f10e63966a8f7c49be72cc'),
    ('mmproj-tiny-voca-F32.gguf', 13641888, 'c70b6ba351335bb72bdd1ae42412f64490816c3accaec2a2c2990fbcfc33c48c'),
)

CMAKE_OPTIONS = [
    '-DBUILD_SHARED_LIBS=ON',
    '-DLLAMA_BUILD_TESTS=OFF',
    '-DLLAMA_BUILD_EXAMPLES=OFF',
    '-DLLAMA_BUILD_TOOLS=ON',
    '-DLLAMA_BUILD_SERVER=ON',
    '-DLLAMA_BUILD_APP=OFF',
    '-DLLAMA_BUILD_UI=OFF',
    '-DLLAMA_USE_PREBUILT_UI=OFF',
    '-DLLAMA_CONNECT=OFF',
    '-DLLAMA_OPENSSL=OFF',
    '-DGGML_NATIVE=OFF',
    '-DGGML_OPENMP=OFF',
    '-DGGML_METAL=OFF',
    '-DGGML_CUDA=OFF',
    '-DGGML_VULKAN=OFF',
]
LINUX_OPTIONS = ['-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DGGML_BACKEND_DL=ON', '-DGGML_CPU_ALL_VARIANTS=ON',
                 '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON', '-DCMAKE_INSTALL_RPATH=$ORIGIN']
# Haiku: as Linux. Sockets are in libnetwork, which llama.cpp's CMake links
# only into the server, not into libllama-common, whose downloader uses them.
HAIKU_OPTIONS = [*LINUX_OPTIONS, '-DCMAKE_SHARED_LINKER_FLAGS=-lnetwork', '-DCMAKE_EXE_LINKER_FLAGS=-lnetwork']
# Windows: one llama-server.exe with llama.cpp and ggml linked in. Built as
# DLLs with the C runtime linked statically, each DLL would have its own heap,
# and llama.cpp's libraries pass C++ strings and vectors between each other: the
# server aborts as it loads the model (0xC0000409). One executable has one
# heap. Without separate backend DLLs, ggml's CPU code is built once, for
# SSE4.2, which every x86-64 PC Windows runs on has: tiny-voca is small enough
# that it is still far faster than real time.
WINDOWS_OPTIONS = ['-A', 'x64', '-DBUILD_SHARED_LIBS=OFF', '-DGGML_BACKEND_DL=OFF', '-DGGML_CPU_ALL_VARIANTS=OFF',
                   '-DGGML_AVX=OFF', '-DGGML_AVX2=OFF', '-DGGML_FMA=OFF', '-DGGML_F16C=OFF', '-DGGML_BMI2=OFF',
                   '-DGGML_AVX512=OFF', '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded',
                   '-DCMAKE_POLICY_DEFAULT_CMP0091=NEW']
MACOS_OPTIONS = ['-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_OSX_ARCHITECTURES=arm64',
                 '-DCMAKE_OSX_DEPLOYMENT_TARGET=13.3', '-DGGML_BLAS=OFF',
                 '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON', '-DCMAKE_INSTALL_RPATH=@loader_path']
MACOS_MIN = (13, 3)

SERVER = {'linux': 'llama-server', 'macos': 'llama-server', 'windows': 'llama-server.exe', 'haiku': 'llama-server'}

# Libraries every Linux desktop has; the build may link nothing else dynamically.
LINUX_SYSTEM = {
    'libc.so.6', 'libm.so.6', 'libdl.so.2', 'libpthread.so.0', 'librt.so.1',
    'ld-linux-x86-64.so.2', 'libstdc++.so.6', 'libgcc_s.so.1',
}
# The newest symbol version each AlmaLinux 8 (glibc 2.28, GCC 8) library provides.
LINUX_CEILINGS = {'GLIBC': (2, 28), 'GLIBCXX': (3, 4, 25), 'CXXABI': (1, 3, 11), 'GCC': (8, 0, 0)}
# Haiku: libroot is its C library; libstdc++ and libgcc_s come with every
# install (gcc_syslibs), and libnetwork holds the sockets.
HAIKU_SYSTEM = {'libroot.so', 'libstdc++.so.6', 'libgcc_s.so.1', 'libnetwork.so', 'libbsd.so'}
WINDOWS_SYSTEM = {'kernel32.dll', 'user32.dll', 'advapi32.dll', 'shell32.dll', 'ole32.dll', 'oleaut32.dll',
                  'ws2_32.dll', 'bcrypt.dll', 'ntdll.dll', 'shlwapi.dll', 'dbghelp.dll', 'psapi.dll',
                  'version.dll', 'winmm.dll', 'crypt32.dll', 'secur32.dll', 'userenv.dll', 'mswsock.dll'}

# What llama-server compiles in besides llama.cpp and ggml (one licence,
# llama.cpp's MIT, covers both), and the licence files in the llama.cpp tree.
# Single-header libraries carry their licence at the end of the header; it is
# copied from there.
VENDORED = {
    'cpp-httplib': ('file', 'vendor/cpp-httplib/LICENSE'),
    'nlohmann-json': ('header', 'vendor/nlohmann/json.hpp'),
    'miniaudio': ('header', 'vendor/miniaudio/miniaudio.h'),
    'stb_image': ('header', 'vendor/stb/stb_image.h'),
    'subprocess.h': ('header', 'vendor/sheredom/subprocess.h'),
    'hash': ('dir', 'vendor/hash'),
}


def run(cmd, cwd=None, env=None):
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True)


def output(cmd, cwd=None):
    env = dict(os.environ, LC_ALL='C')
    return subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True,
                          capture_output=True, text=True).stdout


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fresh(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


GIT = ['git', '-c', 'core.autocrlf=false', '-c', 'core.eol=lf']


def checkout():
    """The pinned commit, shallow."""
    if not (SRC / '.git').exists():
        SRC.mkdir(parents=True, exist_ok=True)
        run([*GIT, 'init', '-q'], cwd=SRC)
        run([*GIT, 'remote', 'add', 'origin', REPO + '.git'], cwd=SRC)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=SRC, capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        run([*GIT, 'fetch', '-q', '--depth', '1', 'origin', COMMIT], cwd=SRC)
    run([*GIT, 'checkout', '-q', '--force', COMMIT], cwd=SRC)
    run([*GIT, 'clean', '-q', '-fd'], cwd=SRC)


def apply_patches(platform):
    """tools/patches/<platform>/*.patch, in order: what llama.cpp needs to build there."""
    for patch in sorted((ROOT / 'tools' / 'patches' / platform).glob('*.patch')):
        run([*GIT, 'apply', '--verbose', patch], cwd=SRC)


def fetch_models():
    """The pinned GGUF files, each checked against its sha256."""
    fresh(MODELS)
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    for name, size, digest in MODEL_FILES:
        cached = DOWNLOADS / name
        if not (cached.is_file() and cached.stat().st_size == size and sha256(cached) == digest):
            url = f'https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}'
            print('+ fetch', url, flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, cached.open('wb') as out:
                shutil.copyfileobj(response, out)
        if cached.stat().st_size != size or sha256(cached) != digest:
            sys.exit(f'FATAL: {name} does not match its pinned size and sha256')
        shutil.copyfile(cached, MODELS / name)


def build(platform, jobs):
    configure = ['cmake', '-S', SRC, '-B', NATIVE_BUILD, *CMAKE_OPTIONS]
    configure += {'linux': LINUX_OPTIONS, 'windows': WINDOWS_OPTIONS, 'macos': MACOS_OPTIONS,
                  'haiku': HAIKU_OPTIONS}[platform]
    run(configure)
    run(['cmake', '--build', NATIVE_BUILD, '--config', 'Release', '--parallel', str(jobs), '--target', 'llama-server'])
    found = [p for p in NATIVE_BUILD.rglob(SERVER[platform]) if p.is_file() and 'CMakeFiles' not in p.parts]
    if len(found) != 1:
        sys.exit(f'FATAL: expected one {SERVER[platform]} under {NATIVE_BUILD}, found {found}')
    return found[0].parent


def soname(lib):
    m = re.search(r'\(SONAME\)\s+Library soname: \[([^\]]+)\]', output(['readelf', '-d', lib]))
    return m.group(1) if m else lib.name


def install_name(lib):
    """macOS: the name a dylib is loaded by (its LC_ID_DYLIB), without the @rpath/ part."""
    lines = output(['otool', '-D', lib]).strip().splitlines()
    return lines[-1].rsplit('/', 1)[-1] if len(lines) > 1 else lib.name


def collect(platform, built):
    """llama-server and every library beside it, each under the name it is loaded by
    (Linux: its SONAME; macOS: its install name)."""
    fresh(RUNTIME)
    shutil.copyfile(built / SERVER[platform], RUNTIME / SERVER[platform])
    os.chmod(RUNTIME / SERVER[platform], 0o755)
    pattern = {'linux': '*.so*', 'haiku': '*.so*', 'macos': '*.dylib', 'windows': '*.dll'}[platform]
    for lib in sorted(built.glob(pattern)):
        if lib.is_symlink():
            continue
        name = {'linux': soname, 'haiku': soname, 'macos': install_name}.get(platform, lambda path: path.name)(lib)
        shutil.copyfile(lib, RUNTIME / name)


# ---- Linux: refuse anything AlmaLinux 8 couldn't load ---------------------

def needed(lib):
    return re.findall(r'\(NEEDED\)\s+Shared library: \[([^\]]+)\]', output(['readelf', '-d', lib]))


def version_needs(lib):
    needs, current = {}, None
    for line in output(['readelf', '-V', '--wide', lib]).splitlines():
        m = re.search(r'File: (\S+)', line)
        if m:
            current = needs.setdefault(m.group(1), [])
            continue
        m = re.search(r'Name: (\S+)', line)
        if m and current is not None:
            current.append(m.group(1))
    return needs


def check_linux():
    files = sorted(p for p in RUNTIME.iterdir() if p.is_file())
    shipped = {p.name for p in files}
    problems = []
    for lib in files:
        problems += [f'{lib.name} needs {dep}, which is neither shipped nor a system library'
                     for dep in needed(lib) if dep not in LINUX_SYSTEM | shipped]
        for dep, versions in version_needs(lib).items():
            for version in versions:
                m = re.fullmatch(r'([A-Z]+)_([0-9.]+)', version)
                if not m or m.group(1) not in LINUX_CEILINGS:
                    continue
                if tuple(int(x) for x in m.group(2).split('.')) > LINUX_CEILINGS[m.group(1)]:
                    problems.append(f'{lib.name} needs {version} from {dep}')
        runpath = re.findall(r'\((?:RUNPATH|RPATH)\)\s+Library (?:runpath|rpath): \[([^\]]+)\]',
                             output(['readelf', '-d', lib]))
        if needed(lib) and set(needed(lib)) & shipped and runpath != ['$ORIGIN']:
            problems.append(f'{lib.name} has runpath {runpath}, not $ORIGIN')
    if problems:
        sys.exit('not loadable on glibc 2.28 / GCC 8:\n  ' + '\n  '.join(sorted(set(problems))))
    print('llama-server and its libraries load on glibc 2.28 with the GCC 8 runtime, from one folder')


# ---- Haiku: only the system's libraries, and each other from one folder ---

def check_haiku():
    files = sorted(p for p in RUNTIME.iterdir() if p.is_file())
    shipped = {p.name for p in files}
    problems = []
    for lib in files:
        problems += [f'{lib.name} needs {dep}, which is neither shipped nor a system library'
                     for dep in needed(lib) if dep not in HAIKU_SYSTEM | shipped]
        runpath = re.findall(r'\((?:RUNPATH|RPATH)\)\s+Library (?:runpath|rpath): \[([^\]]+)\]',
                             output(['readelf', '-d', lib]))
        if set(needed(lib)) & shipped and runpath != ['$ORIGIN']:
            problems.append(f'{lib.name} has runpath {runpath}, not $ORIGIN')
    if problems:
        sys.exit('not loadable on Haiku:\n  ' + '\n  '.join(sorted(set(problems))))
    print('llama-server and its libraries need only Haiku system libraries, and find each other in their own folder')


# ---- macOS: refuse anything a Mac with macOS 13.3 couldn't load -----------

def check_macos():
    files = sorted(p for p in RUNTIME.iterdir() if p.is_file())
    shipped = {p.name for p in files}
    problems = []
    for lib in files:
        for line in output(['otool', '-L', lib]).splitlines()[1:]:
            dep = line.strip().split(' (', 1)[0]
            if dep.startswith(('/usr/lib/', '/System/Library/')):
                continue
            if dep.startswith(('@rpath/', '@loader_path/')) and dep.rsplit('/', 1)[-1] in shipped:
                continue
            problems.append(f'{lib.name} needs {dep}, which is neither shipped nor part of macOS')
        for major, minor in re.findall(r'^\s*minos (\d+)\.(\d+)', output(['otool', '-l', lib]), re.M):
            if (int(major), int(minor)) > MACOS_MIN:
                problems.append(f'{lib.name} needs macOS {major}.{minor}')
        if 'arm64' not in output(['lipo', '-archs', lib]):
            problems.append(f'{lib.name} is not built for Apple Silicon')
    if problems:
        sys.exit('not loadable on macOS 13.3:\n  ' + '\n  '.join(sorted(set(problems))))
    print(f'llama-server and its libraries load on macOS {MACOS_MIN[0]}.{MACOS_MIN[1]} and newer, on Apple Silicon')


# ---- Windows: only system DLLs, no C runtime DLL --------------------------

def _pe_imports(path):
    import pefile  # PyInstaller's own dependency on Windows
    pe = pefile.PE(str(path), fast_load=True)
    pe.parse_data_directories([pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']])
    return [entry.dll.decode() for entry in getattr(pe, 'DIRECTORY_ENTRY_IMPORT', [])]


def check_windows():
    files = sorted(p for p in RUNTIME.iterdir() if p.suffix.lower() in ('.dll', '.exe'))
    shipped = {p.name.lower() for p in files}
    problems = []
    for lib in files:
        for dep in _pe_imports(lib):
            low = dep.lower()
            if low in shipped:
                continue
            if re.match(r'(msvcp|vcruntime|vcomp|concrt)\d+', low) or low.startswith('api-ms-win-crt-'):
                problems.append(f'{lib.name} needs {dep}: the C runtime should be linked in statically')
            elif not (low.startswith('api-ms-win-') or low in WINDOWS_SYSTEM):
                problems.append(f'{lib.name} needs {dep}, which is neither shipped nor a system DLL')
    if problems:
        sys.exit('not loadable on a bare Windows:\n  ' + '\n  '.join(sorted(set(problems))))
    print('llama-server and its DLLs need only Windows system DLLs, and find each other in their own folder')


# ---- licences ---------------------------------------------------------------

def _header_licence(path):
    """The licence block of a single-header library: its last comment that names a licence."""
    text = path.read_text(encoding='utf-8', errors='replace')
    blocks = re.findall(r'/\*.*?\*/', text, re.S)
    for block in reversed(blocks):
        if re.search(r'licen[cs]e|public domain|permission is hereby granted', block, re.I):
            return block
    for block in blocks[:3]:
        if re.search(r'licen[cs]e|public domain|permission is hereby granted', block, re.I):
            return block
    head = '\n'.join(line for line in text.splitlines()[:80] if line.lstrip().startswith('//'))
    if re.search(r'licen[cs]e|permission is hereby granted', head, re.I):
        return head
    sys.exit(f'FATAL: no licence text found in {path}')


def collect_licences():
    fresh(LICENCES)
    shutil.copyfile(SRC / 'LICENSE', LICENCES / 'LICENSE')
    for name, (kind, relative) in VENDORED.items():
        folder = LICENCES / 'third_party' / name
        folder.mkdir(parents=True)
        source = SRC / relative
        if kind == 'file':
            shutil.copyfile(source, folder / source.name)
        elif kind == 'header':
            (folder / 'LICENSE.txt').write_text(_header_licence(source) + '\n', encoding='utf-8')
        else:
            notices = []
            for header in sorted(source.rglob('*')):
                if header.is_file() and header.suffix in ('.h', '.hpp', '.c', '.cpp'):
                    head = '\n'.join(header.read_text(encoding='utf-8', errors='replace').splitlines()[:40])
                    if re.search(r'licen[cs]e|public domain|copyright', head, re.I):
                        notices.append(f'--- {header.relative_to(SRC)} ---\n{head}')
            if not notices:
                sys.exit(f'FATAL: no licence notice in {source}')
            (folder / 'LICENSE.txt').write_text('\n\n'.join(notices) + '\n', encoding='utf-8')


def main():
    host = 'haiku' if sys.platform.startswith('haiku') else {'linux': 'linux', 'win32': 'windows',
                                                               'darwin': 'macos'}[sys.platform]
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--jobs', type=int, default=int(os.environ.get('JOBS', os.cpu_count() or 4)))
    args = parser.parse_args()

    checkout()
    apply_patches(host)
    built = build(host, args.jobs)
    collect(host, built)
    {'linux': check_linux, 'macos': check_macos, 'windows': check_windows, 'haiku': check_haiku}[host]()
    collect_licences()
    fetch_models()

    files = {p.relative_to(ASSETS).as_posix(): sha256(p) for p in sorted(ASSETS.rglob('*')) if p.is_file()}
    record = {'runtime': f'llama.cpp {COMMIT[:10]} llama-server, CPU only', 'repo': REPO, 'commit': COMMIT,
              'model': f'{MODEL_REPO}@{MODEL_REVISION}', 'platform': host,
              'entry': f'llama/{SERVER[host]}', 'files': files}
    (ASSETS / 'native.json').write_text(json.dumps(record, indent=1) + '\n')
    size = sum(p.stat().st_size for p in RUNTIME.iterdir()) / 2**20
    print(f'{len(files)} files in {ASSETS}, {size:.1f} MiB of runtime')


if __name__ == '__main__':
    main()
