#!/usr/bin/env bash
# Run tests/test_installed.py against the release archive, in a clean container.
#
# Copyright 2026 Subtitld contributors
# SPDX-License-Identifier: Apache-2.0
#
# The archive is unpacked the way Subtitld's installer does it (zipfile's
# extractall, then the executable bit), into a distribution that has none of the
# build's packages. The container gets only what the tests drive the binary
# with: Python and pytest. The model is in the archive, so every test speaks
# for real.
#
#     tools/test_archive.sh almalinux:8
#     tools/test_archive.sh ubuntu:20.04
#
# Needs Docker and the network.

set -euo pipefail

IMAGE="${1:?usage: tools/test_archive.sh <image> [archive]}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(sed -n 's/^__version__ = .\(.*\).$/\1/p' "$ROOT/tiny_voca_addon/__init__.py")"
ARCHIVE="$(realpath "${2:-$ROOT/tiny-voca-${VERSION}-linux-x86_64.zip}")"
[ -f "$ARCHIVE" ] || { echo "no archive at $ARCHIVE; run tools/build_linux.sh" >&2; exit 1; }

MOUNTS=(-v "$ARCHIVE:/artifact/addon.zip:ro" -v "$ROOT/tests:/repo/tests:ro"
        -v "$ROOT/tiny_voca_addon:/repo/tiny_voca_addon:ro")
ENVS=(-e "PYTEST_MARK=")

# Installing packages needs root; the tests then run as you, so what the
# add-on writes into the model folder stays yours.
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -e "TEST_UID=$(id -u)" -e "TEST_GID=$(id -g)" \
    "${ENVS[@]}" "${MOUNTS[@]}" "$IMAGE" \
    bash -euo pipefail -c '
    PY=python3
    . /etc/os-release
    if command -v apt-get >/dev/null; then
        apt-get update -qq >/dev/null
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ca-certificates python3 python3-pytest procps >/dev/null
    elif [ "${PLATFORM_ID:-}" = platform:el8 ]; then
        # EL8 python3 is 3.6, and its pytest 3.4: the tests need 3.8 or newer.
        dnf install -y -q python3.12 python3.12-pip procps-ng >/dev/null
        python3.12 -m pip install -q --root-user-action=ignore pytest
        PY=python3.12
    elif command -v dnf >/dev/null; then
        dnf install -y -q python3 python3-pytest >/dev/null
    elif command -v pacman >/dev/null; then
        pacman -Sy --noconfirm archlinux-keyring >/dev/null
        pacman -Syu --noconfirm --needed python python-pytest >/dev/null
    else
        echo "no supported package manager in this image" >&2; exit 1
    fi
    echo "image: $PRETTY_NAME, $(ldd --version | head -1), test driver $($PY --version)"
    $PY -c "
import os, stat, zipfile
zipfile.ZipFile(\"/artifact/addon.zip\").extractall(\"/opt/addon\")
exe = \"/opt/addon/bin/tiny-voca-addon\"
os.chmod(exe, os.stat(exe).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
"
    # Subtitld unpacks add-ons into a folder of the user who runs them: so does this.
    chown -R "$TEST_UID:$TEST_GID" /opt/addon
    mkdir -p /tmp/run && chown "$TEST_UID:$TEST_GID" /tmp/run && cd /tmp/run
    setpriv --reuid="$TEST_UID" --regid="$TEST_GID" --clear-groups \
        env HOME=/tmp/run TINY_VOCA_TEST_BINARY=/opt/addon/bin/tiny-voca-addon \
        $PY -m pytest -q -rs -p no:cacheprovider ${PYTEST_MARK:+-m "$PYTEST_MARK"} /repo/tests/test_installed.py
'
