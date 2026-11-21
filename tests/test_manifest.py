"""manifest.json agrees with the code: voices, languages, settings, version, model.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tiny_voca_addon import __main__ as addon
from tiny_voca_addon import __version__, engine, voices

REPO = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((REPO / 'manifest.json').read_text(encoding='utf-8'))
FIELDS = {field['key']: field for field in MANIFEST['config_schema']['fields']}


def test_identity_and_version():
    assert MANIFEST['id'] == 'tiny-voca'
    assert MANIFEST['executable'] == 'bin/tiny-voca-addon'
    assert MANIFEST['tasks'] == ['tts.synthesize']
    pyproject = (REPO / 'pyproject.toml').read_text(encoding='utf-8')
    assert re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1) == __version__
    assert MANIFEST['version'] == __version__ == addon.ADDON_VERSION


def test_voices_and_languages_match_the_code():
    assert MANIFEST['voices'] == voices.manifest_voices()
    assert MANIFEST['languages'] == ['en'] and MANIFEST['voice_clone'] is False
    assert {v.model_voice for v in voices.VOICES} == set(engine.VOICES)


def test_every_setting_the_code_reads_is_declared():
    assert set(addon._options()) == set(FIELDS)


def test_setting_defaults_are_valid_and_held_to_range(monkeypatch):
    for key, field in FIELDS.items():
        assert field['min'] <= field['default'] <= field['max'], key
        monkeypatch.setenv(f'TINY_VOCA_{key.upper()}', str(field['max'] * 10 + 1))
        assert addon._options()[key] == field['max'], key
        monkeypatch.setenv(f'TINY_VOCA_{key.upper()}', str(field['min'] - 1))
        assert addon._options()[key] == field['min'], key


def test_the_model_is_bundled_and_pinned():
    model = MANIFEST['models'][0]
    assert model['bundled'] is True and model['license'] == 'Apache-2.0'
    build = (REPO / 'tools' / 'build_native.py').read_text(encoding='utf-8')
    assert model['revision'] in build and model['repo'] in build
    assert MANIFEST['compute'] == 'cpu' and 'accelerators' not in MANIFEST
