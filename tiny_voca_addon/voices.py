"""The voices the add-on offers.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

tiny-voca has two voices, both American English, made by its author from
text descriptions (no real speaker was recorded): F2, a woman, and M2, a man.
It does not clone.
"""

from __future__ import annotations

from dataclasses import dataclass

LANGUAGES = ('en',)


@dataclass(frozen=True)
class Voice:
    id: str
    display_name: str
    model_voice: str   # the name llama-server knows it by


VOICES = (
    Voice('tiny-voca-f2', 'Woman (American English)', 'F2'),
    Voice('tiny-voca-m2', 'Man (American English)', 'M2'),
)
BY_ID = {voice.id: voice for voice in VOICES}


def manifest_voices() -> list[dict]:
    """The voice list as manifest.json and the hello frame carry it."""
    return [{'id': voice.id, 'language': 'en', 'display_name': voice.display_name} for voice in VOICES]
