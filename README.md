# tiny-voca add-on for Subtitld

Lightweight English text-to-speech for [Subtitld](https://subtitld.org)
dubbing with [tiny-voca](https://huggingface.co/n4ze3m/tiny-voca), a
4.5M-parameter model: American English at 24 kHz in two voices, a woman and a
man, dozens of times faster than real time on a CPU. Numbers, dates, times,
money and units are read out as words. The model is part of the add-on, so
nothing is downloaded. The add-on and the model are Apache-2.0.

## Voices

- **Woman (American English)**: tiny-voca's F2.
- **Man (American English)**: tiny-voca's M2.

Both were made by tiny-voca's author from text descriptions; no real speaker
was recorded. tiny-voca does not clone voices.

## What it needs

- **Disk:** about 120 MB, the model (26 MB) included.
- **Memory:** a few hundred MB.
- **Time:** a line takes a few hundredths of a second on an Intel Core
  i7-13700T. The first line also starts the speech server, which takes about
  half a second.

## Settings

| Setting | Default | |
|---|---|---|
| Speed | 1.0 | the pace of the voices, 0.5 to 2; a speaker's *Rate* in Subtitld scales it further |
| CPU threads | 0 | up to four, plenty for a model this small |

## How it runs the model

tiny-voca runs in [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT),
which also turns the text into phonemes: the model's GGUF carries a port of
tiny-voca's phonemizer, [dizione](https://github.com/n4ze3m/dizione), with its
pronunciation dictionary and rules. `tools/build_native.py` builds llama.cpp's
server, `llama-server`, from source at a pinned commit (on main after the
v1.0.0-b12359 release: tiny-voca support arrived after it), for the CPU only,
without its chat UI and without HTTPS, and fetches the pinned model files from
ggml-org/tiny-voca-GGUF.

The add-on starts `llama-server` on 127.0.0.1 with the first line and sends
each line to its `/tts` endpoint. It stops the server when it exits; on Linux
the server also dies with the add-on (the parent-death signal), and on Windows
with the job object it is assigned to. The add-on itself needs nothing but
Python's standard library.

## Building

Linux, in the AlmaLinux 8 container, so the archive runs on glibc 2.28 and
newer (needs Docker):

```bash
tools/build_linux.sh                      # llama.cpp, the model, tests, the bundle, the archive
tools/test_archive.sh almalinux:8         # the archive, installed as Subtitld does, on AlmaLinux 8
```

From source:

```bash
python tools/build_native.py              # Linux: tools/build_native_linux.sh
pip install -e '.[build,dev]'
pytest
pyinstaller pyinstaller.spec --noconfirm
dist/tiny-voca-addon/tiny-voca-addon --self-test
```

The release workflow builds, proves and publishes every platform on a tag
`v<version>` matching `manifest.json`; a manual run can build only some
platforms (`targets`). `tools/assemble_licences.py` fails the build when a
native file in the bundle has no notice.

## Licence

The add-on is Apache-2.0. tiny-voca's weights are Apache-2.0 (n4ze3m); the
phonemizer data in the model comes from dizione (Apache-2.0), the CMU
Pronouncing Dictionary (BSD-2-Clause) and Google's Wikipedia Homograph Data
(Apache-2.0). llama.cpp and ggml are MIT, with cpp-httplib and nlohmann/json
(MIT), miniaudio and stb_image (MIT-0 or public domain) and subprocess.h
(Unlicense) compiled in. Each archive carries `THIRD_PARTY_LICENSES/` with the
licence of everything frozen into it.
