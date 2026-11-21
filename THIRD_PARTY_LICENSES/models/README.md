# The model

tiny-voca's model is part of this archive, in `bin/_internal/models/`.

tiny-voca is by Muhammed Nazeem (n4ze3m), published as
[n4ze3m/tiny-voca](https://huggingface.co/n4ze3m/tiny-voca) under the Apache
License 2.0 ([Apache-2.0.txt](Apache-2.0.txt)). Its two voices were made from
text descriptions and its training audio is synthetic: no recording of a real
speaker is involved.

The add-on uses the model as converted to GGUF for llama.cpp by ggml-org,
published in [ggml-org/tiny-voca-GGUF](https://huggingface.co/ggml-org/tiny-voca-GGUF)
under the same licence, at revision `8a785351a6dac0b9168bc14bc5de635a98a7b809`,
each file checked against its sha256 when the add-on was built. The files
also carry the pronunciation dictionary and rules of tiny-voca's phonemizer,
[dizione](https://github.com/n4ze3m/dizione) (Apache-2.0), which llama.cpp
reads as the model's `tts` template. That data is built from the CMU
Pronouncing Dictionary ([CMUdict-LICENSE.txt](CMUdict-LICENSE.txt),
BSD-2-Clause) and Google's Wikipedia Homograph Data
([Wikipedia-Homograph-Data-LICENSE.txt](Wikipedia-Homograph-Data-LICENSE.txt),
Apache-2.0).

| File | What it is |
|------|------------|
| `tiny-voca-F32.gguf` (12 MB) | The phoneme encoder, with the phonemizer's dictionary and rules |
| `mmproj-tiny-voca-F32.gguf` (14 MB) | The duration predictor, the frame transformers and the source-filter synthesizer |
