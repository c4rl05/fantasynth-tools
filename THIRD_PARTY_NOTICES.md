# Third-party notices

This file lists code from other projects that has been ported into this repo, with the
notices their licences require. Models and pip dependencies that are only downloaded or
installed at runtime are not part of this repo; their licences are listed in each tool's
README (for music-events: [Model weights and licences](music-events/README.md#model-weights-and-licences)).

## pdoom-video

Source: https://github.com/mexicat/pdoom-video

The lyric alignment in `music-events` is a port of pdoom-video's analysis code:

| File in this repo | Ported from |
|---|---|
| `music-events/scripts/lyrics_align.py` | `analysis/ctcalign.py` and `analysis/align.py`: the CTC emission fusion, the constrained Viterbi with a garbage token, the start refinement (rest-onset, onset-snap, fricative walk-back) and the agreement-based confidence. The vocal-signal features (RMS, sibilance, log-mel flux, pitch) follow `analysis/vocal_feats.py` |
| `music-events/scripts/lyrics.py` | the chunked CTC emission computation (20 s chunks with 3 s of context each side, 20 ms frames), from `analysis/ctc_emissions.py` |
| `music-events/scripts/assemble.py` | the pickup rule that maps a vocal entry to a bar (`entry_bar`), from `analysis/analyze.py` |

The port changes the method in places: the rest-onset rule is capped at 0.5 s, the flux
onset padding is corrected (`n_fft=1024`), and a line-time prior is added. pdoom-video's
song, lyrics and fonts are not covered by its MIT licence and nothing from them is included
here.

```
MIT License

Copyright (c) 2026 Giacomo Magnanini

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
