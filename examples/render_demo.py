#!/usr/bin/env python3
"""
Command-line demo: render the default patch (and optionally every preset) to WAV
files, without the web server.

    python examples/render_demo.py "we are the robots"
    python examples/render_demo.py "harder better faster stronger" --all
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daftdsp import EngineParams, process, util
from daftdsp.presets import PRESETS


def render(text, overrides, path):
    params = EngineParams.from_dict({"text": text, **overrides})
    t0 = time.time()
    stereo, meta = process(params)
    dt = time.time() - t0
    with open(path, "wb") as f:
        f.write(util.write_wav(stereo, params.sr))
    print(f"  {path}  ({dt:.2f}s, {meta['duration_s']}s audio, "
          f"f0={meta.get('median_f0')}Hz, chord={meta['chord_midi']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="we are the robots")
    ap.add_argument("--all", action="store_true", help="render every preset")
    ap.add_argument("--outdir", default="renders")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f'Rendering: "{args.text}"')
    render(args.text, {}, os.path.join(args.outdir, "default.wav"))
    if args.all:
        for name, ov in PRESETS.items():
            slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            render(args.text, ov, os.path.join(args.outdir, f"{slug}.wav"))


if __name__ == "__main__":
    main()
