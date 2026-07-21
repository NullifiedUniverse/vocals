#!/usr/bin/env python3
"""
Render the singing-synth songs to WAV files.

    python examples/sing_demo.py                 # render every song
    python examples/sing_demo.py twinkle         # render one song
    python examples/sing_demo.py --outdir out    # choose output folder

Songs are defined in ``daftdsp/songs.py`` as (syllable, note, beats) scores.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daftdsp import singing, util
from daftdsp.songs import SONGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song", nargs="?", choices=list(SONGS) + ["all"], default="all")
    ap.add_argument("--outdir", default="renders")
    ap.add_argument("--sr", type=int, default=44100)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    names = list(SONGS) if args.song == "all" else [args.song]
    for name in names:
        s = SONGS[name]
        t0 = time.time()
        audio = singing.render_song(s["score"], sr=args.sr, bpm=s["bpm"],
                                    voice=s["voice"], base_pitch=s["base_pitch"],
                                    phoneme=s.get("phoneme", False))
        path = os.path.join(args.outdir, f"miku_{name}.wav")
        with open(path, "wb") as f:
            f.write(util.write_wav(audio, args.sr))
        print(f"  {path}  ({time.time() - t0:.1f}s, {audio.shape[0] / args.sr:.1f}s audio)")


if __name__ == "__main__":
    main()
