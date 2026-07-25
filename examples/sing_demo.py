#!/usr/bin/env python3
"""
Render sung songs to WAV files with the Aria singer (daftdsp/singer.py).

    python examples/sing_demo.py                 # render every song
    python examples/sing_demo.py twinkle         # render one song
    python examples/sing_demo.py --outdir out    # choose output folder

Songs are word-based scores in ``daftdsp/songs.py``: items are ``("rest", beats)``
or ``(word, [(note, beats), ...])`` with one note per syllable.  A neural voice
speaks each phrase and the WORLD vocoder re-pitches it onto the melody.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daftdsp import singer, util
from daftdsp.songs import SONGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song", nargs="?", choices=list(SONGS) + ["all"],
                    default="all")
    ap.add_argument("--outdir", default="renders")
    ap.add_argument("--sr", type=int, default=singer.DEFAULT_SR)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    names = list(SONGS) if args.song == "all" else [args.song]
    for name in names:
        s = SONGS[name]
        t0 = time.time()
        audio = singer.render_song(s["score"], sr=args.sr, bpm=s["bpm"])
        path = os.path.join(args.outdir, f"aria_{name}.wav")
        with open(path, "wb") as f:
            f.write(util.write_wav(audio, args.sr))
        print(f"  {path}  ({time.time() - t0:.1f}s, {audio.shape[0] / args.sr:.1f}s audio)")


if __name__ == "__main__":
    main()
