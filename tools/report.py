#!/usr/bin/env python3
"""
Print an objective quality report for the singing voice.

    python -m tools.report              # every song
    python -m tools.report twinkle      # one song
    python -m tools.report --master     # measure the mastered mix too

Pitch is measured on the **dry** signal: reverb tails overlap neighbouring notes
and smear any pitch tracker, so a wet mix understates tuning accuracy.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daftdsp import quality, singer
from daftdsp.songs import SONGS, note_timeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song", nargs="?", choices=list(SONGS) + ["all"], default="all")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--voice-mode", default="natural", choices=["natural", "synth"])
    ap.add_argument("--master", action="store_true",
                    help="also report the mastered (wet) mix")
    args = ap.parse_args()

    names = list(SONGS) if args.song == "all" else [args.song]
    for name in names:
        s = SONGS[name]
        tl = note_timeline(s["score"], s["bpm"])
        dry = singer.sing(s["score"], args.sr, s["bpm"], s["voice"],
                          s["base_pitch"], voice_mode=args.voice_mode)
        print(quality.format_report(quality.summarize(dry, args.sr, tl),
                                    f"{name} [dry/{args.voice_mode}]"))
        if args.master:
            wet = singer.render_song(s["score"], args.sr, s["bpm"], s["voice"],
                                     s["base_pitch"], voice_mode=args.voice_mode)
            rep = quality.summarize(wet, args.sr)      # no pitch on the wet mix
            print(quality.format_report(rep, f"{name} [master]"))
        print()


if __name__ == "__main__":
    main()
