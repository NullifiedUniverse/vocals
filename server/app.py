"""
Flask server for the Daft Punk vocal DSP engine.

Endpoints
  GET  /                    -> the slider UI (web/index.html)
  GET  /api/meta            -> defaults, presets, and option lists for the UI
  POST /api/synthesize      -> JSON params in, JSON {audio_b64, meta} out (16-bit)
  POST /api/synthesize.wav  -> same params, raw WAV download (?float32=1 for 32-bit)
  GET  /health              -> "ok"
"""
from __future__ import annotations

import base64
import os
import sys
import time

from flask import Flask, Response, jsonify, request, send_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daftdsp import EngineParams, process, quality, singer, util  # noqa: E402
from daftdsp.formant import VOWELS  # noqa: E402
from daftdsp.presets import PRESETS  # noqa: E402
from daftdsp.songs import (SONGS, format_score, note_timeline,  # noqa: E402
                           parse_score)
from daftdsp.tts import espeak_path  # noqa: E402
from daftdsp.util import SCALES  # noqa: E402

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

app = Flask(__name__, static_folder=WEB_DIR, static_url_path="/static")

CHORD_QUALITIES = ["major", "minor", "maj7", "min7", "dom7", "sus2", "sus4",
                   "power", "min9", "add9"]
VOICES = ["en+f3", "en+f4", "en+f2", "en+m2", "en+m3", "en", "en-us", "en-gb",
          "en+croak", "en+whisper"]


@app.get("/")
def index():
    return send_file(os.path.join(WEB_DIR, "index.html"))


@app.get("/health")
def health():
    return "ok"


@app.get("/favicon.ico")
def favicon():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
           '<text y="26" font-size="26">🤖</text></svg>')
    return Response(svg, mimetype="image/svg+xml")


@app.get("/api/meta")
def meta():
    return jsonify({
        "defaults": EngineParams().to_dict(),
        "presets": PRESETS,
        "scales": list(SCALES.keys()),
        "vowels": list(VOWELS.keys()),
        "chord_qualities": CHORD_QUALITIES,
        "voices": VOICES,
        "espeak": bool(espeak_path()),
    })


def _render(payload):
    params = EngineParams.from_dict(payload or {})
    t0 = time.time()
    stereo, info = process(params)
    info["render_ms"] = int((time.time() - t0) * 1000)
    info["sr"] = params.sr
    return stereo, info, params


@app.post("/api/synthesize")
def synthesize():
    try:
        stereo, info, params = _render(request.get_json(force=True, silent=True) or {})
    except Exception as exc:  # surface engine errors to the UI
        return jsonify({"error": str(exc)}), 400
    wav = util.write_wav(stereo, params.sr, float32=False)
    return jsonify({
        "audio_b64": base64.b64encode(wav).decode("ascii"),
        "meta": info,
    })


@app.post("/api/synthesize.wav")
def synthesize_wav():
    float32 = request.args.get("float32") in ("1", "true", "yes")
    stereo, info, params = _render(request.get_json(force=True, silent=True) or {})
    wav = util.write_wav(stereo, params.sr, float32=float32)
    return Response(
        wav, mimetype="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="daftpunk_vocal.wav"'},
    )


# ---------------------------------------------------------------------------
# Singing voice (Aria)
# ---------------------------------------------------------------------------

SING_VOICES = ["en+f4", "en+f3", "en+f2", "en+m3", "en+m2", "en"]


@app.get("/api/sing/meta")
def sing_meta():
    # A list (not a dict) so the curated song order survives JSON serialisation.
    return jsonify({
        "songs": [{"name": name, "score": format_score(s["score"]),
                   "bpm": s["bpm"], "voice": s["voice"],
                   "base_pitch": s["base_pitch"]}
                  for name, s in SONGS.items()],
        "voices": SING_VOICES,
        "voice_modes": ["natural", "synth"],
        "defaults": {"bpm": 108, "voice": "en+f4", "base_pitch": 64,
                     "voice_mode": "natural", "formant_shift": 1.0,
                     "reverb_mix": 0.18, "width": 1.2},
    })


def _sing_render(payload):
    """Shared render path for both singing endpoints."""
    p = payload or {}
    score = parse_score(p.get("score", ""))
    sr = 44100
    bpm = float(p.get("bpm", 108))
    kw = dict(
        voice=str(p.get("voice", "en+f4")),
        base_pitch=int(float(p.get("base_pitch", 64))),
        voice_mode=("synth" if p.get("voice_mode") == "synth" else "natural"),
        formant_shift=float(p.get("formant_shift", 1.0)),
    )
    t0 = time.time()
    stereo = singer.render_song(score, sr=sr, bpm=bpm,
                                reverb_mix=float(p.get("reverb_mix", 0.18)),
                                width=float(p.get("width", 1.2)), **kw)
    # Quality is reported from the dry signal (reverb smears pitch tracking).
    dry = singer.sing(score, sr, bpm, kw["voice"], kw["base_pitch"],
                      voice_mode=kw["voice_mode"],
                      formant_shift=kw["formant_shift"])
    rep = quality.summarize(dry, sr, note_timeline(score, bpm))
    meta = {
        "render_ms": int((time.time() - t0) * 1000),
        "duration_s": round(stereo.shape[0] / sr, 2),
        "notes": rep["pitch"]["notes"],
        "in_tune": rep["pitch"]["within_50c"],
        "mean_cents": round(rep["pitch"]["mean_abs_cents"], 1),
        "clicks": rep["discontinuity"]["count"],
        "dropout": round(rep["dropout"], 3),
        "sr": sr,
    }
    return stereo, meta, sr


@app.post("/api/sing")
def sing_endpoint():
    try:
        stereo, meta, sr = _sing_render(request.get_json(force=True, silent=True))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001 - surface engine errors to the UI
        return jsonify({"error": f"render failed: {exc}"}), 500
    return jsonify({"audio_b64": base64.b64encode(
        util.write_wav(stereo, sr)).decode("ascii"), "meta": meta})


@app.post("/api/sing.wav")
def sing_wav():
    try:
        stereo, _, sr = _sing_render(request.get_json(force=True, silent=True))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    float32 = request.args.get("float32") in ("1", "true", "yes")
    return Response(
        util.write_wav(stereo, sr, float32=float32), mimetype="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="aria_song.wav"'},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"espeak available: {bool(espeak_path())}")
    print(f"Serving Daft Punk Vocal DSP on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
