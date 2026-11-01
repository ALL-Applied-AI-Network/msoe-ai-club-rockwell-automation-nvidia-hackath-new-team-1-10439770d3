"""Transcribe source-video audio through the authenticated ElevenLabs CLI."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    cli = shutil.which("elevenlabs.cmd") or shutil.which("elevenlabs")
    if not cli:
        raise SystemExit("ElevenLabs CLI not found on PATH")
    for video in sorted(args.input_dir.glob("*.mp4")):
        out_path = args.output_dir / f"{video.stem}.json"
        if out_path.is_file():
            result = json.loads(out_path.read_text(encoding="utf-8"))
            summaries.append({
                "video": video.name, "text": result.get("text", ""),
                "language_code": result.get("language_code"),
                "language_probability": result.get("language_probability"),
                "audio_duration_secs": result.get("audio_duration_secs"),
                "word_or_event_count": len(result.get("words", [])),
            })
            print(f"{video.name}: cached", flush=True)
            continue
        audio_only = args.output_dir / f"{video.stem}.m4a"
        media_input = audio_only if audio_only.is_file() else video
        command = [
            cli, "speech-to-text", "convert",
            "--file", str(media_input), "--model-id", "scribe_v2",
            "--timestamps-granularity", "word", "--diarize", "true",
            "--tag-audio-events", "true", "--format", "json", "--no-pager",
        ]
        completed = None
        for attempt in range(3):
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode == 0:
                break
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
        if completed is None or completed.returncode != 0:
            print(f"{video.name}: transcription failed: {completed.stderr.strip() if completed else 'unknown error'}", flush=True)
            continue
        result = json.loads(completed.stdout)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        summaries.append({
            "video": video.name,
            "text": result.get("text", ""),
            "language_code": result.get("language_code"),
            "language_probability": result.get("language_probability"),
            "audio_duration_secs": result.get("audio_duration_secs"),
            "word_or_event_count": len(result.get("words", [])),
        })
        print(f"{video.name}: {result.get('text', '')}", flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
