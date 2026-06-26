"""
Run locally with Claude Code before uploading to Modal.
Extracts audio from videos, chunks them, validates quality.
"""
import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class Sample:
    audio_path: str
    transcript: str
    duration: float
    language: str = "ur"


def extract_audio(video_path: str, output_path: str) -> bool:
    """Extract 16kHz mono WAV from video using ffmpeg."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-ar", "16000",        # 16kHz sample rate (Whisper requirement)
        "-ac", "1",            # Mono
        "-c:a", "pcm_s16le",  # 16-bit PCM
        output_path,
        "-y", "-loglevel", "error"
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def get_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def chunk_audio(audio_path: str, output_dir: str, 
                max_duration: int = 28) -> list[str]:
    """
    Split audio into chunks of max_duration seconds.
    28s leaves buffer under Whisper's 30s limit.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    duration = get_duration(audio_path)
    
    stem = Path(audio_path).stem
    chunks = []
    
    if duration <= max_duration:
        # Short file, no chunking needed
        out = f"{output_dir}/{stem}_000.wav"
        subprocess.run(["cp", audio_path, out])
        chunks.append(out)
    else:
        num_chunks = int(duration // max_duration) + 1
        for i in range(num_chunks):
            start = i * max_duration
            out = f"{output_dir}/{stem}_{i:03d}.wav"
            cmd = [
                "ffmpeg", "-i", audio_path,
                "-ss", str(start),
                "-t", str(max_duration),
                "-ar", "16000", "-ac", "1",
                out, "-y", "-loglevel", "error"
            ]
            subprocess.run(cmd)
            # Skip chunks shorter than 1 second
            if get_duration(out) >= 1.0:
                chunks.append(out)
    
    return chunks


def validate_transcript(transcript: str) -> bool:
    """Basic validation — customize for your use case."""
    if not transcript or len(transcript.strip()) < 2:
        return False
    if len(transcript) > 500:  # Likely a bad transcript
        return False
    return True


def prepare_dataset(
    input_dir: str,
    transcript_dir: str,  # Folder with .txt files matching audio names
    output_dir: str
):
    """
    Main preparation pipeline.
    Expects: input_dir/video.mp4 + transcript_dir/video.txt
    """
    audio_dir = Path(output_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    samples = []
    video_files = list(Path(input_dir).glob("*.mp4")) + \
                  list(Path(input_dir).glob("*.mkv")) + \
                  list(Path(input_dir).glob("*.webm"))
    
    print(f"Found {len(video_files)} video files")
    
    for video_path in video_files:
        transcript_path = Path(transcript_dir) / f"{video_path.stem}.txt"
        
        if not transcript_path.exists():
            print(f"  ⚠ No transcript for {video_path.name}, skipping")
            continue
        
        transcript = transcript_path.read_text(encoding="utf-8").strip()
        
        if not validate_transcript(transcript):
            print(f"  ⚠ Invalid transcript for {video_path.name}, skipping")
            continue
        
        # Extract audio
        wav_path = str(audio_dir / f"{video_path.stem}.wav")
        if not extract_audio(str(video_path), wav_path):
            print(f"  ✗ Audio extraction failed: {video_path.name}")
            continue
        
        duration = get_duration(wav_path)
        
        # Chunk if longer than 28 seconds
        if duration > 28:
            chunks = chunk_audio(wav_path, str(audio_dir / "chunks"))
            # For chunked files, you need per-chunk transcripts
            # This is a placeholder — handle per your transcript format
            print(f"  ℹ {video_path.name} is {duration:.1f}s, needs chunked transcripts")
            continue
        
        samples.append(Sample(
            audio_path=wav_path,
            transcript=transcript,
            duration=duration
        ))
        print(f"  ✓ {video_path.name} ({duration:.1f}s)")
    
    # Save manifest
    manifest_path = Path(output_dir) / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in samples], f, ensure_ascii=False, indent=2)
    
    total_hours = sum(s.duration for s in samples) / 3600
    print(f"\n✅ Prepared {len(samples)} samples ({total_hours:.2f} hours)")
    print(f"   Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--transcript_dir", required=True)
    parser.add_argument("--output_dir", default="./data/processed")
    args = parser.parse_args()
    
    prepare_dataset(args.input_dir, args.transcript_dir, args.output_dir)