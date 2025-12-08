"""
YouTube to Transcript using Whisper
Usage: python youtube_to_transcript.py <youtube_url_or_video_id>
"""

import subprocess
import whisper
import sys
import os
import json

def download_audio(video_id, output_dir="./audio"):
    """Download audio from YouTube video"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle full URL or just video ID
    if "youtube.com" in video_id or "youtu.be" in video_id:
        url = video_id
    else:
        url = f"https://www.youtube.com/watch?v={video_id}"
    
    output_path = os.path.join(output_dir, "%(id)s.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", output_path,
        url
    ]
    
    print(f"Downloading audio from: {url}")
    subprocess.run(cmd, check=True)
    
    # Find the downloaded file
    for f in os.listdir(output_dir):
        if f.endswith(".mp3"):
            return os.path.join(output_dir, f)
    
    raise FileNotFoundError("Audio file not found after download")


def transcribe_audio(audio_path, model_name="large-v3"):
    """Transcribe audio using Whisper"""
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    
    print(f"Transcribing: {audio_path}")
    result = model.transcribe(
        audio_path,
        language="ru",
        verbose=True
    )
    
    return result


def save_transcript(result, video_id, output_dir="./transcripts"):
    """Save transcript to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plain text
    txt_path = os.path.join(output_dir, f"{video_id}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    
    # Save with timestamps (JSON)
    json_path = os.path.join(output_dir, f"{video_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "video_id": video_id,
            "text": result["text"],
            "segments": result["segments"]
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Saved: {txt_path}")
    print(f"Saved: {json_path}")
    
    return txt_path, json_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python youtube_to_transcript.py <youtube_url_or_video_id>")
        print("Example: python youtube_to_transcript.py rJvn19GKNA")
        sys.exit(1)
    
    video_input = sys.argv[1]
    
    # Extract video ID from URL if needed
    if "youtube.com" in video_input:
        video_id = video_input.split("v=")[-1].split("&")[0]
    elif "youtu.be" in video_input:
        video_id = video_input.split("/")[-1]
    else:
        video_id = video_input
    
    # Download
    audio_path = download_audio(video_input)
    
    # Transcribe
    result = transcribe_audio(audio_path)
    
    # Save
    save_transcript(result, video_id)
    
    # Print preview
    print("\n--- Transcript preview (first 500 chars) ---")
    print(result["text"][:500])


if __name__ == "__main__":
    main()