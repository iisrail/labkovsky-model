"""
YouTube to Transcript using Whisper
Usage: python youtube_to_transcript_simple.py <youtube_url_or_video_id>
Output: Plain text file for review
"""

import subprocess
import whisper
import sys
import os

def download_audio(video_id, output_dir="./audio"):
    """Download audio from YouTube video"""
    os.makedirs(output_dir, exist_ok=True)
    
    if "youtube.com" in video_id or "youtu.be" in video_id:
        url = video_id
    else:
        url = f"https://www.youtube.com/watch?v={video_id}"
    
    output_path = os.path.join(output_dir, "%(id)s.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", output_path,
        url
    ]
    
    print(f"Downloading: {url}")
    subprocess.run(cmd, check=True)
    
    for f in os.listdir(output_dir):
        if f.endswith(".mp3"):
            return os.path.join(output_dir, f)
    
    raise FileNotFoundError("Audio not found")


def transcribe_audio(audio_path, model_name="large-v3"):
    """Transcribe audio using Whisper"""
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    
    print(f"Transcribing...")
    result = model.transcribe(audio_path, language="ru")
    
    return result["text"]


def main():
    if len(sys.argv) < 2:
        print("Usage: python youtube_to_transcript_simple.py <video_id>")
        sys.exit(1)
    
    video_input = sys.argv[1]
    
    # Extract video ID
    if "youtube.com" in video_input:
        video_id = video_input.split("v=")[-1].split("&")[0]
    elif "youtu.be" in video_input:
        video_id = video_input.split("/")[-1]
    else:
        video_id = video_input
    
    # Download
    audio_path = download_audio(video_input)
    
    # Transcribe
    text = transcribe_audio(audio_path)
    
    # Save as plain text for review
    os.makedirs("./transcripts_review", exist_ok=True)
    output_path = f"./transcripts_review/{video_id}.txt"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"\nSaved to: {output_path}")
    print("Review and edit, then extract Q&A pairs manually.")


if __name__ == "__main__":
    main()