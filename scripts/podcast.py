from openai import OpenAI
from datetime import datetime
import os
from pathlib import Path
from pydub import AudioSegment
import configparser
from typing import List
from utils.utils import open_file, cut_off_string

def generate_podcast(config: configparser.ConfigParser) -> None:
    """Generate a podcast from newsletter text content."""
    newsletter_text_location = config.get('podcast', 'newsletter_text_location')
    audio_files_path = config.get('podcast', 'audio_files_directory_path')
    
    newsletter_content: str = open_file(newsletter_text_location).strip()
    audio_files_path: Path = Path(audio_files_path)
    audio_files_path.mkdir(exist_ok=True)

    segment_files: List[Path] = generate_audio_segments(newsletter_content, audio_files_path, config)
    full_audio: AudioSegment = concatenate_audio_segments(segment_files)

    save_final_audio(full_audio, audio_files_path)
    cleanup_segment_files(segment_files)

def generate_audio_segments(content: str, audio_path: Path, config: configparser.ConfigParser) -> List[Path]:
    """Generate audio segments from text content."""
    cutoff_str: str = "\n"*4
    remaining_text: str = content
    segment_files: List[Path] = []
    client: OpenAI = OpenAI(api_key=open(config.get('openai', 'api_key_location')).read().strip())

    while remaining_text:
        segment_text, remaining_text = cut_off_string(remaining_text, cutoff_str)
        
        if not segment_text.strip():
            continue

        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=segment_text[:4096]
        )
        
        segment_file_path: Path = audio_path / f"segment_{len(segment_files)}.mp3"
        with open(segment_file_path, 'wb') as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
        segment_files.append(segment_file_path)

    return segment_files

def concatenate_audio_segments(segment_files: List[Path]) -> AudioSegment:
    """Concatenate audio segments into a single audio file."""
    full_audio: AudioSegment = AudioSegment.empty()
    for segment_file in segment_files:
        segment_audio: AudioSegment = AudioSegment.from_mp3(segment_file)
        full_audio += segment_audio
    return full_audio

def save_final_audio(audio: AudioSegment, audio_path: Path) -> None:
    """Save the final concatenated audio file."""
    date_str: str = datetime.now().strftime('%Y-%m-%d')
    print(f"Saving final audio to {audio_path / f"{date_str}_newsletter_podcast.mp3"}")
    final_audio_path: Path = audio_path / f"{date_str}_newsletter_podcast.mp3"
    audio.export(final_audio_path, format="mp3")

def cleanup_segment_files(segment_files: List[Path]) -> None:
    """Remove temporary audio segment files."""
    for segment_file in segment_files:
        os.remove(segment_file)
