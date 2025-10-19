#!/usr/bin/env python3
"""
Audio Samples Creator

Processes audio segment files by splitting them into configurable duration samples,
trimming silence, and saving with sequential naming.

Configuration is loaded from config.json in the project root.
"""

import os
import shutil
import json
from pathlib import Path
from typing import List

from pydub import AudioSegment


# Configuration constants
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"
INPUT_DIR = PROJECT_ROOT / "data" / "audio-segments"
OUTPUT_DIR = PROJECT_ROOT / "data" / "audio-samples"
SILENCE_THRESHOLD_DB: int = -40
NORMALIZE_TARGET_DBFS: float = -20.0  # Target volume for normalization


class AudioSegmentNotFoundError(Exception):
    """ Raised when no segment files are found in the input directory """


class NoSamplesGeneratedError(Exception):
    """ Raised when a segment is too short to produce any samples """


def _load_config() -> dict:
    """ Load configuration from config.json """
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if 'create_audio_samples' not in config:
            raise ValueError("Missing 'create_audio_samples' section in config.json")

        return config['create_audio_samples']

    except FileNotFoundError as exc:
        raise ValueError(f"Configuration file not found: {CONFIG_PATH}") from exc
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {CONFIG_PATH}: {e}") from e


def _ensure_output_directory_empty() -> None:
    """ Create output directory and ensure it's empty """
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _get_segment_files() -> List[str]:
    """ Get sorted list of all .mp4 files from input directory """
    if not os.path.exists(INPUT_DIR):
        raise AudioSegmentNotFoundError(f"Input directory not found: {INPUT_DIR}")

    segment_files = []
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith('.mp4'):
            segment_files.append(os.path.join(INPUT_DIR, filename))

    if not segment_files:
        raise AudioSegmentNotFoundError(f"No .mp4 files found in {INPUT_DIR}")

    return sorted(segment_files)


def _load_audio_segment(filepath: str) -> AudioSegment:
    """ Load single segment file using pydub """
    try:
        return AudioSegment.from_file(filepath, format="mp4")
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {filepath}: {e}") from e


def _split_into_samples(audio: AudioSegment, sample_duration_ms: int) -> List[AudioSegment]:
    """ Split audio into chunks of specified duration, discard remainder """
    samples = []
    duration_ms = len(audio)

    if duration_ms < sample_duration_ms:
        raise NoSamplesGeneratedError(f"Segment too short: {duration_ms}ms < {sample_duration_ms}ms")

    # Split into chunks of specified duration
    for start_ms in range(0, duration_ms, sample_duration_ms):
        end_ms = min(start_ms + sample_duration_ms, duration_ms)

        # Only include chunks that are exactly the specified duration
        if end_ms - start_ms == sample_duration_ms:
            sample = audio[start_ms:end_ms]
            samples.append(sample)

    if not samples:
        raise NoSamplesGeneratedError("No valid samples generated from segment")

    return samples


def _trim_sample(sample: AudioSegment) -> AudioSegment:
    """ Apply silence trimming to sample """
    return sample.strip_silence(silence_thresh=SILENCE_THRESHOLD_DB)


def _normalize_sample(sample: AudioSegment) -> AudioSegment:
    """ Normalize sample to target volume """
    return sample.apply_gain(NORMALIZE_TARGET_DBFS - sample.dBFS)


def _save_sample(sample: AudioSegment, base_name: str, index: int) -> None:
    """ Save single sample with specified index """
    filename = f"{base_name}-{index:03d}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)

    try:
        sample.export(filepath, format="wav")
        print(f"Saved: {filename}")
    except Exception as e:
        raise RuntimeError(f"Failed to save sample {filename}: {e}") from e


def main() -> None:
    """ Main function to orchestrate audio sample creation process """
    # Load configuration
    config = _load_config()
    sample_duration_ms = config['sample_duration_ms']

    print("Starting audio sample creation...")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Sample duration: {sample_duration_ms}ms")
    print(f"Silence threshold: {SILENCE_THRESHOLD_DB}dB")
    print(f"Normalize target: {NORMALIZE_TARGET_DBFS}dBFS")
    print("Output format: WAV")
    print("-" * 50)

    try:
        # Ensure output directory is empty
        _ensure_output_directory_empty()
        print(f"Created/cleared output directory: {OUTPUT_DIR}")

        # Get segment files
        print("Finding segment files...")
        segment_files = _get_segment_files()
        print(f"Found {len(segment_files)} segment files")

        total_samples = 0
        processed_segments = 0

        # Process each segment
        print("Processing segments...")
        for segment_file in segment_files:
            try:
                # Extract base name (e.g., "segment_001" from "segment_001.mp4")
                base_name = os.path.splitext(os.path.basename(segment_file))[0]

                print(f"Processing {base_name}...")

                # Load segment
                audio = _load_audio_segment(segment_file)
                print(f"Loaded: {len(audio)}ms duration")

                # Split into samples
                samples = _split_into_samples(audio, sample_duration_ms)
                print(f"Generated {len(samples)} samples")

                # Trim, normalize and save samples
                print("Trimming, normalizing and saving samples...")
                for i, sample in enumerate(samples):
                    trimmed_sample = _trim_sample(sample)
                    normalized_sample = _normalize_sample(trimmed_sample)
                    _save_sample(normalized_sample, base_name, i)
                    total_samples += 1

                processed_segments += 1
                print(f"Completed {base_name}")

            except NoSamplesGeneratedError as e:
                print(f"Warning: Skipping {os.path.basename(segment_file)} - {e}")
                continue
            except RuntimeError as e:
                print(f"Error processing {os.path.basename(segment_file)}: {e}")
                continue

        print("\n" + "=" * 50)
        print("Audio sample creation complete!")
        print(f"Processed {processed_segments} segments")
        print(f"Generated {total_samples} samples in {OUTPUT_DIR}")

    except AudioSegmentNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure audio segment files exist in the data/audio-segments/ directory")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
