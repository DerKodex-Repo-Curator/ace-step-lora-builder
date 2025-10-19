#!/usr/bin/env python3
"""
ACE-Step LoRA Training Automation Script

This script automates the entire LoRA training pipeline for ACE-Step:
1. Validates and prepares audio data (WAV→MP3, creates required file pairs)
2. Converts data to Hugging Face dataset format
3. Trains LoRA adapter using ACE-Step trainer

The script handles all intermediate steps and cleanup automatically.
"""

import sys
import subprocess
import shutil
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants - extracted to top level per coding standards
PROJECT_ROOT = Path(__file__).parent.parent
ACE_STEP_DIR = PROJECT_ROOT / "dependencies" / "ace-step"
ACE_STEP_VENV = ACE_STEP_DIR / "venv" / "bin" / "python"
CONFIG_PATH = PROJECT_ROOT / "config.json"

# Training directory structure - centralized configuration
TRAINING_BASE_DIR = PROJECT_ROOT / "data" / "lora-training"
TRAINING_CHECKPOINTS_DIR = TRAINING_BASE_DIR / "checkpoints"
TRAINING_WORKING_DIR = TRAINING_BASE_DIR / "working"
TRAINING_DATASET_DIR = TRAINING_BASE_DIR / "lora-dataset"


def load_config() -> Dict[str, Any]:
    """ Load configuration from config.json """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if 'train_lora' not in config:
            raise ValueError("Missing 'train_lora' section in config.json")

        return config['train_lora']

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}") from e


def setup_directories() -> None:
    """ Create necessary directories and clean working directory """
    logger.info("Setting up directories...")

    # Clean and recreate entire training directory
    if TRAINING_BASE_DIR.exists():
        shutil.rmtree(TRAINING_BASE_DIR)

    # Create all directories fresh
    TRAINING_CHECKPOINTS_DIR.mkdir(parents=True)
    TRAINING_WORKING_DIR.mkdir(parents=True)
    TRAINING_DATASET_DIR.mkdir(parents=True)


def validate_source_data() -> List[Tuple[Path, Path]]:
    """
    Validate that each WAV file has a corresponding TXT file

    Returns:
        List of tuples (wav_path, txt_path) for valid pairs
    """
    logger.info("Validating source data...")

    audio_samples_dir = PROJECT_ROOT / "data" / "audio-samples"
    if not audio_samples_dir.exists():
        raise FileNotFoundError(f"Audio samples directory not found: {audio_samples_dir}")

    wav_files = sorted(audio_samples_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {audio_samples_dir}")

    valid_pairs = []
    missing_txt_files = []

    for wav_file in wav_files:
        txt_file = wav_file.with_suffix('.txt')
        if txt_file.exists():
            valid_pairs.append((wav_file, txt_file))
        else:
            missing_txt_files.append(txt_file.name)

    if missing_txt_files:
        raise FileNotFoundError(
            f"Missing TXT files for WAV files: {', '.join(missing_txt_files)}"
        )

    logger.info("Found %d valid audio-text pairs", len(valid_pairs))
    return valid_pairs


def convert_wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    """
    Convert WAV file to MP3 using ffmpeg

    Args:
        wav_path: Source WAV file path
        mp3_path: Destination MP3 file path
    """
    try:
        # Use ffmpeg to convert WAV to MP3
        cmd = [
            "ffmpeg", "-i", str(wav_path),
            "-codec:a", "libmp3lame", "-b:a", "192k",
            "-y",  # Overwrite output file
            str(mp3_path)
        ]

        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert {wav_path} to MP3: {e.stderr}") from e
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg to convert audio files."
        ) from exc


def prepare_working_data(valid_pairs: List[Tuple[Path, Path]]) -> None:
    """
    Prepare data in working directory with correct naming and format

    Args:
        valid_pairs: List of (wav_path, txt_path) tuples
    """
    logger.info("Preparing working data...")

    for wav_path, txt_path in valid_pairs:
        base_name = wav_path.stem

        # Convert WAV to MP3
        mp3_path = TRAINING_WORKING_DIR / f"{base_name}.mp3"
        logger.info("Converting %s to MP3...", wav_path.name)
        convert_wav_to_mp3(wav_path, mp3_path)

        # Copy and rename TXT to _prompt.txt
        prompt_path = TRAINING_WORKING_DIR / f"{base_name}_prompt.txt"
        shutil.copy2(txt_path, prompt_path)

        # Create empty _lyrics.txt file
        lyrics_path = TRAINING_WORKING_DIR / f"{base_name}_lyrics.txt"
        lyrics_path.write_text("", encoding='utf-8')

    logger.info("Prepared %d audio samples in working directory", len(valid_pairs))


def run_dataset_conversion() -> None:
    """ Run ACE-Step's convert2hf_dataset.py to create Hugging Face dataset """
    logger.info("Converting data to Hugging Face dataset format...")

    # Ensure ACE-Step directory exists
    if not ACE_STEP_DIR.exists():
        raise FileNotFoundError(f"ACE-Step directory not found: {ACE_STEP_DIR}")

    # Check if ACE-Step venv exists
    if not ACE_STEP_VENV.exists():
        raise FileNotFoundError(f"ACE-Step venv not found: {ACE_STEP_VENV}")

    # Prepare command with sensible defaults
    cmd = [
        str(ACE_STEP_VENV),
        str(ACE_STEP_DIR / "convert2hf_dataset.py"),
        "--data_dir", str(TRAINING_WORKING_DIR),
        "--repeat_count", "1",  # Use 1 instead of 2000 to avoid empty samples
        "--output_name", str(TRAINING_DATASET_DIR)
    ]

    logger.info("Running dataset conversion: %s", ' '.join(cmd))

    try:
        result = subprocess.run(
            cmd,
            cwd=str(ACE_STEP_DIR),
            capture_output=True,
            text=True,
            check=True
        )

        logger.info("Dataset conversion completed successfully")
        logger.debug("Conversion stdout: %s", result.stdout)

    except subprocess.CalledProcessError as e:
        logger.error("Dataset conversion failed: %s", e.stderr)
        raise RuntimeError(f"Dataset conversion failed: {e.stderr}") from e


def run_training(config: Dict[str, Any]) -> None:
    """ Run ACE-Step's trainer.py to train LoRA adapter """
    logger.info("Starting LoRA training...")

    # Check if LoRA config exists
    lora_config_path = PROJECT_ROOT / config['lora_config_path']
    if not lora_config_path.exists():
        raise FileNotFoundError(f"LoRA config not found: {lora_config_path}")

    # ACE-Step will handle model downloading automatically
    logger.info("ACE-Step will download and cache models automatically")

    # Prepare command with sensible defaults
    cmd = [
        str(ACE_STEP_VENV),
        str(ACE_STEP_DIR / "trainer.py"),
        "--dataset_path", str(TRAINING_DATASET_DIR),
        "--logger_dir", str(TRAINING_CHECKPOINTS_DIR),
        "--lora_config_path", str(lora_config_path),
        "--exp_name", config['exp_name'],
        "--max_steps", str(config['max_steps']),
        "--every_n_train_steps", str(config['checkpoint_generation_period']),
        "--every_plot_step", str(config['plot_generation_period']),
        "--save_top_k", str(config['save_historical_checkpoints'])
    ]

    logger.info("Running training: %s", ' '.join(cmd))

    try:
        # Run training with real-time output
        process = subprocess.Popen(
            cmd,
            cwd=str(ACE_STEP_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream output in real-time
        for line in process.stdout:
            print(line.rstrip())

        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"Training failed with return code {process.returncode}")

        logger.info("Training completed successfully")

    except Exception as e:
        logger.error("Training failed: %s", e)
        raise


def cleanup_working_directory() -> None:
    """ Clean up temporary working directory """
    logger.info("Cleaning up working directory...")

    if TRAINING_WORKING_DIR.exists():
        shutil.rmtree(TRAINING_WORKING_DIR)
        logger.info("Working directory cleaned up")


def main() -> None:
    """ Main execution function """
    try:
        logger.info("Starting ACE-Step LoRA training automation...")

        # Load configuration
        config = load_config()

        # Phase 1: Setup and validation
        setup_directories()
        valid_pairs = validate_source_data()

        # Phase 2: Data preparation
        prepare_working_data(valid_pairs)

        # Phase 3: Dataset conversion
        run_dataset_conversion()

        # Phase 4: Training
        run_training(config)

        logger.info("LoRA training pipeline completed successfully!")

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        raise

    finally:
        # Always clean up working directory
        cleanup_working_directory()


if __name__ == "__main__":
    main()
