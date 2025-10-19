# ACE-Step LoRA Training Pipeline

Automated pipeline for creating and training LoRAs for ACE-Step music generation from audio samples.

## Overview

This project provides a complete 3-step pipeline to train custom LoRAs for ACE-Step:

1. **Create Audio Samples** - Split audio segments into training samples
2. **Generate Audio Tags** - Use Gemini API to create descriptive tags for each sample  
3. **Train LoRA** - Train the custom LoRA using ACE-Step's training infrastructure

## Prerequisites

### Environment Setup

1. **Install pyenv** (if not already installed):
   ```bash
   # macOS
   brew install pyenv
   
   # Ubuntu/Debian  
   sudo apt install pyenv
   ```

2. **Run the installation script**:
   ```bash
   ./install.sh
   ```
   This script will:
   - Set up Python 3.11.10 using pyenv
   - Create the virtual environment
   - Install dependencies from `requirements.txt`
   - Initialize ACE-Step submodule and its dependencies

### External Dependencies

- **ffmpeg**: Required for audio processing
  ```bash
  # macOS
  brew install ffmpeg
  # Ubuntu/Debian
  sudo apt install ffmpeg
  ```
- **Gemini API Key**: Required for audio tagging

## Configuration

Create a `config.json` in the project root (easiest option is to copy/rename `example-config.json`)

```json
{
  "create_audio_samples": {
    "sample_duration_ms": 10000
  },
  "generate_audio_tags": {
    "gemini_api_key": "your-api-key-here",
    "gemini_prompt": "Analyze this audio clip and provide a concise, clean, comma-separated list of 5-8 descriptive keywords (tags). Do not include any introductory or concluding text, just the tags",
    "model": "gemini-2.5-flash"
  },
  "train_lora": {
    "lora_config_path": "dependencies/lora_config.json",
    "max_steps": 5000,
    "exp_name": "your_lora_name"
  }
}
```

The Gemini prompt is configurable as it's a good idea to lead Gemini into what kind of samples you're posting so you can get a more consistent description back each time.  e.g., when training a dystopian ambient music LoRA, the prompt could be as follows

`Analyze this audio clip, which is an instrumental track in a sci-fi or cyberpunk style. Provide a concise, clean, comma-separated list of up to 5 descriptive keywords (tags) covering: 1. Instrumentation/Synthesis (e.g., analog synth, arpeggios, gated reverb), 2. Style/Sub-genre (e.g., Synthwave, Dystopian, Dark Ambient), and 3. Mood/Narrative (e.g., relaxed, high note, passive, neon-drenched, melancholic). Do not include any introductory or concluding text, just the tags`

## Usage

### Step 1: Prepare Audio Data

Place your source audio files in `data/audio-segments/` as MP4 files:
```
data/audio-segments/
├── segment_001.mp4
├── segment_002.mp4
└── ...
```

These can be any length, the assumption being that they are multi-minute segments (e.g. complete tracks).  It goes without saying, do not use copyrighted material for this input.

### Step 2: Create Audio Samples

Split segments into training samples:
```bash
python src/01_create_audio_samples.py
```

This creates WAV files in `data/audio-samples/` with:
- Configurable duration (default: 10 seconds)
- Silence trimming and normalization
- Sequential naming: `segment_001-000.wav`, `segment_001-001.wav`, etc.

### Step 3: Generate Audio Tags

Create descriptive tags using Gemini API:
```bash
python src/02_generate_audio_tags.py
```

Generates corresponding `.txt` files with comma-separated tags:
- `segment_001-000.wav` → `segment_001-000.txt`
- Processes files in parallel with rate limiting
- Creates master tag list at `data/all_tags.txt`

### Step 4: Train LoRA

Train your custom LoRA:
```bash
python src/03_train_lora.py
```

This script:
- Converts WAV/TXT pairs to Hugging Face dataset format
- Runs ACE-Step training with configured parameters
- Saves checkpoints to `data/loras/checkpoints/`
- Outputs trained LoRA model

