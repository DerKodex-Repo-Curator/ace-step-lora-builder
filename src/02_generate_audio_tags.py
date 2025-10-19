#!/usr/bin/env python3
"""
Audio tagging script using Gemini API.

Processes WAV files in a directory, uploads them to Gemini Files API,
generates keyword tags using configured Gemini model, and saves results to text files.
Immediately deletes uploaded files to stay within free tier limits.

Configuration is loaded from config.json in the project root.
Dependencies: pip install google-genai
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai


# Configuration constants
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"
AUDIO_DIR = PROJECT_ROOT / "data" / "audio-samples"
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2
BATCH_SIZE = 1


# Custom exceptions
class AuthenticationError(Exception):
    """ Raised when API key cannot be loaded """


class UploadError(Exception):
    """ Raised when file upload fails """


class GenerationError(Exception):
    """ Raised when content generation fails """


def _load_config() -> dict:
    """ Load configuration from config.json """
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if 'generate_audio_tags' not in config:
            raise AuthenticationError("Missing 'generate_audio_tags' section in config.json")

        return config['generate_audio_tags']

    except FileNotFoundError as exc:
        raise AuthenticationError(f"Configuration file not found: {CONFIG_PATH}") from exc
    except json.JSONDecodeError as e:
        raise AuthenticationError(f"Invalid JSON in {CONFIG_PATH}: {e}") from e


def _has_paired_txt(wav_path: Path) -> bool:
    """ Check if corresponding .txt file exists """
    txt_path = wav_path.with_suffix('.txt')
    return txt_path.exists()


def _upload_with_retry(client: genai.Client, file_path: Path) -> genai.types.File:
    """ Upload WAV file to Gemini Files API """
    try:
        logging.info("Uploading %s...", file_path.name)
        uploaded_file = client.files.upload(file=file_path)
        logging.info("Successfully uploaded %s", file_path.name)
        return uploaded_file
    except Exception as exc:
        raise UploadError(f"Failed to upload {file_path.name}: {exc}") from exc


def _generate_tags_with_retry(client: genai.Client, uploaded_file: genai.types.File, prompt: str, model: str) -> str:
    """ Call client.models.generate_content() with exponential backoff """
    delay = INITIAL_RETRY_DELAY

    for attempt in range(MAX_RETRIES):
        try:
            logging.info("Generating tags (attempt %d/%d)...", attempt + 1, MAX_RETRIES)

            # Request multimodal generation (text prompt + audio file)
            response = client.models.generate_content(
                model=model,
                contents=[prompt, uploaded_file]
            )

            if not response.text:
                raise GenerationError("Empty response from model")

            logging.info("Successfully generated tags")
            return response.text.strip()

        except (OSError, IOError, ConnectionError, ValueError, RuntimeError) as exc:
            error_msg = str(exc).lower()
            if "quota" in error_msg or "rate" in error_msg:
                if attempt < MAX_RETRIES - 1:
                    logging.warning("Rate limit hit, retrying in %ds...", delay)
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    logging.error("Exhausted all retries for content generation")
                    sys.exit(1)
            else:
                raise GenerationError(f"Content generation failed: {exc}") from exc

    # This should never be reached due to sys.exit(1) above
    raise GenerationError("Unexpected error in retry logic")


def _delete_file_safe(client: genai.Client, file_name: str) -> None:
    """ Attempt to delete uploaded file from API """
    try:
        client.files.delete(name=file_name)
        logging.info("Successfully deleted %s from API", file_name)
    except (OSError, IOError, ConnectionError) as exc:
        logging.warning("Failed to delete %s from API: %s. File will auto-expire after 48h.", file_name, exc)


def _analyze_tag_frequency(all_tags: set) -> str:
    """ Analyze tag frequency across all existing .txt files """
    tag_counts = defaultdict(int)
    tag_file_counts = defaultdict(int)  # Track how many files contain each tag

    # Get all .txt files in audio-samples directory
    txt_files = list(AUDIO_DIR.glob("*.txt"))
    total_files = len(txt_files)

    if total_files == 0:
        return ""

    # Count occurrences of each tag and track file presence
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            # Split by comma and normalize tags to lowercase (already deduplicated in files)
            file_tags = set(tag.strip().lower() for tag in content.split(',') if tag.strip())

            # Count occurrences and file presence for each tag from all_tags set
            for tag in all_tags:
                if tag in file_tags:
                    tag_counts[tag] += 1
                    tag_file_counts[tag] += 1

        except (OSError, IOError) as exc:
            logging.warning("Error reading %s for frequency analysis: %s", txt_file.name, exc)

    # Sort tags by frequency (descending)
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)

    # Format frequency analysis as CSV
    frequency_lines = ["Tags,Instance Count,% of Data Set"]
    for tag, count in sorted_tags:
        file_count = tag_file_counts[tag]
        percentage = (file_count / total_files) * 100
        frequency_lines.append(f"{tag},{count},{percentage:.2f}%")

    return "\n".join(frequency_lines)


def _extract_tags_from_csv() -> list[str]:
    """ Extract unique tags from all_tags.csv file """
    csv_path = PROJECT_ROOT / "data" / "all_tags.csv"

    if not csv_path.exists():
        logging.warning("all_tags.csv not found, skipping normalization")
        return []

    tags = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Skip header row and extract first column (tag names)
        for line in lines[1:]:  # Skip "Tags,Instance Count,% of Data Set"
            if line.strip():
                tag = line.split(',')[0].strip()
                if tag:
                    tags.append(tag)

    except (OSError, IOError) as exc:
        logging.error("Error reading all_tags.csv: %s", exc)
        return []

    return tags


def _normalize_tags_with_gemini(client: genai.Client, tags: list[str], model: str) -> dict[str, str]:
    """ Call Gemini API to normalize tags and return mapping dictionary """
    if not tags:
        return {}

    # Build the normalization prompt
    tags_list = ', '.join(tags)
    prompt = f"""Please normalize the provided list of audio tags for Lora training. The goal is to reduce the overall number of unique tags to approximately 20-30 by consolidating similar, redundant, or semantically related terms.

The normalization should be done for all tags, even those that do not need to be consolidated. If you identify a better, more descriptive, or higher-level term than any in the original group, use it as the target tag.

Every tag that is included in this list must be included in the output, and the final tag suggestion must not be generic.  e.g. "sound design" means very little and doesn't describe a particular sound or style.

The output must strictly follow this exact format for every group:
[list of original tags to change] tag to change to

The list of original tags on the left must include the final target tag on the right if it was present in the original list. Tags in the list should be comma-separated.

Example formats to follow:
[synth pad, synth pads, dark synth pad, lush synth pad, evolving synth pad] synth pad
[dystopian, dystopian sci-fi, sci-fi dystopian, urban decay] dystopian
[foreboding, ominous, eerie, brooding, dark, dread, unsettling, tension] tension

Original List of Tags to Normalize:
{tags_list}"""

    try:
        logging.info("Sending normalization request to Gemini API...")
        response = client.models.generate_content(
            model=model,
            contents=[prompt]
        )

        if not response.text:
            logging.error("Empty response from Gemini API for normalization")
            return {}

        # Parse the response
        normalization_map = {}
        lines = response.text.strip().split('\n')

        logging.info("Normalization mapping:")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse format: [tag1, tag2, tag3] target_tag
            if line.startswith('[') and ']' in line:
                try:
                    # Find the closing bracket
                    bracket_end = line.find(']')
                    if bracket_end == -1:
                        logging.warning("Invalid format in line: %s", line)
                        continue

                    # Extract tags list and target tag
                    tags_part = line[1:bracket_end].strip()
                    target_part = line[bracket_end + 1:].strip()

                    if not target_part:
                        logging.warning("No target tag found in line: %s", line)
                        continue

                    # Split tags by comma and clean them
                    source_tags = [tag.strip() for tag in tags_part.split(',') if tag.strip()]

                    # Add mapping for each source tag
                    for source_tag in source_tags:
                        normalization_map[source_tag] = target_part

                    # Print the mapping for user review
                    logging.info("%s", line)

                except (ValueError, IndexError) as exc:
                    logging.warning("Error parsing normalization line '%s': %s", line, exc)
                    continue

        logging.info("Generated %d tag mappings", len(normalization_map))
        return normalization_map

    except (OSError, IOError, ConnectionError, ValueError, RuntimeError) as exc:
        logging.error("Failed to normalize tags with Gemini API: %s", exc)
        return {}


def _apply_tag_normalization(normalization_map: dict[str, str]) -> None:
    """ Apply tag normalization mapping to all txt files """
    if not normalization_map:
        logging.info("No normalization mappings to apply")
        return

    txt_files = list(AUDIO_DIR.glob("*.txt"))
    if not txt_files:
        logging.info("No txt files found to normalize")
        return

    logging.info("Applying normalization to %d files...", len(txt_files))

    files_updated = 0
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                continue

            # Split by comma and normalize each tag
            tags = [tag.strip() for tag in content.split(',') if tag.strip()]
            normalized_tags = []

            for tag in tags:
                # Apply normalization if mapping exists, otherwise remove tag
                if tag in normalization_map:
                    normalized_tag = normalization_map[tag]
                    normalized_tags.append(normalized_tag)
                # Tags not in normalization_map are removed (not added to normalized_tags)

            # Deduplicate and sort
            unique_tags = sorted(set(normalized_tags))
            new_content = ','.join(unique_tags)

            # Write back if content changed
            if new_content != content:
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                files_updated += 1
                logging.debug("Updated %s", txt_file.name)

        except (OSError, IOError) as exc:
            logging.warning("Error updating %s: %s", txt_file.name, exc)

    logging.info("Normalization complete. Updated %d files", files_updated)


def _ensure_top_tags_in_all_files() -> None:
    """ Ensure all txt files contain the two most frequent tags from all_tags.csv """
    csv_path = PROJECT_ROOT / "data" / "all_tags.csv"

    if not csv_path.exists():
        logging.warning("all_tags.csv not found, skipping top tag enforcement")
        return

    # Read the top two tags from CSV
    top_tags = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Skip header and get first two tags
        for line in lines[1:3]:  # Lines 1 and 2 (top two tags)
            if line.strip():
                tag = line.split(',')[0].strip()
                if tag:
                    top_tags.append(tag)

    except (OSError, IOError) as exc:
        logging.error("Error reading all_tags.csv for top tags: %s", exc)
        return

    if len(top_tags) < 2:
        logging.warning("Less than 2 tags found in all_tags.csv, skipping top tag enforcement")
        return

    logging.info("Ensuring top tags '%s' and '%s' are in all files...", top_tags[0], top_tags[1])

    txt_files = list(AUDIO_DIR.glob("*.txt"))
    if not txt_files:
        logging.info("No txt files found for top tag enforcement")
        return

    files_updated = 0
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                # If file is empty, just add the top two tags
                new_content = ','.join(top_tags)
            else:
                # Split existing tags and add top tags if missing
                existing_tags = [tag.strip() for tag in content.split(',') if tag.strip()]
                all_tags = set(existing_tags)

                # Add top tags if not present
                for top_tag in top_tags:
                    all_tags.add(top_tag)

                # Sort and join
                new_content = ','.join(sorted(all_tags))

            # Write back if content changed
            if new_content != content:
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                files_updated += 1
                logging.debug("Updated %s with top tags", txt_file.name)

        except (OSError, IOError) as exc:
            logging.warning("Error updating %s with top tags: %s", txt_file.name, exc)

    logging.info("Top tag enforcement complete. Updated %d files", files_updated)


def _process_single_file(client: genai.Client, wav_file: Path, config: dict) -> tuple[str, bool]:
    """ Process a single WAV file and return (tags, success) """
    try:
        logging.info("Processing %s...", wav_file.name)

        # Upload file
        uploaded_file = _upload_with_retry(client, wav_file)

        # Generate tags
        tags = _generate_tags_with_retry(client, uploaded_file, config['gemini_prompt'], config['model'])

        # Normalize and deduplicate tags
        tag_list = [tag.strip().lower() for tag in tags.split(',') if tag.strip()]
        normalized_tags = ','.join(sorted(set(tag_list)))

        # Save to text file
        txt_file = wav_file.with_suffix('.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(normalized_tags)

        logging.info("Saved tags to %s", txt_file.name)

        # Delete uploaded file
        _delete_file_safe(client, uploaded_file.name)

        return normalized_tags, True

    except (UploadError, GenerationError) as exc:
        logging.error("Failed to process %s: %s", wav_file.name, exc)
        return "", False


def _parse_args() -> argparse.Namespace:
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(
        description="Generate audio tags using Gemini API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--number-to-process",
        type=int,
        help="Number of files to process before stopping (processes all if not specified)"
    )
    return parser.parse_args()


def main() -> None:
    """ Main processing function """
    # Parse command line arguments
    args = _parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Load configuration and API key
        config = _load_config()
        api_key = config['gemini_api_key']
        if not api_key:
            raise AuthenticationError("gemini_api_key not found in config.json")
        client = genai.Client(api_key=api_key)
        logging.info("Initialized Gemini client")

        # Scan directory for WAV files
        if not AUDIO_DIR.exists():
            logging.error("Audio directory not found: %s", AUDIO_DIR)
            sys.exit(1)

        wav_files = [f for f in AUDIO_DIR.iterdir()
                     if f.is_file() and f.suffix.lower() == '.wav' and not _has_paired_txt(f)]

        if not wav_files:
            logging.info("No WAV files found without paired TXT files")
            # Still proceed to consolidate existing tags

        # Sort files alphabetically for consistent processing order
        wav_files.sort(key=lambda f: f.name.lower())

        # Apply number limit if specified
        if args.number_to_process is not None:
            wav_files = wav_files[:args.number_to_process]
            logging.info("Limited to processing %d files as requested", args.number_to_process)

        logging.info("Found %d WAV files to process", len(wav_files))

        # Process files in parallel batches
        success_count = 0
        failure_count = 0
        all_tags = set()  # Use set to automatically deduplicate tags

        # First, collect tags from existing txt files
        logging.info("Collecting tags from existing txt files...")
        existing_txt_files = [f for f in AUDIO_DIR.iterdir() if f.is_file() and f.suffix.lower() == '.txt']

        for txt_file in existing_txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    # Collect tags (split by comma, clean, and normalize to lowercase)
                    tag_list = [tag.strip().lower() for tag in content.split(',') if tag.strip()]
                    all_tags.update(tag_list)
            except (OSError, IOError) as exc:
                logging.warning("Error reading existing txt file %s: %s", txt_file.name, exc)

        logging.info("Collected %d unique tags from existing files", len(all_tags))

        # Process files in batches of BATCH_SIZE
        for i in range(0, len(wav_files), BATCH_SIZE):
            batch = wav_files[i:i + BATCH_SIZE]
            logging.info("Processing batch %d/%d (%d files)",
                         i // BATCH_SIZE + 1,
                         (len(wav_files) + BATCH_SIZE - 1) // BATCH_SIZE,
                         len(batch))

            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
                # Submit all files in the batch
                future_to_file = {
                    executor.submit(_process_single_file, client, wav_file, config): wav_file
                    for wav_file in batch
                }

                # Collect results as they complete
                for future in as_completed(future_to_file):
                    wav_file = future_to_file[future]
                    try:
                        tags, success = future.result()
                        if success:
                            success_count += 1
                            # Collect tags for master list (split by comma, clean, and normalize to lowercase)
                            tag_list = [tag.strip().lower() for tag in tags.split(',') if tag.strip()]
                            all_tags.update(tag_list)
                        else:
                            failure_count += 1
                    except Exception as exc:  # pylint: disable=broad-except
                        # Catch any unexpected errors to continue processing other files
                        logging.error("Unexpected error processing %s: %s", wav_file.name, exc)
                        failure_count += 1

            logging.info("Batch complete. Success: %d, Failed: %d", success_count, failure_count)

        # Save deduplicated master tag list with frequency analysis
        if all_tags:
            master_tags_file = Path("data/all_tags.csv")
            master_tags_file.parent.mkdir(exist_ok=True)

            # Sort tags alphabetically for consistent output
            sorted_tags = sorted(all_tags, key=str.lower)

            # Generate frequency analysis
            frequency_analysis = _analyze_tag_frequency(all_tags)

            with open(master_tags_file, 'w', encoding='utf-8') as f:
                # Write only the frequency analysis
                f.write(frequency_analysis)

            logging.info("Saved %d unique tags with frequency analysis to %s", len(sorted_tags), master_tags_file)
        else:
            logging.info("No tags collected to save")

        # Tag normalization step
        logging.info("Starting tag normalization...")

        # Extract tags from CSV for normalization
        csv_tags = _extract_tags_from_csv()

        if len(csv_tags) < 30:
            logging.info("Skipping normalization: Only %d tags found (threshold: 30)", len(csv_tags))
        else:
            logging.info("Found %d tags, proceeding with normalization", len(csv_tags))

            # Generate normalization mapping using Gemini API
            normalization_map = _normalize_tags_with_gemini(client, csv_tags, config['model'])

            if normalization_map:
                # Apply normalization to all txt files
                _apply_tag_normalization(normalization_map)

                # Regenerate all_tags.csv with normalized tags
                logging.info("Regenerating all_tags.csv with normalized tags...")

                # Collect all tags from updated txt files
                updated_all_tags = set()
                txt_files = list(AUDIO_DIR.glob("*.txt"))

                for txt_file in txt_files:
                    try:
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            tag_list = [tag.strip().lower() for tag in content.split(',') if tag.strip()]
                            updated_all_tags.update(tag_list)
                    except (OSError, IOError) as exc:
                        logging.warning("Error reading %s for regeneration: %s", txt_file.name, exc)

                # Generate updated frequency analysis
                if updated_all_tags:
                    updated_frequency_analysis = _analyze_tag_frequency(updated_all_tags)

                    with open(master_tags_file, 'w', encoding='utf-8') as f:
                        f.write(updated_frequency_analysis)

                    logging.info("Regenerated all_tags.csv with %d normalized tags", len(updated_all_tags))

                    # Ensure top two tags are in all files
                    _ensure_top_tags_in_all_files()

                    # Final regeneration after top tag enforcement
                    logging.info("Final regeneration after top tag enforcement...")
                    final_all_tags = set()

                    for txt_file in txt_files:
                        try:
                            with open(txt_file, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                tag_list = [tag.strip().lower() for tag in content.split(',') if tag.strip()]
                                final_all_tags.update(tag_list)
                        except (OSError, IOError) as exc:
                            logging.warning("Error reading %s for final regeneration: %s", txt_file.name, exc)

                    if final_all_tags:
                        final_frequency_analysis = _analyze_tag_frequency(final_all_tags)

                        with open(master_tags_file, 'w', encoding='utf-8') as f:
                            f.write(final_frequency_analysis)

                        logging.info("Final all_tags.csv generated with %d tags", len(final_all_tags))
                else:
                    logging.warning("No tags found after normalization")
            else:
                logging.warning("Failed to generate normalization mapping, skipping normalization")

        # Print summary
        logging.info("Processing complete. Success: %d, Failed: %d", success_count, failure_count)

    except AuthenticationError as exc:
        logging.error("Authentication error: %s", exc)
        sys.exit(1)
    except (UploadError, GenerationError) as exc:
        logging.error("Processing error: %s", exc)
        sys.exit(1)
    except (OSError, IOError) as exc:
        logging.error("File system error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
