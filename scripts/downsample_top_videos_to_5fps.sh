#!/usr/bin/env bash

set -Eeuo pipefail

# Default input directory requested for the RW-RL-Dataset top-camera videos.
SOURCE_DIR="/Users/hongru/Downloads/RW-RL-Dataset/piperx_insert_copper_screw/videos/observation.images.camera_top/chunk-000"
OUTPUT_DIR="${SOURCE_DIR}/5fps"
OVERWRITE=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/downsample_top_videos_to_5fps.sh [--overwrite]

The script converts all MP4 files directly under the configured source directory
from their original frame rate to constant 5 FPS and writes them to the 5fps/
subdirectory. Existing output files are skipped unless --overwrite is provided.
EOF
}

for arg in "$@"; do
    case "$arg" in
        --overwrite)
            OVERWRITE=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg is not installed or is not in PATH." >&2
    exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Error: source directory does not exist:" >&2
    echo "  $SOURCE_DIR" >&2
    exit 1
fi

case "$SOURCE_DIR" in
    /Users/*/Downloads/*) ;;
    *)
        echo "Error: SOURCE_DIR must be an absolute macOS path beginning with /Users/." >&2
        printf 'Received: %q\n' "$SOURCE_DIR" >&2
        exit 1
        ;;
esac

echo "Input directory: $SOURCE_DIR"
echo "Output directory: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

processed=0
skipped=0
failed=0

shopt -s nullglob
input_files=("${SOURCE_DIR}"/*.mp4)

if [[ "${#input_files[@]}" -eq 0 ]]; then
    echo "Error: no MP4 files found in SOURCE_DIR." >&2
    exit 1
fi

for input_file in "${input_files[@]}"; do
    filename="$(basename "$input_file")"
    output_file="${OUTPUT_DIR}/${filename}"

    if [[ -e "$output_file" && "$OVERWRITE" -eq 0 ]]; then
        echo "Skip existing: $filename"
        skipped=$((skipped + 1))
        continue
    fi

    printf 'Converting: %s\n' "$input_file"
    temporary_output="${output_file}.part"

    # The video stream must be re-encoded because its frame rate changes.
    # Audio, subtitles, data streams, metadata, chapters, resolution, aspect
    # ratio, and rotation metadata are retained when present.
    if ffmpeg \
        -hide_banner \
        -loglevel error \
        -y \
        -i "$input_file" \
        -map 0 \
        -map_metadata 0 \
        -map_chapters 0 \
        -vf "fps=5" \
        -fps_mode cfr \
        -c:v libx264 \
        -preset medium \
        -crf 18 \
        -c:a copy \
        -c:s copy \
        -c:d copy \
        -movflags +faststart \
        -f mp4 \
        "$temporary_output"; then
        mv -f "$temporary_output" "$output_file"
        processed=$((processed + 1))
    else
        rm -f "$temporary_output"
        echo "Failed: $filename" >&2
        failed=$((failed + 1))
    fi
done

echo
echo "Finished. Converted: $processed, skipped: $skipped, failed: $failed"
echo "Output directory: $OUTPUT_DIR"

if [[ "$failed" -ne 0 ]]; then
    exit 1
fi
