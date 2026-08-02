#!/usr/bin/env python3
"""Convert a video to an H.264 MP4 suitable for VS Code preview."""

# cd /home/mivia/Desktop/UR-Application

# docker run --rm \
#   --mount type=bind,src="$PWD/SeeDo/convert_video_h264.py",dst=/home/seedo/convert_video_h264.py,readonly \
#   --mount type=bind,src="$PWD/testdata/seedo",dst=/testdata \
#   seedo:baseline \
#   python /home/seedo/convert_video_h264.py \
#     --input /testdata/visual-prompting/short_demo1-tracked.mp4 \
#     --output-dir /testdata/visual-prompting

import argparse
import shutil
import subprocess
from pathlib import Path


def convert_video(input_path: Path, output_dir: Path) -> Path:
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found in PATH")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}-h264.mp4"

    if output_path == input_path:
        raise ValueError("Input and output paths must be different")
    if output_path.exists():
        raise FileExistsError(f"Output video already exists: {output_path}")

    command = [
        "ffmpeg",
        "-n",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a video to an H.264 MP4 suitable for VS Code preview."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input video path")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where the converted video will be saved",
    )
    args = parser.parse_args()

    output_path = convert_video(args.input, args.output_dir)
    print(f"Converted video: {output_path}")


if __name__ == "__main__":
    main()
