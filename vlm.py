"""Generate a structured SeeDo action plan from visual-prompting results.

Frame numbers, track IDs, colours, and destinations are never derived from the
task name. Use --dry-run to validate local inputs without calling OpenAI.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

DEFAULT_MODEL = "gpt-4o-2024-08-06"
KEYFRAME_RE = re.compile(r"The selected valley frames are:\s*\[([^\]]*)\]")
TRACK_MAP_RE = re.compile(r"^TRACK_ID_MAP:\s*(\{.*\})\s*$", re.MULTILINE)
COORD_BLOCK_RE = re.compile(
    r"key_frame(?P<frame>\d+)\s*\n"
    r"(?P<body>(?:Object\s+\d+:\s*\([^\n]+\)\s*\n?)+)"
)
COORD_RE = re.compile(
    r"Object\s+(?P<id>\d+):\s*\((?P<x>-?\d+),\s*(?P<y>-?\d+)\)"
)

PLAN_SCHEMA: dict[str, Any] = {
    "name": "seedo_action_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "minItems": 0,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "pick_keyframe": {"type": "integer"},
                        "place_keyframe": {"type": "integer"},
                        "picked_track_id": {"type": "integer"},
                        "picked_category": {"type": "string"},
                        "picked_color": {"type": "string"},
                        "destination_track_id": {"type": "integer"},
                        "destination_category": {"type": "string"},
                        "destination_ordinal_from_left": {"type": "integer"},
                        "relation": {"type": "string"},
                        "action": {"type": "string"},
                    },
                    "required": [
                        "pick_keyframe", "place_keyframe", "picked_track_id",
                        "picked_category", "picked_color", "destination_track_id",
                        "destination_category", "destination_ordinal_from_left",
                        "relation", "action",
                    ],
                    "additionalProperties": False,
                },
            },
            "status": {"type": "string", "enum": ["completed", "ambiguous"]},
            "ambiguities": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["steps", "status", "ambiguities"],
        "additionalProperties": False,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_selected_keyframes(log_path: Path) -> list[int]:
    matches = KEYFRAME_RE.findall(log_path.read_text(errors="replace"))
    if not matches:
        raise ValueError(f"No selected-valley result found in {log_path}")
    frames = [int(value) for value in re.findall(r"\d+", matches[-1])]
    if len(frames) != 2:
        raise ValueError(f"Expected one pick and one place frame, found {frames}")
    if frames[0] <= 0 or frames[0] >= frames[1]:
        raise ValueError(f"Invalid or unordered pick/place frames: {frames}")
    return frames


def read_visual_prompting_log(log_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    text = log_path.read_text(errors="replace")
    map_matches = TRACK_MAP_RE.findall(text)
    if not map_matches:
        raise ValueError(f"TRACK_ID_MAP is missing from {log_path}")
    track_map = json.loads(map_matches[-1])
    coordinates: dict[str, dict[str, list[int]]] = {}
    for match in COORD_BLOCK_RE.finditer(text):
        coordinates[match.group("frame")] = {
            item.group("id"): [int(item.group("x")), int(item.group("y"))]
            for item in COORD_RE.finditer(match.group("body"))
        }
    if not coordinates:
        raise ValueError(f"Keyframe coordinates are missing from {log_path}")
    return track_map, coordinates


def inspect_video(video_path: Path, requested_frames: list[int]) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    invalid = [frame for frame in requested_frames if frame < 0 or frame >= frame_count]
    if invalid:
        raise ValueError(
            f"Frame indexes {invalid} are outside video range 0..{frame_count - 1}"
        )
    return frame_count, fps


def extract_frames(video_path: Path, frame_indexes: list[int]) -> dict[int, bytes]:
    capture = cv2.VideoCapture(str(video_path))
    encoded: dict[int, bytes] = {}
    try:
        for frame_index in frame_indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot read frame {frame_index} from {video_path}")
            ok, image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise ValueError(f"Cannot encode frame {frame_index}")
            encoded[frame_index] = image.tobytes()
    finally:
        capture.release()
    return encoded


def image_data_url(image: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")


def build_prompt(
    pick_frame: int,
    place_frame: int,
    track_map: dict[str, Any],
    coordinates: dict[str, Any],
) -> str:
    evidence = {
        "frame_roles": {
            "0": "initial scene",
            str(pick_frame): "pick event",
            str(place_frame): "place event",
        },
        "track_id_map": track_map,
        "keyframe_center_coordinates": {
            str(pick_frame): coordinates[str(pick_frame)],
            str(place_frame): coordinates[str(place_frame)],
        },
    }
    return (
        "Infer one pick-and-place action from the three chronologically ordered "
        "annotated frames and the tracking evidence below. The domain contains "
        "cubes and storage bins. Use each frame only for its assigned role. Treat the "
        "initial-scene frame as general scene context only; do not use it to decide "
        "the picked track ID or destination. In the pick-event frame, "
        "first identify the track ID of the cube physically grasped or manipulated by "
        "the hand. Do not select a cube merely because of its colour or position. "
        "After selecting that track ID, determine its colour exclusively from the "
        "visible physical surface of that same cube in the pick-event frame. "
        "If, and only if, that same track ID is fully occluded by the hand/gripper or "
        "otherwise has no visible surface in the pick-event frame, then and only then "
        "read the colour from that identical track ID in the initial-scene frame instead. "
        "Never use the initial-scene frame for colour if any part of the cube's surface "
        "is visible in the pick-event frame. Never use the initial-scene frame to change "
        "which track ID was selected — the track ID is fixed before this colour step and "
        "never re-derived from the initial-scene frame. If the cube's surface is not "
        "visible in either frame, report ambiguity. "
        "The bright green contour drawn around every tracked object is "
        "an artificial annotation, not an object colour. Ignore contour pixels and "
        "never classify a cube as green merely because its contour is green. In the "
        "place-event frame, first obtain the centre of the picked cube and the centre "
        "of every object labelled 'storage bin' from that frame's coordinates. For the "
        "relation 'in', select the storage-bin track ID that visually receives or "
        "contains the cube and has the smallest centre-to-centre distance from it. "
        "When coordinates are available, compare all candidate distances and do not "
        "select a farther bin unless the image clearly contradicts the coordinates. "
        "Only after fixing the destination track ID, sort all storage bins by their "
        "place-frame x coordinate in ascending order. The destination ordinal is the "
        "one-based position of the selected track ID in that sorted list: smallest x "
        "is first from left, next is second from left, and so on. Never infer the "
        "destination or its ordinal from track ID, JSON order, or list order. If the "
        "picked cube has no place-frame coordinates, or image and coordinates conflict, "
        "report ambiguity. Track IDs must be copied from the annotations/evidence. "
        "The relation for placing a cube inside a storage bin is 'in'. Do not infer "
        "anything from a task or trajectory name. If evidence is insufficient, set "
        "status to 'ambiguous' and explain why in ambiguities. The action field "
        "must be a complete, natural-language imperative sentence that explicitly "
        "names the picked object's visible colour and category and the destination "
        "container's ordinal position from the left. Never use a generic label such "
        "as 'pick-and-place'. Follow this form: 'Pick the <colour> <picked category> "
        "and place it into the <ordinal> <destination category> from the left.' "
        "Return only data matching the requested JSON schema.\n\nTracking evidence:\n"
        + json.dumps(evidence, indent=2, sort_keys=True)
    )


def build_messages(
    prompt: str, frames: dict[int, bytes], ordered_frames: list[int]
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    roles = ("initial scene", "pick event", "place event")
    for role, frame_index in zip(roles, ordered_frames, strict=True):
        content.append({"type": "text", "text": f"Frame {frame_index}: {role}."})
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_data_url(frames[frame_index]),
                "detail": "high",
            },
        })
    return [
        {
            "role": "system",
            "content": (
                "You are the visual reasoning stage of a robot imitation system. "
                "Use only the supplied visual and tracking evidence. Follow the "
                "required procedure in order and preserve track identity across frames."
            ),
        },
        {"role": "user", "content": content},
    ]


def validate_plan(
    plan: dict[str, Any],
    pick_frame: int,
    place_frame: int,
    track_map: dict[str, Any],
) -> None:
    steps = plan.get("steps")
    if not isinstance(steps, list):
        raise ValueError("The plan steps must be a list")
    if plan.get("status") == "ambiguous" and not steps:
        if not plan.get("ambiguities"):
            raise ValueError("An ambiguous plan must explain its ambiguities")
        return
    if plan.get("status") != "completed" or len(steps) != 1:
        raise ValueError("A completed plan must contain exactly one action step")
    step = steps[0]
    if step.get("pick_keyframe") != pick_frame:
        raise ValueError("The returned pick keyframe does not match the input")
    if step.get("place_keyframe") != place_frame:
        raise ValueError("The returned place keyframe does not match the input")
    known_ids = {int(track_id) for track_id in track_map}
    for field in ("picked_track_id", "destination_track_id"):
        if step.get(field) not in known_ids:
            raise ValueError(f"Unknown {field}: {step.get(field)}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> int:
    video_path = args.input.resolve()
    keyframe_log = args.keyframe_log.resolve()
    visual_log = args.visual_prompting_log.resolve()
    output_dir = args.output_dir.resolve()
    for path in (video_path, keyframe_log, visual_log):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty input: {path}")

    pick_frame, place_frame = read_selected_keyframes(keyframe_log)
    ordered_frames = [0, pick_frame, place_frame]
    track_map, coordinates = read_visual_prompting_log(visual_log)
    missing = [frame for frame in (pick_frame, place_frame) if str(frame) not in coordinates]
    if missing:
        raise ValueError(f"Visual-prompting coordinates are missing for frames {missing}")
    frame_count, fps = inspect_video(video_path, ordered_frames)
    prompt = build_prompt(pick_frame, place_frame, track_map, coordinates)
    manifest = {
        "status": "dry_run_validated" if args.dry_run else "ready_for_openai",
        "model": args.model,
        "input_video": str(video_path),
        "input_sha256": sha256_file(video_path),
        "keyframe_log": str(keyframe_log),
        "visual_prompting_log": str(visual_log),
        "frame_roles": {"initial": 0, "pick": pick_frame, "place": place_frame},
        "video_frame_count": frame_count,
        "video_fps": fps,
        "track_ids": sorted(int(track_id) for track_id in track_map),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        print("Dry run completed: no OpenAI request was made.")
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not configured")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "input_manifest.json", manifest)
    (output_dir / "prompt.txt").write_text(prompt + "\n")
    try:
        frames = extract_frames(video_path, ordered_frames)
        messages = build_messages(prompt, frames, ordered_frames)
        from openai import OpenAI

        response = OpenAI().chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=0,
            max_tokens=800,
            response_format={"type": "json_schema", "json_schema": PLAN_SCHEMA},
        )
        raw_output = response.choices[0].message.content
        if not raw_output:
            refusal = getattr(response.choices[0].message, "refusal", None)
            raise ValueError(f"OpenAI returned no plan. Refusal: {refusal}")
        plan = json.loads(raw_output)
        validate_plan(plan, pick_frame, place_frame, track_map)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error_type"] = type(error).__name__
        manifest["error"] = str(error)
        write_json(output_dir / "input_manifest.json", manifest)
        raise

    manifest["status"] = "completed"
    manifest["openai_response_id"] = response.id
    manifest["usage"] = response.usage.model_dump() if response.usage else None
    write_json(output_dir / "input_manifest.json", manifest)
    (output_dir / "raw_response.json").write_text(raw_output.rstrip() + "\n")
    write_json(output_dir / "action_plan.json", plan)
    if plan["steps"]:
        natural_language_plan = " and then ".join(
            step["action"] for step in plan["steps"]
        )
    else:
        natural_language_plan = "No action plan generated: " + "; ".join(
            plan["ambiguities"]
        )
    (output_dir / "action_plan.txt").write_text(natural_language_plan + "\n")
    print(f"Generated natural-language plan: {natural_language_plan}")
    print(f"Action plan written to {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a SeeDo action plan from one visual-prompting result."
    )
    parser.add_argument("--input", type=Path, required=True, help="Annotated tracked video")
    parser.add_argument(
        "--keyframe-log", type=Path, required=True,
        help="Keyframe-selection log containing the final pick/place frames",
    )
    parser.add_argument(
        "--visual-prompting-log", type=Path, required=True,
        help="Visual-prompting log containing TRACK_ID_MAP and coordinates",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate local inputs without reading the API key or calling OpenAI",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
