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

from results import ActionPlanningResult, ActionStep

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
    demonstration_bin_order: str = "left_to_right",
    perception_mode: str = "generalized",
) -> str:

    perception_mode = str(perception_mode).strip().lower()
    if perception_mode not in {"generalized", "prior_guided"}:
        raise ValueError(
            "Invalid perception_mode: "
            f"{perception_mode!r}"
        )

    demonstration_bin_order = (
        str(demonstration_bin_order)
        .strip()
        .lower()
    )

    if demonstration_bin_order == "left_to_right":
        bin_order_instruction = (
            "Only after fixing the destination track ID, sort all storage bins "
            "by their place-frame x coordinate in ascending order. "
            "The destination ordinal is the one-based position of the selected "
            "track ID in that sorted list: smallest x corresponds to the first "
            "storage bin from the left in the canonical front view, the next "
            "smallest x corresponds to the second, and so on. "
        )

    elif demonstration_bin_order == "right_to_left":
        bin_order_instruction = (
            "The demonstration is observed from the opposite side relative to "
            "the canonical front view. Only after fixing the destination track ID, "
            "sort all storage bins by their place-frame x coordinate in descending "
            "order. The destination ordinal is the one-based position of the "
            "selected track ID in that sorted list: largest x corresponds to the "
            "first storage bin from the left in the canonical front view, the next "
            "largest x corresponds to the second, and so on. "
        )

    else:
        raise ValueError(
            "Invalid demonstration_bin_order: "
            f"{demonstration_bin_order!r}"
        )
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
    generalized_prompt = (
        "Infer one pick-and-place action from the three chronologically ordered "
        "annotated frames and the tracking evidence below. The domain contains "
        "coloured manipulable objects and storage bins. Use each frame only for "
        "its assigned role. Treat the initial-scene frame as general scene context "
        "only; do not use it to decide the picked track ID or destination. In the "
        "pick-event frame, first identify the track ID of the object physically "
        "grasped or manipulated by the hand. Do not select an object merely because "
        "of its colour, type, or position. After selecting the picked track ID, "
        "obtain the picked object's semantic identity exclusively from that "
        "track ID's 'detector_label' in track_id_map. Manipulable object detector "
        "labels have the exact form '<colour> <object type>', for example "
        "'green cube', 'blue ring', or 'red cylinder'. The colour is the first "
        "word and the object type is the remaining part of the detector label. "
        "Never select a 'storage bin' as the picked object. Do not visually infer, "
        "verify, or change the object's colour or type from any video frame. "
        "The GroundingDINO detector label is the authoritative source for both "
        "picked_color and picked_category: copy the colour and object type from it. "
        "In the place-event frame, first obtain the centre of the picked object "
        "and the centre of every object labelled 'storage bin' from that frame's "
        "coordinates. For the relation 'in', select the storage-bin track ID that "
        "visually receives or contains the picked object and has the smallest "
        "centre-to-centre distance from it. When coordinates are available, "
        "compare all candidate distances and do not select a farther bin unless "
        "the image clearly contradicts the coordinates. "
        + bin_order_instruction
        + "Never infer the "
        "destination or its ordinal from track ID, JSON order, or list order. If "
        "the picked object has no place-frame coordinates, or image and coordinates "
        "conflict, report ambiguity. Track IDs must be copied from the "
        "annotations/evidence. The relation for placing an object inside a storage "
        "bin is 'in'. Do not infer anything from a task or trajectory name. If "
        "evidence is insufficient, set status to 'ambiguous' and explain why in "
        "ambiguities. The action field must be a complete, natural-language "
        "imperative sentence that explicitly names the picked object's labelled "
        "colour and category and the destination container's ordinal position "
        "from the left. Never use a generic label such as 'pick-and-place'. "
        "Follow this form: 'Pick the <colour> <picked category> and place it into "
        "the <ordinal> <destination category> from the left.' "
        "Return only data matching the requested JSON schema."
    )

    prior_guided_prompt = (
        "Infer one pick-and-place action from the three chronologically ordered "
        "annotated frames and the tracking evidence below. The domain contains "
        "cubes and storage bins. Use each frame only for its assigned role. Treat the "
        "initial-scene frame as general scene context only; do not use it to decide "
        "the picked track ID or destination. In the pick-event frame, "
        "first identify the track ID of the cube physically grasped or manipulated by "
        "the hand. Do not select a cube merely because of its colour or position. "
        "After selecting the picked track ID, obtain the picked object's semantic "
        "identity exclusively from that track ID's 'detector_label' in track_id_map. "
        "Cube detector labels already contain the semantic colour in the exact form "
        "'<colour> cube', for example 'red cube', 'green cube', 'blue cube', or "
        "'yellow cube'. Do not visually infer, verify, or change the cube colour from "
        "any video frame. The GroundingDINO detector label is the authoritative source "
        "for the picked object's colour and category. Set picked_color to the colour "
        "contained in that detector label and picked_category to 'cube'. In the "
        "place-event frame, first obtain the centre of the picked cube and the centre "
        "of every object labelled 'storage bin' from that frame's coordinates. For the "
        "relation 'in', select the storage-bin track ID that visually receives or "
        "contains the cube and has the smallest centre-to-centre distance from it. "
        "When coordinates are available, compare all candidate distances and do not "
        "select a farther bin unless the image clearly contradicts the coordinates. "
        + bin_order_instruction
        + "Never infer the destination or its ordinal from track ID, JSON order, "
        "or list order. If the picked cube has no place-frame coordinates, or image "
        "and coordinates conflict, report ambiguity. Track IDs must be copied from "
        "the annotations/evidence. The relation for placing a cube inside a storage "
        "bin is 'in'. Do not infer anything from a task or trajectory name. If "
        "evidence is insufficient, set status to 'ambiguous' and explain why in "
        "ambiguities. The action field must be a complete, natural-language "
        "imperative sentence that explicitly names the picked object's labelled "
        "colour and category and the destination container's ordinal position "
        "from the left. Never use a generic label such as 'pick-and-place'. "
        "Follow this form: 'Pick the <colour> <picked category> and place it into "
        "the <ordinal> <destination category> from the left.' "
        "Return only data matching the requested JSON schema."
    )

    # ---------------------------------------------------------
    # Select action-planning prompt
    # ---------------------------------------------------------

    if perception_mode == "generalized":
        prompt = generalized_prompt
    else:
        prompt = prior_guided_prompt

    return (
        prompt
        + "\n\nTracking evidence:\n"
        + json.dumps(
            evidence,
            indent=2,
            sort_keys=True,
        )
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

    picked_track_id = str(step["picked_track_id"])
    picked_info = track_map[picked_track_id]

    detector_label = str(
        picked_info.get("detector_label", "")
    ).strip().lower()

    # The detector label is authoritative and follows
    # "<colour> <object type>" (e.g. "green cube", "blue ring").
    # Storage bins are destinations, never manipulable picked objects.
    label_parts = detector_label.split(maxsplit=1)

    if detector_label == "storage bin" or len(label_parts) != 2:
        raise ValueError(
            "Picked track does not have a valid semantic object detector label "
            "in '<colour> <object type>' format: "
            f"{detector_label!r}"
        )

    expected_color, expected_category = label_parts

    if step.get("picked_category", "").strip().lower() != expected_category:
        raise ValueError(
            "picked_category does not match the GroundingDINO detector label: "
            f"expected={expected_category!r}, "
            f"received={step.get('picked_category')!r}"
        )

    if step.get("picked_color", "").strip().lower() != expected_color:
        raise ValueError(
            "picked_color does not match the GroundingDINO detector label: "
            f"expected={expected_color!r}, "
            f"received={step.get('picked_color')!r}"
        )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

def generate_action_plan(
    annotated_video_path: str | Path,
    keyframes: tuple[int, int],
    track_id_map: dict[int, dict[str, object]],
    key_frame_coordinates: dict[str, list[str]],
    artifacts_dir: str | Path,
    model: str = DEFAULT_MODEL,
    demonstration_bin_order: str = "left_to_right",
    dry_run: bool = False,
    perception_mode: str = "generalized",
) -> ActionPlanningResult:
    """Generate a structured action plan from visual-prompting outputs."""

    video_path = Path(annotated_video_path).expanduser().resolve()
    output_dir = Path(artifacts_dir).expanduser().resolve()

    perception_mode = str(perception_mode).strip().lower()

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Annotated video does not exist: {video_path}"
        )

    if video_path.stat().st_size == 0:
        raise ValueError(
            f"Annotated video is empty: {video_path}"
        )

    normalized_keyframes = tuple(int(frame) for frame in keyframes)

    if len(normalized_keyframes) != 2:
        raise ValueError(
            "Action planning requires exactly two keyframes: "
            "one pick frame and one place frame."
        )

    pick_frame, place_frame = normalized_keyframes

    if pick_frame < 0 or place_frame < 0:
        raise ValueError(
            f"Keyframes cannot be negative: {normalized_keyframes}"
        )

    if pick_frame >= place_frame:
        raise ValueError(
            "The pick keyframe must precede the place keyframe: "
            f"{normalized_keyframes}"
        )

    if not track_id_map:
        raise ValueError(
            "track_id_map cannot be empty."
        )

    if not key_frame_coordinates:
        raise ValueError(
            "key_frame_coordinates cannot be empty."
        )

    required_coordinate_keys = {
        f"key_frame{pick_frame}",
        f"key_frame{place_frame}",
    }

    missing_coordinate_keys = (
        required_coordinate_keys
        - set(key_frame_coordinates.keys())
    )

    if missing_coordinate_keys:
        raise ValueError(
            "Missing coordinates for keyframes: "
            f"{sorted(missing_coordinate_keys)}"
        )

    normalized_track_map = {
        str(track_id): dict(track_info)
        for track_id, track_info in track_id_map.items()
    }

    normalized_coordinates: dict[str, dict[str, list[int]]] = {}

    for frame_index in normalized_keyframes:
        frame_key = f"key_frame{frame_index}"
        frame_coordinates: dict[str, list[int]] = {}

        for coordinate in key_frame_coordinates[frame_key]:
            match = COORD_RE.fullmatch(coordinate.strip())

            if match is None:
                raise ValueError(
                    "Invalid key-frame coordinate entry: "
                    f"{coordinate!r}"
                )

            frame_coordinates[match.group("id")] = [
                int(match.group("x")),
                int(match.group("y")),
            ]

        if not frame_coordinates:
            raise ValueError(
                f"No valid coordinates found for {frame_key}."
            )

        normalized_coordinates[str(frame_index)] = (
            frame_coordinates
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered_frames = [0, pick_frame, place_frame]

    frame_count, fps = inspect_video(
        video_path,
        ordered_frames,
    )

    prompt = build_prompt(
        pick_frame=pick_frame,
        place_frame=place_frame,
        track_map=normalized_track_map,
        coordinates=normalized_coordinates,
        demonstration_bin_order=demonstration_bin_order,
        perception_mode=perception_mode,
    )

    manifest = {
        "status": (
            "dry_run_validated"
            if dry_run
            else "ready_for_openai"
        ),
        "model": model,
        "perception_mode": perception_mode,
        "demonstration_bin_order": demonstration_bin_order,
        "input_video": str(video_path),
        "input_sha256": sha256_file(video_path),
        "frame_roles": {
            "initial": 0,
            "pick": pick_frame,
            "place": place_frame,
        },
        "video_frame_count": frame_count,
        "video_fps": fps,
        "track_ids": sorted(
            int(track_id)
            for track_id in normalized_track_map
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    write_json(
        output_dir / "input_manifest.json",
        manifest,
    )

    (output_dir / "prompt.txt").write_text(
        prompt + "\n"
    )

    if dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        print("Dry run completed: no OpenAI request was made.")

        return ActionPlanningResult(
            steps=(),
            status="ambiguous",
            ambiguities=(
                "Dry run completed without generating an action plan.",
            ),
            natural_language_plan=(
                "No action plan generated during dry run."
            ),
        )

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is not configured."
        )

    try:
        frames = extract_frames(
            video_path,
            ordered_frames,
        )

        messages = build_messages(
            prompt,
            frames,
            ordered_frames,
        )

        from openai import OpenAI

        response = OpenAI().chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=800,
            response_format={
                "type": "json_schema",
                "json_schema": PLAN_SCHEMA,
            },
        )

        raw_output = response.choices[0].message.content

        if not raw_output:
            refusal = getattr(
                response.choices[0].message,
                "refusal",
                None,
            )

            raise ValueError(
                "OpenAI returned no plan. "
                f"Refusal: {refusal}"
            )

        plan = json.loads(raw_output)

        validate_plan(
            plan=plan,
            pick_frame=pick_frame,
            place_frame=place_frame,
            track_map=normalized_track_map,
        )

    except Exception as error:
        manifest["status"] = "failed"
        manifest["error_type"] = type(error).__name__
        manifest["error"] = str(error)

        write_json(
            output_dir / "input_manifest.json",
            manifest,
        )

        raise

    manifest["status"] = "completed"
    manifest["openai_response_id"] = response.id
    manifest["usage"] = (
        response.usage.model_dump()
        if response.usage
        else None
    )

    write_json(
        output_dir / "input_manifest.json",
        manifest,
    )

    (output_dir / "raw_response.json").write_text(
        raw_output.rstrip() + "\n"
    )

    write_json(
        output_dir / "action_plan.json",
        plan,
    )

    action_steps = tuple(
        ActionStep(
            pick_keyframe=int(step["pick_keyframe"]),
            place_keyframe=int(step["place_keyframe"]),
            picked_track_id=int(step["picked_track_id"]),
            picked_category=str(step["picked_category"]),
            picked_color=str(step["picked_color"]),
            destination_track_id=int(
                step["destination_track_id"]
            ),
            destination_category=str(
                step["destination_category"]
            ),
            destination_ordinal_from_left=int(
                step["destination_ordinal_from_left"]
            ),
            relation=str(step["relation"]),
            action=str(step["action"]),
        )
        for step in plan["steps"]
    )

    if action_steps:
        natural_language_plan = " and then ".join(
            step.action
            for step in action_steps
        )
    else:
        natural_language_plan = (
            "No action plan generated: "
            + "; ".join(plan["ambiguities"])
        )

    (output_dir / "action_plan.txt").write_text(
        natural_language_plan + "\n"
    )

    result = ActionPlanningResult(
        steps=action_steps,
        status=str(plan["status"]),
        ambiguities=tuple(
            str(item)
            for item in plan["ambiguities"]
        ),
        natural_language_plan=natural_language_plan,
    )

    print(
        "Generated natural-language plan: "
        f"{result.natural_language_plan}"
    )
    print(f"Action plan written to {output_dir}")

    return result

def run(args: argparse.Namespace) -> int:
    video_path = args.input.expanduser().resolve()
    keyframe_log = args.keyframe_log.expanduser().resolve()
    visual_log = args.visual_prompting_log.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    for path in (
        video_path,
        keyframe_log,
        visual_log,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Input file does not exist: {path}"
            )

        if path.stat().st_size == 0:
            raise ValueError(
                f"Input file is empty: {path}"
            )

    pick_frame, place_frame = read_selected_keyframes(
        keyframe_log
    )

    track_map, coordinates = read_visual_prompting_log(
        visual_log
    )

    key_frame_coordinates: dict[str, list[str]] = {}

    for frame_index in (
        pick_frame,
        place_frame,
    ):
        frame_key = str(frame_index)

        if frame_key not in coordinates:
            raise ValueError(
                "Visual-prompting coordinates are missing "
                f"for frame {frame_index}."
            )

        key_frame_coordinates[
            f"key_frame{frame_index}"
        ] = [
            (
                f"Object {track_id}: "
                f"({position[0]}, {position[1]})"
            )
            for track_id, position
            in coordinates[frame_key].items()
        ]

    generate_action_plan(
        annotated_video_path=video_path,
        keyframes=(
            pick_frame,
            place_frame,
        ),
        track_id_map={
            int(track_id): dict(track_info)
            for track_id, track_info
            in track_map.items()
        },
        key_frame_coordinates=key_frame_coordinates,
        artifacts_dir=output_dir,
        model=args.model,
        demonstration_bin_order=args.demonstration_bin_order,
        dry_run=args.dry_run,
    )

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
        "--demonstration-bin-order",
        choices=[
            "left_to_right",
            "right_to_left",
        ],
        default="left_to_right",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate local inputs without reading the API key or calling OpenAI",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
