import os
import sys
import argparse
import copy
import gc
import json
import re
import csv
import base64
from io import BytesIO
from collections import Counter
import ast

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert
from tqdm import tqdm
import cv2
import scipy.signal
import matplotlib.pyplot as plt

# from diffusers import StableDiffusionInpaintPipeline
from sam2.build_sam import build_sam2_video_predictor

# Grounding DINO
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import (
    annotate,
    load_image,
    predict,
    load_image_from_array,
)

# Segment Anything
from segment_anything import build_sam, SamPredictor

# Hugging Face Hub
from huggingface_hub import hf_hub_download

from results import VisualPromptingResult
from pathlib import Path
import subprocess

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

from ai_controller.models.seedo_controller.timing_utils import TIMING


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def call_openai_api(prompt_messages, client):
    params = {
        "model": "gpt-4o",
        "messages": prompt_messages,
        "max_tokens": 400,
        "temperature": 0,
    }
    result = client.chat.completions.create(**params)
    return result.choices[0].message.content


def get_object_list(video_path, client):
    # Use the first frame for encoding
    video = cv2.VideoCapture(video_path)

    base64Frames = []
    frame_count = 0
    max_frames = 2  # Only process the first 2 frames

    while video.isOpened() and frame_count < max_frames:
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        frame_count += 1

    video.release()
    print(len(base64Frames), "frames read.")

    prompt_messages_state = [
        {
            "role": "system",
            "content": [
                (
                    "You are a visual object detector whose output will be used "
                    "directly as text queries for GroundingDINO."
                ),
                (
                    "The scene contains colored cubes and storage bins. "
                    "For every cube, include its visible color in the detector "
                    "label using the exact form '<color> cube'. "
                    "The only valid cube colors in this benchmark are:"
                    "red, green, blue, and yellow."
                    "For every cube, the detector label MUST therefore be exactly one of:"
                    "'red cube', 'green cube', 'blue cube', or 'yellow cube'."
                    "Do not use any other cube color.\n"
                    "For every storage bin, always use the exact detector label "
                    "'storage bin' without adding color, position, material, "
                    "or other attributes."
                ),
            ],
        },
        {
            "role": "user",
            "content": [
                "Inspect the physical objects visible on the desk and classify them using these rules:",
                "1. For every visible cube or graspable colored block, identify its visible color and return '<color> cube'.",
                "2. Examples of valid cube labels are 'red cube', 'green cube', 'blue cube', and 'yellow cube'.",
                "3. For every bin, box, tray, container, or receptacle, return exactly 'storage bin'.",
                "4. Count every physical instance separately.",
                "5. Repeat 'storage bin' once for every visible storage-bin instance.",
                "6. Do not include hands, grippers, people, the table, or background objects.",
                "7. Do not assign spatial descriptions such as 'first bin from the left' or 'second bin from the left'.",
                "8. Do not add objects that are not visible.",
                "Return exactly two lines and no additional explanation:",
                "Number: <total number of instances>",
                "Objects: <comma-separated detector labels, repeated once per instance>",
                "Example:",
                "Number: 6",
                "Objects: red cube, green cube, storage bin, storage bin, storage bin, storage bin",
                *map(
                    lambda x: {"image": x, "resize": 768},
                    base64Frames[0:1],
                ),
            ],
        },
    ]

    response_state = call_openai_api(prompt_messages_state, client)
    return response_state


def extract_num_object(response_state):
    # Extract number of objects
    num_match = re.search(r"Number: (\d+)", response_state)
    num = int(num_match.group(1)) if num_match else 0

    # Extract objects
    objects_match = re.search(r"Objects: (.+)", response_state)
    objects_list = objects_match.group(1).split(", ") if objects_match else []

    # Construct object list
    objects = [obj for obj in objects_list]

    return num, objects


def parse_object_list(objects):
    object_list = [obj.strip() for obj in objects.split(",") if obj.strip()]
    if not object_list:
        raise ValueError("--objects must contain at least one object name")
    return object_list


def filter_oversized_storage_bin_detections(
    boxes, logits, phrases, detector_label, width_ratio=1.8, min_candidates=3
):
    diagnostics = {
        "applied": False,
        "width_ratio": width_ratio,
        "candidate_count": int(boxes.shape[0]),
        "removed_count": 0,
        "removed_indices": [],
    }
    if detector_label != "storage bin" or boxes.shape[0] < min_candidates:
        return boxes, logits, phrases, diagnostics

    widths = boxes[:, 2]
    median_width = torch.median(widths)
    if median_width.item() <= 0:
        return boxes, logits, phrases, diagnostics

    width_threshold = median_width * width_ratio
    keep_mask = widths <= width_threshold
    removed_indices = (
        (~keep_mask).nonzero(as_tuple=False).flatten().detach().cpu().tolist()
    )
    diagnostics.update(
        {
            "applied": True,
            "median_width": float(median_width.item()),
            "width_threshold": float(width_threshold.item()),
            "candidate_widths": [
                float(value) for value in widths.detach().cpu().tolist()
            ],
            "removed_count": len(removed_indices),
            "removed_indices": removed_indices,
        }
    )
    if not removed_indices:
        return boxes, logits, phrases, diagnostics

    keep_values = keep_mask.detach().cpu().tolist()
    filtered_phrases = [
        phrase for phrase, keep in zip(phrases, keep_values) if keep
    ]
    return boxes[keep_mask], logits[keep_mask], filtered_phrases, diagnostics


def load_model_hf(
    repo_id, filename, ckpt_config_filename, device="cpu", text_encoder_path=None
):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    return load_groundingdino_model(
        cache_config_file, cache_file, device, text_encoder_path
    )


def load_groundingdino_model(
    config_file, checkpoint_file, device="cpu", text_encoder_path=None
):
    args = SLConfig.fromfile(config_file)
    if text_encoder_path is not None:
        args.text_encoder_type = text_encoder_path
    args.device = device
    model = build_model(args)

    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("Model loaded from {} \n => {}".format(checkpoint_file, log))
    _ = model.eval()
    return model


def read_video(video_path):
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print("Error: Could not open video.")
        exit()

    frames = []

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    return frames


def my_annotate(
    image_source: np.ndarray,
    boxes: torch.Tensor,
    logits: torch.Tensor,
    phrases,
) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)

    for box, logit, phrase in zip(xyxy, logits, phrases):
        x1, y1, x2, y2 = map(int, box)
        label = f"{phrase} {logit:.2f}"

        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label background box
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            annotated_frame,
            (x1, y1 - text_height - 4),
            (x1 + text_width, y1),
            (0, 255, 0),
            -1,
        )

        # Draw label text
        cv2.putText(
            annotated_frame,
            label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    return annotated_frame


def video2jpg(video_path, output_folder, sample_freq=1):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        frame_index = 0
        save_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % sample_freq == 0:
                frame_filename = os.path.join(output_folder, f"{save_index:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                save_index += 1

            frame_index += 1

        cap.release()
        print(f"All frames have been saved to {output_folder}.")


color_list = {
    0: np.array([255, 0, 0]),       # Red
    1: np.array([0, 255, 0]),       # Green
    2: np.array([0, 0, 255]),       # Blue
    3: np.array([0, 125, 125]),     # Teal
    4: np.array([125, 0, 125]),     # Purple
    5: np.array([125, 125, 0]),     # Yellow
    6: np.array([255, 165, 0]),     # Orange
    7: np.array([255, 105, 180]),   # Pink
}


def contour_painter(
    input_image,
    input_mask,
    mask_color=5,
    mask_alpha=0.7,
    contour_color=1,
    contour_width=3,
    ann_obj_id=None,
):
    assert (
        input_image.shape[:2] == input_mask.shape
    ), "Different shape between image and mask"
    # 0: background, 1: foreground
    mask = np.clip(input_mask, 0, 1).astype(np.uint8)
    contour_radius = (contour_width - 1) // 2

    dist_transform_fore = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    dist_transform_back = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
    dist_map = dist_transform_fore - dist_transform_back
    contour_radius += 2
    contour_mask = np.abs(np.clip(dist_map, -contour_radius, contour_radius))
    contour_mask = contour_mask / np.max(contour_mask)
    contour_mask[contour_mask > 0.5] = 1.0

    # Paint contour
    painted_image = input_image.copy()
    color = color_list[contour_color]
    mask = 1 - contour_mask
    painted_image[mask.astype(bool)] = (
        painted_image[mask.astype(bool)] * (1 - 1) + color * 1
    ).astype("uint8")

    # Find the center position of the mask
    moments = cv2.moments(mask)
    if moments["m00"] != 0 and ann_obj_id is not None:
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_color = (0, 0, 0)
        font_thickness = 3
        cv2.putText(
            painted_image,
            str(ann_obj_id),
            (cX, cY),
            font,
            font_scale,
            font_color,
            font_thickness,
        )

    return painted_image


def write_video(frames, output_path, fps):
    if not frames:
        print("Error: No frames to write.")
        return

    height, width, _ = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    print("Video writing completed.")
    video_writer.release()


def process_mask_signal(mask_add, mask_min):
    kernel_size = 3

    n = len(mask_add)

    fig, axes = plt.subplots(n * 2, 1, figsize=(10, 5 * n * 2), sharex=True, sharey=True)
    axes = axes.flatten()

    filtered_mask_add = {}
    filtered_mask_min = {}
    min_num = 99999
    max_num = 0

    index = 0
    for k in mask_add.keys():
        mask1 = np.array(mask_add[k])
        mask2 = np.array(mask_min[k])

        # Median filter
        filtered_data1 = scipy.signal.medfilt(mask1, kernel_size=kernel_size)
        filtered_data2 = scipy.signal.medfilt(mask2, kernel_size=kernel_size)

        filtered_mask_add[k] = filtered_data1
        filtered_mask_min[k] = filtered_data2

        max_num = max(max_num, max(filtered_mask_add[k]))
        max_num = max(max_num, max(filtered_mask_min[k]))
        min_num = min(min_num, min(filtered_mask_add[k]))
        min_num = min(min_num, min(filtered_mask_min[k]))

        axes[index].plot(filtered_data1, linestyle="-", color="b")
        axes[index + 1].plot(filtered_data2, linestyle="-", color="b")
        index += 2

    plt.show()

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    fig, axes = plt.subplots(n * 2, 1, figsize=(10, 5 * n * 2), sharex=True, sharey=True)
    axes = axes.flatten()
    index = 0

    for k in filtered_mask_add.keys():
        filtered_mask_add[k] = ((filtered_mask_add[k] - min_num) / max_num) * 2 - 1
        filtered_mask_min[k] = ((filtered_mask_min[k] - min_num) / max_num) * 2 - 1

        filtered_mask_add[k] = sigmoid(filtered_mask_add[k] * 5)
        filtered_mask_min[k] = sigmoid(filtered_mask_min[k] * 5)

        axes[index].plot(filtered_mask_add[k], linestyle="-", color="b")
        axes[index + 1].plot(filtered_mask_min[k], linestyle="-", color="b")
        index += 2

    plt.show()

    fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    index = 0

    final_result = {}
    for k in filtered_mask_add.keys():
        final_result[k] = filtered_mask_add[k] * filtered_mask_min[k]

        axes[index].plot(final_result[k], linestyle="-", color="b")
        index += 1

    plt.show()

def convert_video_to_h264(
        input_path: str | Path,
        output_path: str | Path,
    ) -> None:
        """Convert a video to H.264 for broader playback compatibility."""

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(output_path),
        ]

        completed_process = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if not Path(output_path).is_file():
            raise RuntimeError(
                f"H.264 video was not created: {output_path}"
            )


def run_visual_prompting(
    input_video_path,
    output_video_path,
    artifacts_dir,
    key_frames,
    objects=None,
    grounding_config=None,
    grounding_checkpoint=None,
    bert_model=None,
    sam_checkpoint="sam_vit_h_4b8939.pth",
    sam2_checkpoint="segment-anything-2/checkpoints/sam2_hiera_large.pt",
):  

    key_frames = ast.literal_eval(key_frames)
    artifacts_dir = os.path.abspath(artifacts_dir)
    os.makedirs(artifacts_dir, exist_ok=True)

    if (grounding_config is None) != (grounding_checkpoint is None):
        raise ValueError(
            "--grounding_config and --grounding_checkpoint must be provided together"
        )

    # First Part: Get object list manually or from the VLM.
    if objects is not None:
        obj_list = parse_object_list(objects)
        num = len(obj_list)
        object_discovery_source = "manual"
        print(f"Using manually provided object list: {obj_list}")
    else:
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is required when --objects is not provided"
            )
        client = OpenAI()
        object_list_response = get_object_list(input_video_path, client)
        num, obj_list = extract_num_object(object_list_response)
        object_discovery_source = "openai"
        print(f"Generated prompt: {obj_list}")

    parsed_object_count = len(obj_list)

    print(
        "[visual_prompting] CUDA available after object discovery:",
        torch.cuda.is_available(),
    )
    print(
        "[visual_prompting] CUDA_VISIBLE_DEVICES after object discovery:",
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    print(
        "[visual_prompting] NVIDIA_VISIBLE_DEVICES after object discovery:",
        os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    )

    print(
        "Object discovery counts: "
        f"reported={num}, parsed={parsed_object_count}"
    )
    if num != parsed_object_count:
        print(
            "WARNING: Reported object count does not match the parsed object list: "
            f"reported={num}, parsed={parsed_object_count}"
        )

    # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if DEVICE.type == "cuda":
    #     torch.cuda.set_device(DEVICE)

    cuda_available = torch.cuda.is_available()

    if not cuda_available:
        raise RuntimeError(
            "CUDA is not available. Visual prompting would fall back to CPU, "
            "so the execution has been stopped."
        )

    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)

    print("[visual_prompting] Selected device:", DEVICE)
    print("[visual_prompting] GPU:", torch.cuda.get_device_name(DEVICE))

    if grounding_config is not None:
        groundingdino_model = load_groundingdino_model(
            grounding_config,
            grounding_checkpoint,
            device=str(DEVICE),
            text_encoder_path=bert_model,
        )
    else:
        groundingdino_model = load_model_hf(
            "ShilongLiu/GroundingDINO",
            "groundingdino_swinb_cogcoor.pth",
            "GroundingDINO_SwinB.cfg.py",
            device=str(DEVICE),
            text_encoder_path=bert_model,
        )

    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    # Stable Diffusion is not used by the current visual-prompting pipeline.
    # Keep the original loading code available for future inpainting work, but
    # do not download the model or allocate its memory during object tracking.
    # float_type = torch.float32 if DEVICE.type == "cpu" else torch.float16
    # pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-2-inpainting",
    #     torch_dtype=float_type,
    # )
    # if DEVICE.type != "cpu":
    #     pipe = pipe.to(DEVICE)

    video_path = input_video_path
    sample_freq = 16
    output_video_path = output_video_path

    frames = read_video(video_path)

    # Second Part: Use GroundedSAM2 to track the objects
    # Parameters for GroundingDINO
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25
    object_counts = Counter(obj_list)

    image_source, image = load_image_from_array(frames[0])

    best_boxes = []
    best_phrases = []
    best_logits = []
    best_detector_labels = []
    grounding_counts = {}
    width_filter_diagnostics = {}

    # Iterate over each object and select the box with highest confidence
    for obj, count in object_counts.items():

        with TIMING.measure(
            f"visual.grounding_dino.{obj}",
            cuda=True,
        ):

            boxes, logits, phrases = predict(
                model=groundingdino_model,
                image=image,
                caption=obj,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD,
                device=DEVICE,
            )

            # Remove implausibly large cube detections before selecting
            # the highest-confidence candidates.
            if "cube" in obj:
                box_areas = boxes[:, 2] * boxes[:, 3]
                keep_mask = box_areas < 0.1

                boxes = boxes[keep_mask]
                logits = logits[keep_mask]

                keep_values = keep_mask.detach().cpu().tolist()
                phrases = [
                    phrase
                    for phrase, keep in zip(phrases, keep_values)
                    if keep
                ]

        raw_detected_count = int(boxes.shape[0])

        boxes, logits, phrases, width_filter = (
            filter_oversized_storage_bin_detections(
                boxes, logits, phrases, detector_label=obj
            )
        )
        width_filter_diagnostics[obj] = width_filter
        detected_count = int(boxes.shape[0])
        selected_count = min(count, detected_count)
        grounding_counts[obj] = {
            "requested": count,
            "detected": raw_detected_count,
            "eligible_after_width_filter": detected_count,
            "width_filter_removed": width_filter["removed_count"],
            "selected": selected_count,
        }
        print(
            f"GroundingDINO count for {obj!r}: requested={count}, "
            f"detected={raw_detected_count}, "
            f"eligible_after_width_filter={detected_count}, "
            f"selected={selected_count}"
        )
        if width_filter["removed_count"]:
            print(
                "STORAGE_BIN_WIDTH_FILTER: "
                f"{json.dumps(width_filter, sort_keys=True)}"
            )

        for i in range(selected_count):
            best_boxes.append(boxes[i].unsqueeze(0))
            best_phrases.append(phrases[i])
            best_logits.append(logits[i])
            best_detector_labels.append(obj)

    selected_box_count = len(best_boxes)
    if best_boxes:
        best_boxes = torch.cat(best_boxes)
        best_logits = torch.stack(best_logits)

    annotated_frame = my_annotate(
        image_source=image_source,
        boxes=best_boxes,
        logits=best_logits,
        phrases=best_phrases,
    )
    annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB

    with TIMING.measure(
        "visual.sam",
        cuda=True,
    ):
        sam_predictor.set_image(image_source)
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(best_boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy, image_source.shape[:2]
        ).to(DEVICE)
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

    masks = masks.cpu()
    masks_np = masks.numpy()
    masks_before_filter = len(masks_np)

    h, w = masks_np[0][0].shape
    pixel_cnt = h * w

    indices_to_keep = np.ones(len(masks_np), dtype=bool)
    for i in range(len(masks_np)):
        if np.sum(masks_np[i][0]) > pixel_cnt * 0.3:
            indices_to_keep[i] = False
    masks_np = masks_np[indices_to_keep]
    filtered_detector_labels = [
        label
        for label, keep in zip(best_detector_labels, indices_to_keep)
        if keep
    ]
    masks_after_filter = len(masks_np)

    track_id_map = {}
    for track_id, (detector_label, mask) in enumerate(
        zip(filtered_detector_labels, masks_np)
    ):
        mask_indices = np.argwhere(mask[0] > 0)
        if len(mask_indices) == 0:
            center = None
        else:
            avg_y, avg_x = np.mean(mask_indices, axis=0)
            center = [int(avg_x), int(avg_y)]
        track_id_map[track_id] = {
            "detector_label": detector_label,
            "initial_center": center,
        }

    print(f"TRACK_ID_MAP: {json.dumps(track_id_map, sort_keys=True)}")

    count_diagnostics = {
        "source": object_discovery_source,
        "reported_objects": num,
        "parsed_objects": parsed_object_count,
        "requested_by_label": dict(object_counts),
        "grounding_by_label": grounding_counts,
        "width_filter_by_label": width_filter_diagnostics,
        "selected_boxes": selected_box_count,
        "masks_before_filter": masks_before_filter,
        "masks_after_filter": masks_after_filter,
    }

    del groundingdino_model
    del sam
    del sam_predictor
    # del pipe

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Enable TF32 on supported NVIDIA GPUs.
    if DEVICE.type == "cuda":
        if torch.cuda.get_device_properties(DEVICE).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    model_cfg = "sam2_hiera_l.yaml"

    # SAM2 is executed under BF16 autocast only.
    #
    # Do not manually call __enter__() on the autocast context:
    # the context must be closed before returning to the rest of
    # the SeeDo pipeline, otherwise subsequent CUDA models such as
    # GroundingDINO inherit BF16 and may fail in custom CUDA ops.
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        predictor = build_sam2_video_predictor(
            model_cfg,
            sam2_checkpoint,
            device=DEVICE,
        )

        try:
            predictor_devices = {
                str(parameter.device)
                for parameter in predictor.parameters()
            }

            print(
                "[visual_prompting] SAM2 predictor parameter devices:",
                sorted(predictor_devices),
            )

        except Exception as error:
            print(
                "[visual_prompting] Could not inspect "
                "SAM2 parameter devices:",
                error,
            )

        # ------------------------------------------------------------
        # First round: sampled-video propagation
        # ------------------------------------------------------------

        video_stem = os.path.splitext(
            os.path.basename(video_path)
        )[0]

        video_dir = os.path.join(
            artifacts_dir,
            f"sample_freq_{sample_freq}_{video_stem}",
        )

        if not os.path.exists(video_dir):

            with TIMING.measure(
                "io.video2jpg_sampled"
            ):
                video2jpg(
                    video_path,
                    video_dir,
                    sample_freq,
                )

        frame_names = [
            p
            for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1]
            in [
                ".jpg",
                ".jpeg",
                ".JPG",
                ".JPEG",
            ]
        ]

        frame_names.sort(
            key=lambda p: int(
                os.path.splitext(p)[0]
            )
        )

        inference_state = predictor.init_state(
            video_path=video_dir
        )

        predictor.reset_state(
            inference_state
        )

        prompts = {}

        ann_frame_idx = 0
        ann_obj_id = 1

        for i in range(len(masks_np)):
            _, out_obj_ids, out_mask_logits = (
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=i,
                    mask=masks_np[i][0],
                )
            )

        print(
            "[visual_prompting] Starting "
            "sampled-video SAM2 propagation"
        )

        video_segments = {}

        with TIMING.measure(
            "visual.sam2_sampled",
            cuda=True,
        ):
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in predictor.propagate_in_video(
                inference_state
            ):
                video_segments[out_frame_idx] = {
                    out_obj_id: (
                        out_mask_logits[i] > 0.0
                    )
                    .cpu()
                    .numpy()
                    for i, out_obj_id
                    in enumerate(out_obj_ids)
                }

        print(
            "[visual_prompting] Sampled-video "
            "SAM2 propagation completed"
        )

        # ------------------------------------------------------------
        # Second round: full-video propagation
        # ------------------------------------------------------------

        del inference_state
        del predictor

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        predictor = build_sam2_video_predictor(
            model_cfg,
            sam2_checkpoint,
            device=DEVICE,
        )

        try:
            predictor_devices = {
                str(parameter.device)
                for parameter in predictor.parameters()
            }

            print(
                "[visual_prompting] SAM2 predictor parameter devices:",
                sorted(predictor_devices),
            )

        except Exception as error:
            print(
                "[visual_prompting] Could not inspect "
                "SAM2 parameter devices:",
                error,
            )

        video_dir = os.path.join(
            artifacts_dir,
            video_stem,
        )

        if not os.path.exists(video_dir):

            with TIMING.measure(
                "io.video2jpg_full"
            ):
                video2jpg(
                    video_path,
                    video_dir,
                    1,
                )

        frame_names = [
            p
            for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1]
            in [
                ".jpg",
                ".jpeg",
                ".JPG",
                ".JPEG",
            ]
        ]

        frame_names.sort(
            key=lambda p: int(
                os.path.splitext(p)[0]
            )
        )

        inference_state = predictor.init_state(
            video_path=video_dir
        )

        predictor.reset_state(
            inference_state
        )

        prompts = {}

        ann_frame_idx = 0
        ann_obj_id = 1

        for frame_idx in range(
            0,
            len(frame_names),
            sample_freq,
        ):
            for k in video_segments[
                frame_idx // sample_freq
            ].keys():
                _, out_obj_ids, out_mask_logits = (
                    predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=k,
                        mask=video_segments[
                            frame_idx // sample_freq
                        ][k][0],
                    )
                )

        print(
            "[visual_prompting] Starting "
            "full-video SAM2 propagation"
        )

        video_segments = {}
        
        with TIMING.measure(
            "visual.sam2_full",
            cuda=True,
        ):
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in predictor.propagate_in_video(
                inference_state
            ):
                video_segments[out_frame_idx] = {
                    out_obj_id: (
                        out_mask_logits[i] > 0.0
                    )
                    .cpu()
                    .numpy()
                    for i, out_obj_id
                    in enumerate(out_obj_ids)
                }

        print(
            "[visual_prompting] Full-video "
            "SAM2 propagation completed"
        )

    tracked_counts = [len(segments) for segments in video_segments.values()]
    count_diagnostics["tracked_objects_min"] = min(tracked_counts, default=0)
    count_diagnostics["tracked_objects_max"] = max(tracked_counts, default=0)
    comparable_counts = [
        count_diagnostics["reported_objects"],
        count_diagnostics["parsed_objects"],
        count_diagnostics["selected_boxes"],
        count_diagnostics["masks_after_filter"],
        count_diagnostics["tracked_objects_min"],
        count_diagnostics["tracked_objects_max"],
    ]
    count_diagnostics["count_consistent"] = len(set(comparable_counts)) == 1
    print(f"COUNT_DIAGNOSTICS: {json.dumps(count_diagnostics, sort_keys=True)}")
    if not count_diagnostics["count_consistent"]:
        print(
            "WARNING: Object counts differ across discovery, detection, "
            "segmentation, or tracking; inspect COUNT_DIAGNOSTICS."
        )

    # Third Part: Select key frames and compute center coordinates of masks
    key_frame_coordinates = {}

    # Iterate through the key frames provided
    for frame_idx in key_frames:
        current_frame_coords = []

        # Check if the frame exists in the video segments (contains mask data)
        if frame_idx in video_segments:
            # For each object in the frame, retrieve the mask
            for obj_id, mask in video_segments[frame_idx].items():
                mask_data = mask[0]
                mask_indices = np.argwhere(mask_data > 0)  # Get non-zero pixel indices

                if len(mask_indices) > 0:
                    # Calculate the average x and y coordinates of the mask to get the center
                    avg_y, avg_x = np.mean(mask_indices, axis=0)
                    current_frame_coords.append(
                        f"Object {obj_id}: ({int(avg_x)}, {int(avg_y)})"
                    )
                else:
                    # Print a warning if the mask is empty
                    print(
                        f"Warning: Empty mask for object {obj_id} in frame {frame_idx}"
                    )

        # Store the coordinates for the current frame
        key_frame_coordinates[f"key_frame{frame_idx}"] = current_frame_coords

    # Initialize an empty string to store the result
    bbx_string = ""

    # Iterate through the key_frame_coordinates and generate the string
    for key_frame, coordinates in key_frame_coordinates.items():
        coordinates_str = "\n".join(coordinates)  # Join the coordinates into a single string
        bbx_string += f"{key_frame}\n{coordinates_str}\n\n"  # Append the key frame and coordinates

    # Print the final bounding box string
    print(f"Bounding box extraction completed. Result:\n{bbx_string}")
    # Fourth Part: Append all the painted frames into a video
    painted_frames = []
    for i in range(len(frame_names)):
        img = cv2.imread(os.path.join(video_dir, frame_names[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for k in video_segments[i].keys():
            img = contour_painter(
                img, video_segments[i][k][0], contour_color=1, ann_obj_id=k
            )
        painted_frames.append(img)
    

    mask_add = {}
    mask_min = {}
    for k in video_segments[i].keys():
        mask_add[k] = []
        mask_min[k] = []

    with TIMING.measure(
        "io.write_annotated_video"
    ):
        write_video(painted_frames, output_video_path, fps=30)

    output_video_path = Path(output_video_path)

    h264_output_path = output_video_path.with_name(
        f"{output_video_path.stem}-h264.mp4"
    )

    convert_video_to_h264(
        input_path=output_video_path,
        output_path=h264_output_path,
    )

    for i in range(len(frame_names) - 1):
        for k in video_segments[i].keys():
            mask_before = video_segments[i][k][0].copy()
            mask_after = video_segments[i + 1][k][0].copy()
            mask_after[mask_before] = False
            add_cnt = np.sum(mask_after)

            mask_before = video_segments[i][k][0].copy()
            mask_after = video_segments[i + 1][k][0].copy()
            mask_before[mask_after] = False
            min_cnt = np.sum(mask_before)

            mask_add[k].append(add_cnt.item())
            mask_min[k].append(min_cnt.item())
    # process_mask_signal(mask_add, mask_min)
    # Return the final bounding box string
    print(bbx_string)
    # return bbx_string

    return VisualPromptingResult(
        annotated_video_path=h264_output_path,
        track_id_map=track_id_map,
        key_frame_coordinates=key_frame_coordinates,
        bounding_box_summary=bbx_string,
        count_diagnostics=count_diagnostics,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process video with SAM and GroundingDINO."
    )
    parser.add_argument("--input", type=str, help="Path to the input video")
    parser.add_argument("--output", type=str, help="Path to the output video")
    parser.add_argument("--key_frames", type=str, help="List of key frame indices as a string")
    parser.add_argument(
        "--objects",
        type=str,
        help="Comma-separated object names; skips OpenAI object discovery when set",
    )

    parser.add_argument(
        "--grounding_config",
        type=str,
        help="Local GroundingDINO config; requires --grounding_checkpoint",
    )
    parser.add_argument(
        "--grounding_checkpoint",
        type=str,
        help="Local GroundingDINO checkpoint; requires --grounding_config",
    )
    parser.add_argument(
        "--bert_model",
        type=str,
        help="Local BERT model directory used by GroundingDINO",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="sam_vit_h_4b8939.pth",
        help="Path to the SAM ViT-H checkpoint",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="segment-anything-2/checkpoints/sam2_hiera_large.pt",
        help="Path to the SAM2 Hiera Large checkpoint",
    )

    args = parser.parse_args()

    run_visual_prompting(
        input_video_path=args.input,
        output_video_path=args.output,
        artifacts_dir=os.path.dirname(os.path.abspath(args.output)),
        key_frames=args.key_frames,
        objects=args.objects,
        grounding_config=args.grounding_config,
        grounding_checkpoint=args.grounding_checkpoint,
        bert_model=args.bert_model,
        sam_checkpoint=args.sam_checkpoint,
        sam2_checkpoint=args.sam2_checkpoint,
    )