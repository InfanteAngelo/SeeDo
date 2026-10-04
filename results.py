from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ai_controller.models.seedo_controller.task_types import (
    TaskType,
)


@dataclass(frozen=True)
class FrameExtractorResult:
    """Structured output produced by the original SeeDo FrameExtractor.

    Notes:
        keyframe_images are NumPy arrays in RGB channel order.
    """

    # Frame indices selected as keyframes from the demonstration video.
    keyframes: tuple[int, ...]

    # RGB images corresponding to the selected keyframes.
    keyframe_images: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class VisualPromptingResult:
    """Structured output produced by the SeeDo visual-prompting stage."""

    # Path to the annotated video containing tracked/segmented objects.
    annotated_video_path: Path

    # Mapping from persistent track IDs to detector/tracking metadata.
    track_id_map: dict[int, dict[str, object]]

    # Object coordinates extracted for each selected keyframe.
    key_frame_coordinates: dict[str, list[str]]

    # Human-readable summary of the extracted object bounding boxes.
    bounding_box_summary: str

    # Diagnostics describing object-count consistency across the
    # visual-prompting pipeline.
    count_diagnostics: dict[str, object]


@dataclass(frozen=True)
class ActionStep:
    """Single high-level manipulation step extracted from the demonstration."""

    # Demonstration keyframe in which the object is selected/picked.
    pick_keyframe: int

    # Demonstration keyframe representing the placement phase.
    place_keyframe: int

    # Persistent track ID of the object manipulated by the demonstrator.
    picked_track_id: int

    # Semantic category of the picked object.
    picked_category: str

    # Color attribute associated with the picked object.
    picked_color: str

    # Persistent track ID of the destination object/container.
    destination_track_id: int

    # Semantic category of the destination.
    destination_category: str

    # Left-to-right ordinal position of the destination among objects
    # belonging to the same category.
    destination_ordinal_from_left: int | None

    # Spatial or semantic relation connecting the picked object and
    # destination.
    relation: str

    # Natural-language description of the manipulation action.
    action: str

    # Exact detector label of the picked object.
    # Empty only for legacy action plans.
    picked_detector_label: str = ""


@dataclass(frozen=True)
class ActionPlanningResult:
    """Structured output produced by the demonstration action planner."""

    # Ordered manipulation steps inferred from the demonstration.
    steps: tuple[ActionStep, ...]

    # Planner execution/result status.
    status: str

    # Ambiguities detected while interpreting the demonstration.
    ambiguities: tuple[str, ...]

    # Human-readable description of the complete inferred task.
    natural_language_plan: str

    # Manipulation task inferred from the demonstration.
    # Supported values are "pick_and_place" and "nut_assembly".
    # "unknown" is used when the task cannot be classified reliably.
    task_type: TaskType | None = None

@dataclass(frozen=True)
class StructuredSceneObject:
    """Object used to represent the spatial structure of a scene."""

    # Identifier of the source object.
    # Demonstration track IDs are converted to strings so that the same
    # representation can also be used for runtime object identifiers.
    object_id: str

    # Semantic category of the object.
    category: str

    # SAM-mask centroid expressed in image pixel coordinates.
    center: tuple[float, float]


@dataclass(frozen=True)
class StructuredSceneRelation:
    """Directed qualitative spatial relation between two scene objects."""

    # Object whose position is being described.
    subject_object_id: str

    # Object used as spatial reference.
    reference_object_id: str

    # Qualitative relation of the subject relative to the reference.
    # Examples: LEFT, RIGHT, UP, DOWN, UP_LEFT, ...
    relation: str


@dataclass(frozen=True)
class StructuredScene:
    """Canonical structural representation of a scene."""

    # Objects participating in the structural representation.
    objects: tuple[StructuredSceneObject, ...]

    # Directed spatial relations between scene objects.
    relations: tuple[StructuredSceneRelation, ...]

    # Number of angular directions used to quantize spatial relations.
    # Supported values are 4 and 8.
    directions: int

@dataclass(frozen=True)
class StructuralObjectMatch:
    """Association between one demonstration object and one runtime object."""

    demo_object_id: str
    runtime_object_id: str


@dataclass(frozen=True)
class StructuralMapping:
    """One complete structure-preserving bijection between two scenes."""

    matches: tuple[StructuralObjectMatch, ...]


@dataclass(frozen=True)
class StructuralMatchingResult:
    """Result of structural matching between demonstration and runtime scenes."""

    # All bijections that preserve the complete qualitative spatial structure.
    valid_mappings: tuple[StructuralMapping, ...]

    @property
    def is_valid(self) -> bool:
        """Return whether at least one structure-preserving mapping exists."""
        return bool(self.valid_mappings)

    @property
    def is_unique(self) -> bool:
        """Return whether exactly one structure-preserving mapping exists."""
        return len(self.valid_mappings) == 1

@dataclass(frozen=True)
class ResolvedActionTargets:
    """Runtime targets resolved for one demonstrated manipulation step."""

    # Index of the corresponding ActionStep in the ActionPlanningResult.
    action_step_index: int

    # Runtime SceneObject corresponding to the demonstrated picked object.
    runtime_pick_object_id: str

    # Runtime SceneObject corresponding to the demonstrated place destination.
    runtime_place_object_id: str


@dataclass(frozen=True)
class ReplicabilityResult:
    """Result of checking whether the demonstrated task can be reproduced."""

    # True only when every demonstrated action can be resolved and executed
    # according to the supported task semantics.
    replicable: bool

    # Runtime targets resolved for every demonstrated action step.
    # Empty when the task is not replicable.
    resolved_targets: tuple[ResolvedActionTargets, ...]

    # Human-readable explanations when replicability fails.
    failure_reasons: tuple[str, ...]

@dataclass(frozen=True)
class SceneObject:
    """Object detected in the robot workspace at the initial perception step."""

    # Stable identifier used to reference the object in generated plans.
    object_id: str

    # Semantic object label used by the planner.
    label: str

    # Object center expressed in RGB-image pixel coordinates.
    pixel_coordinates: tuple[int, int]

    # Object 3D position expressed in the camera reference frame.
    position_camera: tuple[float, float, float]

    # Object 3D position transformed into the robot base frame.
    position_base: tuple[float, float, float]

    # Optional semantic category assigned to the object.
    category: str | None = None

    # Optional attribute assigned to the object.
    attributes: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class SceneState:
    """Frozen snapshot of the robot workspace used by the baseline."""

    # Semantically interpreted objects available to CAP/LMP planning.
    objects: tuple[SceneObject, ...]


@dataclass(frozen=True)
class PrimitiveStep:
    """Single robot primitive generated by CAP."""

    # Name of the primitive to execute, e.g. reach, pick, or placing.
    name: str

    # Arguments supplied to the primitive, including object/destination IDs.
    arguments: dict[str, Any]

    # Optional source-code fragment from which the primitive was extracted.
    source_code: str | None = None


@dataclass(frozen=True)
class PrimitivePlan:
    """Ordered primitive sequence generated once from the SeeDo action plan."""

    # Ordered sequence of symbolic primitives to be executed.
    steps: tuple[PrimitiveStep, ...]

    # Complete CAP-generated source code corresponding to the plan.
    source_code: str


@dataclass(frozen=True)
class DetectedObject:
    """Output of the 2D detection/segmentation stage."""

    # Internal identifier assigned to the detected object.
    object_id: str

    # Detector semantic label.
    label: str

    # Representative object position in image pixel coordinates.
    pixel_coordinates: tuple[int, int]

    # Binary/boolean segmentation mask associated with the object.
    mask: np.ndarray

    # Optional detector confidence score.
    confidence: float | None = None

    # Optional semantic category assigned to the object.
    category: str | None = None

    # Optional attribute assigned to the object.
    attributes: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ScenePerceptionResult:
    """Complete output of the runtime geometric-perception stage."""

    # Raw geometric description of all perceived objects.
    raw_scene: RawSceneState

    # Optional path to the visualization produced by the perception stage.
    overlay_image_path: Path | None = None

    # Optional path to the serialized raw geometric scene.
    raw_scene_json_path: Path | None = None


@dataclass(frozen=True)
class RawSceneObject:
    """Pure geometric perception result.

    This object contains only information produced by the perception
    pipeline (GroundingDINO + SAM + depth + geometry). No task-specific
    semantic interpretation is attached.
    """

    # Stable identifier assigned during runtime perception.
    object_id: str

    # Raw detector label before task-specific semantic interpretation.
    label: str

    # Representative object position in RGB-image pixel coordinates.
    pixel_coordinates: tuple[int, int]

    # Geometric object position expressed in the camera frame.
    position_camera: tuple[float, float, float]

    # Geometric object position transformed into the robot base frame.
    position_base: tuple[float, float, float]

    # Optional segmentation mask associated with the object.
    mask: np.ndarray | None = None

    # Optional detector confidence score.
    confidence: float | None = None

    # Optional semantic category assigned to the object.
    category: str | None = None

    # Optional attribute assigned to the object.
    attributes: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class RawSceneState:
    """Scene before semantic interpretation."""

    # Geometric objects detected in the initial runtime scene.
    objects: tuple[RawSceneObject, ...]

    def get_objects(
        self,
        label: str | None = None,
    ) -> tuple[RawSceneObject, ...]:
        """Return all objects or only the objects matching a raw label."""

        if label is None:
            return self.objects

        return tuple(
            obj
            for obj in self.objects
            if obj.label == label
        )