"""Observation-only open-vocabulary detector/segmenter trials (method C).

The heavy models are loaded lazily so the deterministic benchmark remains
usable without torch.  Every 3-D pose is reconstructed from submitted RGB-D
and camera calibration; model outputs are only 2-D boxes or masks.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from .camera_geometry import canonical_axis_yaw, canonical_square_yaw
from .contract import assert_policy_source_clean, validate_observation_manifest
from .policies import (
    BIN_SIZE,
    CUBE_SIZE,
    TRUSTED_CAMERA_TO_WORLD,
    CameraObservation,
    SceneObservation,
    _deduplicate,
    _pixel_world_maps,
    _rect_from_xy,
    load_scene_observation,
)


PROMPTS = (
    "red cube",
    "orange cube",
    "blue cube",
    "red storage bin",
    "orange storage bin",
    "blue storage bin",
    "toy flatbed truck",
)


def _identity(label: str) -> tuple[str, str]:
    text = label.lower().strip()
    if "truck" in text:
        return "truck", "truck"
    category = "cube" if "cube" in text else "bin"
    colour = next((value for value in ("red", "orange", "blue") if value in text), "unknown")
    return category, colour


def _trusted_observation(observation: SceneObservation) -> SceneObservation:
    cameras = {}
    for name, item in observation.cameras.items():
        camera = item.camera
        if name in TRUSTED_CAMERA_TO_WORLD:
            camera = type(camera)(
                width=camera.width,
                height=camera.height,
                focal_length_mm=camera.focal_length_mm,
                horizontal_aperture_mm=camera.horizontal_aperture_mm,
                vertical_aperture_mm=camera.vertical_aperture_mm,
                camera_to_world=TRUSTED_CAMERA_TO_WORLD[name].copy(),
                table_z=camera.table_z,
            )
        cameras[name] = CameraObservation(name, item.rgb_bgr, item.depth, camera)
    return SceneObservation(
        scene_id=observation.scene_id,
        cameras=cameras,
        robot_state=observation.robot_state,
        variation=observation.variation,
    )


class YoloWorldProvider:
    name = "yolo_world"

    def __init__(self, weights: str, confidence: float):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.model.set_classes(list(PROMPTS))
        self.confidence = confidence

    def predict(self, image_bgr: np.ndarray) -> list[dict]:
        result = self.model.predict(
            source=image_bgr,
            imgsz=1280,
            conf=self.confidence,
            iou=0.45,
            verbose=False,
        )[0]
        names = result.names
        proposals = []
        if result.boxes is None:
            return proposals
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, index in zip(boxes, scores, classes):
            proposals.append(
                {"label": str(names[index]), "score": float(score), "box": [float(v) for v in box]}
            )
        return proposals


class GroundingDinoProvider:
    name = "grounding_dino"

    def __init__(self, model_id: str, confidence: float):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.confidence = confidence

    def predict(self, image_bgr: np.ndarray) -> list[dict]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        text = ". ".join(PROMPTS) + "."
        inputs = self.processor(images=rgb, text=text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.confidence,
            text_threshold=0.10,
            target_sizes=[rgb.shape[:2]],
        )[0]
        proposals = []
        labels = result.get("text_labels", result["labels"])
        for box, score, label in zip(result["boxes"], result["scores"], labels):
            proposals.append(
                {
                    "label": str(label),
                    "score": float(score.detach().cpu()),
                    "box": [float(value) for value in box.detach().cpu().tolist()],
                }
            )
        return proposals


class UltralyticsPromptSegmenter:
    """FastSAM or MobileSAM mask refinement for detector boxes."""

    def __init__(self, kind: str, weights: str):
        if kind == "fastsam":
            from ultralytics import FastSAM

            self.model = FastSAM(weights)
        elif kind == "mobilesam":
            from ultralytics import SAM

            self.model = SAM(weights)
        else:
            raise ValueError(kind)
        self.kind = kind

    def refine(self, image_bgr: np.ndarray, proposals: list[dict]) -> list[dict]:
        if not proposals:
            return proposals
        boxes = [proposal["box"] for proposal in proposals]
        kwargs = {"source": image_bgr, "bboxes": boxes, "verbose": False}
        if self.kind == "fastsam":
            kwargs.update({"imgsz": 1024, "conf": 0.08, "iou": 0.9, "retina_masks": True})
        result = self.model.predict(**kwargs)[0]
        if result.masks is None:
            return proposals
        masks = result.masks.data.detach().cpu().numpy()
        height, width = image_bgr.shape[:2]
        refined = []
        for proposal, mask in zip(proposals, masks):
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
            item = dict(proposal)
            item["mask"] = mask > 0.5
            refined.append(item)
        return refined


def _proposal_mask(shape: tuple[int, int], proposal: dict) -> np.ndarray:
    if "mask" in proposal:
        return np.asarray(proposal["mask"], dtype=bool)
    height, width = shape
    x0, y0, x1, y1 = proposal["box"]
    x0, x1 = sorted((max(0, int(math.floor(x0))), min(width, int(math.ceil(x1)))))
    y0, y1 = sorted((max(0, int(math.floor(y0))), min(height, int(math.ceil(y1)))))
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    # Erode model boundaries to reject mixed object/table depth pixels.
    if min(x1 - x0, y1 - y0) >= 8:
        mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return mask


def _pose_from_proposal(camera_obs: CameraObservation, proposal: dict) -> dict | None:
    if camera_obs.depth is None:
        return None
    category, colour = _identity(proposal["label"])
    if category != "truck" and colour == "unknown":
        return None
    world, valid = _pixel_world_maps(camera_obs.depth, camera_obs.camera, stride=1)
    mask = _proposal_mask(valid.shape, proposal)
    use = mask & valid
    z = world[..., 2]
    minimum = 0.004 if category == "cube" else 0.012
    use &= z > camera_obs.camera.table_z + minimum
    use &= z < camera_obs.camera.table_z + 0.145
    use &= world[..., 0] >= 0.16
    use &= world[..., 0] <= 0.69
    use &= world[..., 1] >= -0.38
    use &= world[..., 1] <= 0.30
    points = world[use]
    if len(points) < (8 if category == "cube" else 20):
        return None
    centre, extents, yaw = _rect_from_xy(points[:, :2])
    raw_centre = centre.copy()
    if category == "cube":
        expected = CUBE_SIZE
        pose_yaw = canonical_square_yaw(yaw)
    elif category == "bin":
        expected = BIN_SIZE
        top = points[:, 2] >= max(camera_obs.camera.table_z + 0.035, np.percentile(points[:, 2], 55))
        if int(top.sum()) >= 12:
            centre, extents, yaw = _rect_from_xy(points[top, :2])
        pose_yaw = canonical_square_yaw(yaw)
    else:
        expected = 0.18
        pose_yaw = canonical_axis_yaw(yaw)
        # Estimate the bed only from the measured-height horizontal band.
        deck = np.abs(points[:, 2] - (camera_obs.camera.table_z + 0.0335)) < 0.005
        if int(deck.sum()) >= 20:
            centre, _, _ = _rect_from_xy(points[deck, :2])
        colour = "truck"
    long_side, short_side = sorted((float(extents[0]), float(extents[1])), reverse=True)
    size = math.sqrt(max(long_side * short_side, 1e-12))
    confidence = float(proposal["score"] * math.exp(-0.5 * abs(size - expected) / expected))
    item = {
        "category": category,
        "colour": colour,
        "xy": [float(centre[0]), float(centre[1])],
        "yaw_deg": float(pose_yaw),
        "confidence": confidence,
        "extents_m": [long_side, short_side],
        "source": f"{camera_obs.name}_{proposal.get('backend', 'open_vocab')}_mask_depth",
        "proposal_score": float(proposal["score"]),
        "raw_centre_xy": [float(raw_centre[0]), float(raw_centre[1])],
    }
    if category == "truck":
        item["deck_centre_xy"] = item["xy"]
        item["deck_yaw_deg"] = item["yaw_deg"]
    return item


def _fuse(items: list[dict]) -> list[dict]:
    result = []
    for identity in sorted({(item["category"], item["colour"]) for item in items}):
        group = [item for item in items if (item["category"], item["colour"]) == identity]
        radius = 0.014 if identity[0] == "cube" else 0.035
        result.extend(_deduplicate(group, radius=radius))
    # Bound inventory so an open-vocabulary proposal storm cannot game counts.
    bounded = []
    for category, colour in (("cube", c) for c in ("red", "orange", "blue")):
        group = [item for item in result if item["category"] == category and item["colour"] == colour]
        bounded.extend(sorted(group, key=lambda value: value["confidence"], reverse=True)[:4])
    for category, colour in [("bin", c) for c in ("red", "orange", "blue")] + [("truck", "truck")]:
        group = [item for item in result if item["category"] == category and item["colour"] == colour]
        bounded.extend(sorted(group, key=lambda value: value["confidence"], reverse=True)[:1])
    return bounded


def run_scene(
    provider,
    observation: SceneObservation,
    segmenter=None,
    camera_names: tuple[str, ...] = ("table", "wrist_scan_near", "wrist_scan_far", "wrist_scan_oblique"),
) -> dict:
    started = time.perf_counter()
    corrected = _trusted_observation(observation)
    detections = []
    proposal_count = 0
    for name in camera_names:
        camera = corrected.cameras.get(name)
        if camera is None:
            continue
        proposals = provider.predict(camera.rgb_bgr)
        proposal_count += len(proposals)
        for proposal in proposals:
            proposal["backend"] = provider.name
        if segmenter is not None:
            proposals = segmenter.refine(camera.rgb_bgr, proposals)
        for proposal in proposals:
            item = _pose_from_proposal(camera, proposal)
            if item is not None:
                detections.append(item)
    return {
        "scene_id": observation.scene_id,
        "detections": _fuse(detections),
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "notes": [
            f"{provider.name} 2-D proposals; submitted RGB-D back-projection only",
            f"proposal_count={proposal_count}",
            f"segmenter={getattr(segmenter, 'kind', 'none')}",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", choices=("yolo_world", "grounding_dino"), required=True)
    parser.add_argument("--segmenter", choices=("none", "fastsam", "mobilesam"), default="none")
    parser.add_argument("--weights", default="")
    parser.add_argument("--confidence", type=float, default=0.08)
    parser.add_argument("--max-scenes", type=int, default=0)
    args = parser.parse_args()

    assert_policy_source_clean(Path(__file__))
    if args.backend == "yolo_world":
        provider = YoloWorldProvider(args.weights or "yolov8s-worldv2.pt", args.confidence)
    else:
        provider = GroundingDinoProvider(args.weights or "IDEA-Research/grounding-dino-tiny", args.confidence)
    segmenter = None
    if args.segmenter != "none":
        default = "FastSAM-s.pt" if args.segmenter == "fastsam" else "mobile_sam.pt"
        segmenter = UltralyticsPromptSegmenter(args.segmenter, default)

    observation_root = Path(args.observations)
    scenes = sorted(path.parent for path in observation_root.glob("scene_*/observation.json"))
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]
    policy_name = f"C_{args.backend}_{args.segmenter}_mask_depth"
    output = Path(args.out) / "predictions" / policy_name
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    for index, scene_dir in enumerate(scenes):
        try:
            validate_observation_manifest(scene_dir / "observation.json")
            prediction = run_scene(provider, load_scene_observation(scene_dir), segmenter)
            prediction["policy"] = policy_name
            (output / f"{scene_dir.name}.json").write_text(json.dumps(prediction, indent=2), encoding="utf-8")
            print(f"[neural] {index + 1}/{len(scenes)} {scene_dir.name}", flush=True)
        except Exception as exc:
            errors.append({"scene": scene_dir.name, "error": repr(exc)})
    summary = {"policy": policy_name, "scenes": len(scenes), "errors": errors}
    (Path(args.out) / f"{policy_name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
