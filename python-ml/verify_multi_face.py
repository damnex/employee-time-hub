#!/usr/bin/env python3
"""Verify multiple faces in one image against the exported employee profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import face_recognition
import numpy as np

FACE_MATCH_THRESHOLD = 0.87
FACE_PRIMARY_THRESHOLD = 0.88
FACE_ANCHOR_AVG_THRESHOLD = 0.85
FACE_ANCHOR_RATIO_THRESHOLD = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect all faces in one image and verify each face against "
            "the exported employee profiles."
        )
    )
    parser.add_argument("--profiles", required=True, type=Path, help="Path to output/profiles.json")
    parser.add_argument("--image", required=True, type=Path, help="Image that may contain many employee faces.")
    parser.add_argument("--top-k", type=int, default=3, help="How many best matches to include per detected face.")
    parser.add_argument("--max-faces", type=int, default=60, help="Hard cap for accepted face detections per frame.")
    parser.add_argument("--upsample", type=int, default=1, help="Face detection upsample count.")
    parser.add_argument("--detection-model", choices=("hog", "cnn"), default="hog", help="Face detector backend.")
    parser.add_argument(
        "--landmark-model",
        choices=("small", "large"),
        default="small",
        help="Landmark model used by face_recognition.face_encodings.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=1600,
        help="Resize wide images before detection to improve throughput. Use 0 to disable resizing.",
    )
    parser.add_argument("--match-threshold", type=float, default=FACE_MATCH_THRESHOLD, help="Minimum final confidence.")
    parser.add_argument("--primary-threshold", type=float, default=FACE_PRIMARY_THRESHOLD, help="Minimum primary confidence.")
    parser.add_argument(
        "--anchor-threshold",
        type=float,
        default=FACE_ANCHOR_AVG_THRESHOLD,
        help="Minimum anchor average confidence.",
    )
    parser.add_argument(
        "--anchor-ratio-threshold",
        type=float,
        default=FACE_ANCHOR_RATIO_THRESHOLD,
        help="Minimum strong-anchor ratio.",
    )
    return parser.parse_args()


def round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def calculate_legacy_match_confidence(v1: np.ndarray, v2: np.ndarray) -> float:
    if v1.size == 0 or v1.shape != v2.shape:
        return 0.0

    centered_v1 = v1 - np.mean(v1)
    centered_v2 = v2 - np.mean(v2)
    magnitude_v1 = float(np.linalg.norm(centered_v1))
    magnitude_v2 = float(np.linalg.norm(centered_v2))
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0

    score = float(np.dot(centered_v1 / magnitude_v1, centered_v2 / magnitude_v2))
    return round_float((score + 1.0) / 2.0)


def load_profiles(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("profiles.json must contain a list")
    return payload


def build_match_metrics(live_descriptor: np.ndarray, stored_profile: dict[str, Any]) -> dict[str, float]:
    primary_descriptor = np.asarray(stored_profile["faceDescriptor"]["primaryDescriptor"], dtype=np.float32)
    anchor_descriptors = [
        np.asarray(descriptor, dtype=np.float32)
        for descriptor in stored_profile["faceDescriptor"]["anchorDescriptors"]
    ]

    primary_confidence = max(
        calculate_legacy_match_confidence(live_descriptor, primary_descriptor),
        *(calculate_legacy_match_confidence(live_descriptor, anchor) for anchor in anchor_descriptors),
    )
    anchor_scores = [
        max(
            calculate_legacy_match_confidence(live_descriptor, primary_descriptor),
            *(calculate_legacy_match_confidence(live_descriptor, anchor) for anchor in anchor_descriptors),
        )
    ]
    anchor_average = average(anchor_scores)
    peak_anchor_confidence = max(anchor_scores, default=0.0)
    strong_anchor_ratio = len(
        [score for score in anchor_scores if score >= FACE_ANCHOR_AVG_THRESHOLD]
    ) / max(1, len(anchor_scores))
    final_confidence = round_float(
        primary_confidence * 0.45
        + anchor_average * 0.45
        + strong_anchor_ratio * 0.1
    )

    return {
        "primaryConfidence": round_float(primary_confidence),
        "anchorAverage": round_float(anchor_average),
        "peakAnchorConfidence": round_float(peak_anchor_confidence),
        "strongAnchorRatio": round_float(strong_anchor_ratio),
        "finalConfidence": final_confidence,
    }


def scale_locations(
    locations: list[tuple[int, int, int, int]],
    scale_factor: float,
) -> list[tuple[int, int, int, int]]:
    if scale_factor == 1.0:
        return locations

    scaled: list[tuple[int, int, int, int]] = []
    for top, right, bottom, left in locations:
        scaled.append(
            (
                int(round(top / scale_factor)),
                int(round(right / scale_factor)),
                int(round(bottom / scale_factor)),
                int(round(left / scale_factor)),
            )
        )
    return scaled


def maybe_resize_image(image: np.ndarray, resize_width: int) -> tuple[np.ndarray, float]:
    if resize_width <= 0:
        return image, 1.0

    height, width = image.shape[:2]
    if width <= resize_width:
        return image, 1.0

    scale_factor = resize_width / float(width)
    resized_height = max(1, int(round(height * scale_factor)))
    indices_y = np.linspace(0, height - 1, resized_height).astype(np.int32)
    indices_x = np.linspace(0, width - 1, resize_width).astype(np.int32)
    resized = image[np.ix_(indices_y, indices_x)]
    return resized, scale_factor


def verify_faces(
    image_path: Path,
    profiles: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    original_image = face_recognition.load_image_file(str(image_path.resolve()))
    working_image, scale_factor = maybe_resize_image(original_image, args.resize_width)

    locations = face_recognition.face_locations(
        working_image,
        number_of_times_to_upsample=args.upsample,
        model=args.detection_model,
    )
    if len(locations) > args.max_faces:
        locations = locations[: args.max_faces]

    encodings = face_recognition.face_encodings(
        working_image,
        known_face_locations=locations,
        model=args.landmark_model,
    )
    full_size_locations = scale_locations(locations, scale_factor)

    results: list[dict[str, Any]] = []
    for index, encoding in enumerate(encodings):
        descriptor = np.asarray(encoding, dtype=np.float32)
        ranked_matches = []
        for profile in profiles:
            metrics = build_match_metrics(descriptor, profile)
            ranked_matches.append(
                {
                    "folderName": profile["folderName"],
                    "displayName": profile.get("displayName") or profile["folderName"],
                    "sampleCount": profile.get("sampleCount"),
                    "metrics": metrics,
                }
            )

        ranked_matches.sort(
            key=lambda item: item["metrics"]["finalConfidence"],
            reverse=True,
        )
        top_matches = ranked_matches[: max(1, args.top_k)]
        best_match = top_matches[0]
        best_metrics = best_match["metrics"]
        verified = (
            best_metrics["finalConfidence"] >= args.match_threshold
            and best_metrics["primaryConfidence"] >= args.primary_threshold
            and best_metrics["anchorAverage"] >= args.anchor_threshold
            and best_metrics["strongAnchorRatio"] >= args.anchor_ratio_threshold
        )
        top, right, bottom, left = full_size_locations[index]

        results.append(
            {
                "faceIndex": index,
                "box": {
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "left": left,
                },
                "verified": verified,
                "bestMatch": best_match if verified else None,
                "topMatches": top_matches,
            }
        )

    return {
        "image": str(image_path.resolve()),
        "rosterSize": len(profiles),
        "facesDetected": len(locations),
        "facesEncoded": len(encodings),
        "results": results,
    }


def main() -> int:
    args = parse_args()
    profiles = load_profiles(args.profiles)
    result = verify_faces(args.image, profiles, args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
