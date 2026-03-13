#!/usr/bin/env python3
"""Run inference against a trained classifier on a single image."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import face_recognition
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a person label from one image.")
    parser.add_argument("--model", required=True, type=Path, help="Path to face_classifier.pkl")
    parser.add_argument("--image", required=True, type=Path, help="Image to classify")
    parser.add_argument("--upsample", type=int, default=1, help="Face detection upsample count.")
    parser.add_argument("--detection-model", choices=("hog", "cnn"), default="hog", help="Face detector backend.")
    parser.add_argument(
        "--landmark-model",
        choices=("small", "large"),
        default="small",
        help="Landmark model used by face_recognition.face_encodings.",
    )
    parser.add_argument("--distance-threshold", type=float, help="Override the threshold stored in the model artifact.")
    return parser.parse_args()


def predict_with_knn(classifier: Any, encoding: np.ndarray, distance_threshold: float) -> dict[str, Any]:
    distances, _indices = classifier.kneighbors([encoding], n_neighbors=1)
    distance = float(distances[0][0])
    label = str(classifier.predict([encoding])[0])
    accepted = distance <= distance_threshold
    return {
        "label": label if accepted else "unknown",
        "rawLabel": label,
        "distance": round(distance, 4),
        "accepted": accepted,
        "threshold": distance_threshold,
    }


def predict_with_svm(classifier: Any, encoding: np.ndarray) -> dict[str, Any]:
    label = str(classifier.predict([encoding])[0])
    confidence = None
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba([encoding])[0]
        confidence = float(np.max(probabilities))
    return {
        "label": label,
        "rawLabel": label,
        "accepted": True,
        "confidence": round(confidence, 4) if confidence is not None else None,
    }


def main() -> int:
    args = parse_args()

    with args.model.resolve().open("rb") as handle:
        artifact = pickle.load(handle)

    classifier = artifact["classifier"]
    model_type = artifact["modelType"]
    distance_threshold = args.distance_threshold or float(artifact.get("distanceThreshold", 0.55))

    image = face_recognition.load_image_file(str(args.image.resolve()))
    locations = face_recognition.face_locations(
        image,
        number_of_times_to_upsample=args.upsample,
        model=args.detection_model,
    )
    if len(locations) != 1:
        print(
            json.dumps(
                {
                    "image": str(args.image.resolve()),
                    "facesDetected": len(locations),
                    "error": "Expected exactly one face in the image.",
                },
                indent=2,
            )
        )
        return 1

    encodings = face_recognition.face_encodings(
        image,
        known_face_locations=locations,
        model=args.landmark_model,
    )
    if not encodings:
        print(
            json.dumps(
                {
                    "image": str(args.image.resolve()),
                    "facesDetected": 1,
                    "error": "Face encoding could not be generated.",
                },
                indent=2,
            )
        )
        return 1

    encoding = np.asarray(encodings[0], dtype=np.float32)
    if model_type == "knn":
        prediction = predict_with_knn(classifier, encoding, distance_threshold)
    else:
        prediction = predict_with_svm(classifier, encoding)

    print(
        json.dumps(
            {
                "image": str(args.image.resolve()),
                "facesDetected": 1,
                "modelType": model_type,
                "prediction": prediction,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
