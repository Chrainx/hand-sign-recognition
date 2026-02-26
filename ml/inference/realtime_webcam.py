from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from ml.detection.mediapipe_detector import MediaPipeHandDetector
from ml.features.landmark_extractor import LandmarkFeatureExtractor
from ml.inference.predictor import DigitPredictor
from ml.training.model_mlp import DigitMLP


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def put_hud(frame, label_text, conf_text, latency_text, has_hand):
    x, y = 10, 30
    line = 28

    cv2.putText(frame, label_text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.putText(frame, conf_text, (x, y + line),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, latency_text, (x, y + 2 * line),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if not has_hand:
        cv2.putText(frame, "No hand detected", (x, y + 3 * line),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", type=str, default=default_device())
    args = parser.parse_args()

    detector = MediaPipeHandDetector()
    extractor = LandmarkFeatureExtractor()

    def model_ctor():
        return DigitMLP(
            input_dim=63,
            hidden_dims=(128, 64),
            num_classes=10,
            dropout=0.2,
        )

    predictor = DigitPredictor.load_from_checkpoint(
        model_ctor=model_ctor,
        checkpoint_path=Path(args.checkpoint),
        device=args.device,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    ema_latency: Optional[float] = None
    alpha = 0.1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            start = time.perf_counter()

            results = detector.process_frame(frame)
            has_hand = bool(results.multi_hand_landmarks)

            label_text = "Digit: -"
            conf_text = "Conf: -"
            
            if has_hand:
                features = extractor.extract(results)

                pred = predictor.predict(features)

                label_text = f"Digit: {pred.label}"
                conf_text = f"Conf: {pred.confidence:.2f}"

                detector.draw_landmarks(frame, results)

            end = time.perf_counter()
            latency = (end - start) * 1000.0
            ema_latency = latency if ema_latency is None else (
                alpha * latency + (1 - alpha) * ema_latency
            )

            latency_text = f"Latency: {ema_latency:.1f} ms"

            put_hud(frame, label_text, conf_text, latency_text, has_hand)

            cv2.imshow("Digits 0-9 Real-Time", frame)

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if hasattr(detector.hands, "close"):
            detector.hands.close()


if __name__ == "__main__":
    main()