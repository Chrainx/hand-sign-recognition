import argparse
import csv
from pathlib import Path
import cv2

from ml.detection.mediapipe_detector import MediaPipeHandDetector
from ml.features.landmark_extractor import LandmarkFeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="Digit Data Collection")
    parser.add_argument("--digit", type=int, required=True, help="Digit label (0-9)")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.digit < 0 or args.digit > 9:
        raise ValueError("Digit must be between 0 and 9")

    save_dir = Path("data/raw")
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "digits_dataset.csv"

    detector = MediaPipeHandDetector()
    extractor = LandmarkFeatureExtractor()

    cap = cv2.VideoCapture(0)
    collected = 0

    print("Press 's' to save sample | 'q' to quit")

    while cap.isOpened() and collected < args.samples:
        ret, frame = cap.read()
        if not ret:
            continue

        results = detector.process_frame(frame)
        features = extractor.extract(results)

        if features is not None and features.shape[0] == 63:
            cv2.putText(frame,
                        f"Digit: {args.digit} | {collected}/{args.samples}",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([args.digit] + features.tolist())
                collected += 1
                print(f"Saved {collected}/{args.samples}")

            elif key == ord("q"):
                break

        cv2.imshow("Digit Collector", frame)

    cap.release()
    cv2.destroyAllWindows()
    print("Collection complete.")


if __name__ == "__main__":
    main()