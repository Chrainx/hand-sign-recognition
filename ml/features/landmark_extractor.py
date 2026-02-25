import numpy as np


class LandmarkFeatureExtractor:
    def __init__(self):
        pass

    def extract(self, results):
        """
        Extract a 63-dimensional feature vector from MediaPipe results.

        Returns:
            np.ndarray of shape (63,) if hand detected,
            None otherwise.
        """

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]

        # Wrist landmark (index 0)
        wrist = hand_landmarks.landmark[0]

        features = []

        for landmark in hand_landmarks.landmark:
            features.extend([
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z
            ])

        return np.array(features, dtype=np.float32)