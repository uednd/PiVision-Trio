"""
Face detection module using OpenCV DNN or Haar Cascade.
"""
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from config.settings import FaceDetectorConfig
from src.detectors.base import BaseDetector, DetectionResult


@dataclass
class FaceBox:
    """Face bounding box with confidence."""
    x: int
    y: int
    width: int
    height: int
    confidence: float


class FaceDetector(BaseDetector):
    """
    Face detector using OpenCV DNN (preferred) or Haar Cascade (fallback).

    DNN uses SSD-ResNet10 model for better accuracy.
    Falls back to Haar Cascade if DNN model files not found.
    """

    def __init__(self, config: Optional[FaceDetectorConfig] = None):
        """
        Initialize face detector.

        Args:
            config: Face detector configuration
        """
        super().__init__("Face Detection")
        self._config = config or FaceDetectorConfig()
        self._net: Optional[cv2.dnn.Net] = None
        self._cascade: Optional[cv2.CascadeClassifier] = None
        self._use_dnn = False

    def initialize(self) -> bool:
        """
        Initialize face detection model.

        Tries DNN first, falls back to Haar Cascade.
        """
        if self._is_initialized:
            return True

        # Try DNN model first
        if self._config.use_dnn:
            self._use_dnn = self._init_dnn()

        # Fallback to Haar Cascade
        if not self._use_dnn:
            self._use_dnn = False
            if not self._init_haar():
                return False

        self._is_initialized = True
        return True

    def _init_dnn(self) -> bool:
        """Initialize DNN-based face detector."""
        model_path = self._config.model_file
        config_path = self._config.config_file

        if not os.path.exists(model_path) or not os.path.exists(config_path):
            print(f"[FaceDetector] DNN model not found, falling back to Haar Cascade")
            return False

        try:
            self._net = cv2.dnn.readNetFromCaffe(config_path, model_path)
            print("[FaceDetector] Using DNN (SSD-ResNet10) model")
            return True
        except cv2.error as e:
            print(f"[FaceDetector] Failed to load DNN model: {e}")
            return False

    def _init_haar(self) -> bool:
        """Initialize Haar Cascade face detector."""
        # Try to find Haar Cascade file
        cascade_paths = [
            self._config.haar_cascade,
            cv2.data.haarcascades + self._config.haar_cascade,
        ]

        for path in cascade_paths:
            if os.path.exists(path):
                self._cascade = cv2.CascadeClassifier(path)
                if not self._cascade.empty():
                    print(f"[FaceDetector] Using Haar Cascade from {path}")
                    return True

        print("[FaceDetector] Failed to load Haar Cascade")
        return False

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect faces in frame.

        Args:
            frame: BGR image frame

        Returns:
            DetectionResult with list of FaceBox in data
        """
        if not self._is_initialized:
            return DetectionResult(False, message="Detector not initialized")

        if self._use_dnn:
            faces = self._detect_dnn(frame)
        else:
            faces = self._detect_haar(frame)

        if not faces:
            return DetectionResult(
                success=True,
                data=[],
                message="No faces detected",
                confidence=0.0
            )

        avg_confidence = sum(f.confidence for f in faces) / len(faces)

        return DetectionResult(
            success=True,
            data=faces,
            message=f"Detected {len(faces)} face(s)",
            confidence=avg_confidence
        )

    def _detect_dnn(self, frame: np.ndarray) -> List[FaceBox]:
        """Detect faces using DNN."""
        h, w = frame.shape[:2]

        # Prepare input blob
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0), False, False
        )

        self._net.setInput(blob)
        detections = self._net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self._config.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)

                # Clamp to frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                faces.append(FaceBox(
                    x=x1, y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    confidence=float(confidence)
                ))

        return faces

    def _detect_haar(self, frame: np.ndarray) -> List[FaceBox]:
        """Detect faces using Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        faces = []
        for (x, y, w, h) in detections:
            faces.append(FaceBox(
                x=x, y=y,
                width=w, height=h,
                confidence=0.8  # Haar doesn't provide confidence
            ))

        return faces

    def draw(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """
        Draw face detection results on frame.

        Args:
            frame: BGR image frame
            result: Detection result

        Returns:
            Frame with face boxes drawn
        """
        if not result.success or not result.data:
            return frame

        output = frame.copy()

        for face in result.data:
            # Draw bounding box
            cv2.rectangle(
                output,
                (face.x, face.y),
                (face.x + face.width, face.y + face.height),
                (0, 255, 0),  # Green
                2
            )

            # Draw confidence label
            label = f"{face.confidence:.0%}"
            label_y = face.y - 10 if face.y > 20 else face.y + 20

            cv2.putText(
                output, label,
                (face.x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

        return output

    def release(self) -> None:
        """Release detector resources."""
        self._net = None
        self._cascade = None
        super().release()
