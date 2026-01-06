"""
Gesture recognition module using MediaPipe Hands.
Recognizes hand gestures for numbers 0-10.
"""
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

import cv2
import numpy as np

import math

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from config.settings import GestureDetectorConfig
from src.detectors.base import BaseDetector, DetectionResult


class Gesture(IntEnum):
    UNKNOWN = -1
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10


@dataclass
class GestureInfo:
    gesture: Gesture
    landmarks: Optional[List[List[Tuple[float, float, float]]]] = None
    handedness: List[str] = None
    finger_count: int = 0


FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]
FINGER_MCPS = [2, 5, 9, 13, 17]


class GestureDetector(BaseDetector):
    def __init__(self, config: Optional[GestureDetectorConfig] = None):
        super().__init__("Gesture Recognition")
        self._config = config or GestureDetectorConfig()
        self._hands = None
        self._mp_hands = None
        self._mp_draw = None

    def initialize(self) -> bool:
        if not MEDIAPIPE_AVAILABLE:
            print("[GestureDetector] MediaPipe not installed")
            return False

        if self._is_initialized:
            return True

        try:
            self._mp_hands = mp.solutions.hands
            self._mp_draw = mp.solutions.drawing_utils

            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self._config.max_num_hands,
                min_detection_confidence=self._config.min_detection_confidence,
                min_tracking_confidence=self._config.min_tracking_confidence
            )

            self._is_initialized = True
            print("[GestureDetector] MediaPipe Hands initialized (supports 0-10)")
            return True

        except Exception as e:
            print(f"[GestureDetector] Failed to initialize: {e}")
            return False

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if not self._is_initialized:
            return DetectionResult(False, message="Detector not initialized")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return DetectionResult(
                success=True,
                data=GestureInfo(gesture=Gesture.UNKNOWN, finger_count=0),
                message="No hand detected",
                confidence=0.0
            )

        all_landmarks = []
        all_handedness = []
        total_fingers = 0

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = "Right"
            if results.multi_handedness and i < len(results.multi_handedness):
                handedness = results.multi_handedness[i].classification[0].label

            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            all_landmarks.append(landmarks)
            all_handedness.append(handedness)

            fingers = self._count_extended_fingers(landmarks, handedness)
            total_fingers += fingers

        gesture = self._map_count_to_gesture(total_fingers)

        gesture_info = GestureInfo(
            gesture=gesture,
            landmarks=all_landmarks,
            handedness=all_handedness,
            finger_count=total_fingers
        )

        if gesture == Gesture.UNKNOWN:
            message = f"Fingers: {total_fingers}"
            confidence = 0.3
        else:
            message = f"Gesture: {gesture.value}"
            confidence = 0.9

        return DetectionResult(
            success=True,
            data=gesture_info,
            message=message,
            confidence=confidence
        )

    def _map_count_to_gesture(self, count: int) -> Gesture:
        if 0 <= count <= 10:
            return Gesture(count)
        return Gesture.UNKNOWN

    def _count_extended_fingers(
        self,
        landmarks: List[Tuple[float, float, float]],
        handedness: str
    ) -> int:
        count = 0

        thumb_angle = self._calc_angle(landmarks[1], landmarks[2], landmarks[4])
        if thumb_angle > 150:
            count += 1

        for i in range(1, 5):
            mcp = landmarks[FINGER_MCPS[i]]
            pip = landmarks[FINGER_PIPS[i]]
            tip = landmarks[FINGER_TIPS[i]]

            angle = self._calc_angle(mcp, pip, tip)
            if angle > 160:
                count += 1

        return count

    def _calc_angle(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        p3: Tuple[float, float, float]
    ) -> float:
        v1 = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
        v2 = (p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2])

        dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

        len1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
        len2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)

        if len1 == 0 or len2 == 0:
            return 180.0

        cos_angle = dot / (len1 * len2)
        cos_angle = max(-1.0, min(1.0, cos_angle))

        return math.degrees(math.acos(cos_angle))

    def draw(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        if not result.success:
            return frame

        output = frame.copy()
        gesture_info: GestureInfo = result.data

        if gesture_info.landmarks:
            h, w = frame.shape[:2]
            colors = [(0, 255, 0), (255, 0, 0)]

            for hand_idx, landmarks in enumerate(gesture_info.landmarks):
                color = colors[hand_idx % len(colors)]

                connections = self._mp_hands.HAND_CONNECTIONS if self._mp_hands else []
                for connection in connections:
                    start_idx, end_idx = connection
                    start = landmarks[start_idx]
                    end = landmarks[end_idx]

                    start_point = (int(start[0] * w), int(start[1] * h))
                    end_point = (int(end[0] * w), int(end[1] * h))

                    cv2.line(output, start_point, end_point, color, 2)

                for lm in landmarks:
                    cx, cy = int(lm[0] * w), int(lm[1] * h)
                    cv2.circle(output, (cx, cy), 5, (255, 0, 255), -1)

            if gesture_info.gesture != Gesture.UNKNOWN:
                gesture_text = str(gesture_info.gesture.value)
            else:
                gesture_text = str(gesture_info.finger_count)

            font_scale = 3
            thickness = 5

            text_size = cv2.getTextSize(
                gesture_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness
            )[0]

            text_x = w - text_size[0] - 30
            text_y = text_size[1] + 30

            cv2.rectangle(
                output,
                (text_x - 10, text_y - text_size[1] - 10),
                (text_x + text_size[0] + 10, text_y + 10),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                output, gesture_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 255, 255), thickness
            )

            num_hands = len(gesture_info.landmarks)
            if num_hands > 1:
                info_text = f"Hands: {num_hands}"
                cv2.putText(
                    output, info_text,
                    (text_x, text_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2
                )

        return output

    def release(self) -> None:
        if self._hands:
            self._hands.close()
            self._hands = None
        self._mp_hands = None
        self._mp_draw = None
        super().release()
