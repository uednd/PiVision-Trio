"""
Color recognition module using HSV color space analysis.
Supports 256+ color variations.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
from config.settings import ColorDetectorConfig
from src.detectors.base import BaseDetector, DetectionResult


@dataclass
class ColorInfo:
    name: str
    h: int
    s: int
    v: int
    bgr: Tuple[int, int, int]


HUE_NAMES = [
    (0, "Red"),
    (15, "Orange"),
    (30, "Yellow"),
    (45, "Lime"),
    (60, "Green"),
    (75, "Emerald"),
    (90, "Cyan"),
    (105, "SkyBlue"),
    (120, "Blue"),
    (135, "Indigo"),
    (150, "Purple"),
    (165, "Magenta"),
    (180, "Red"),
]

SAT_PREFIXES = [
    (0, "Gray"),
    (30, "Pale "),
    (80, "Light "),
    (150, ""),
    (220, "Vivid "),
]

VAL_PREFIXES = [
    (0, "Black"),
    (30, "Dark "),
    (80, "Deep "),
    (150, ""),
    (200, "Bright "),
    (240, "White"),
]


class ColorDetector(BaseDetector):
    def __init__(self, config: Optional[ColorDetectorConfig] = None):
        super().__init__("Color Recognition")
        self._config = config or ColorDetectorConfig()
        self._roi: Optional[Tuple[int, int, int, int]] = None

    def initialize(self) -> bool:
        self._is_initialized = True
        print("[ColorDetector] Initialized with 256+ color support")
        return True

    def _get_roi(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        roi_w = int(w * self._config.roi_scale)
        roi_h = int(h * self._config.roi_scale)
        roi_x = (w - roi_w) // 2
        roi_y = (h - roi_h) // 2
        return roi_x, roi_y, roi_w, roi_h

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if not self._is_initialized:
            return DetectionResult(False, message="Detector not initialized")

        roi_x, roi_y, roi_w, roi_h = self._get_roi(frame)
        self._roi = (roi_x, roi_y, roi_w, roi_h)
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        mask = (s_channel >= self._config.min_saturation) & (v_channel >= self._config.min_value)
        if np.any(mask):
            h_values = h_channel[mask]
            s_values = s_channel[mask]
            v_values = v_channel[mask]
        else:
            h_values = h_channel.reshape(-1)
            s_values = s_channel.reshape(-1)
            v_values = v_channel.reshape(-1)

        h_mean = self._circular_mean(h_values)
        s_mean = int(np.median(s_values))
        v_mean = int(np.median(v_values))

        color_name = self._get_color_name(h_mean, s_mean, v_mean)
        bgr_color = self._hsv_to_bgr(h_mean, s_mean, v_mean)

        color_info = ColorInfo(
            name=color_name,
            h=h_mean,
            s=s_mean,
            v=v_mean,
            bgr=bgr_color
        )

        return DetectionResult(
            success=True,
            data={
                "roi": self._roi,
                "color": color_info,
                "hsv": (h_mean, s_mean, v_mean)
            },
            message=color_name,
            confidence=1.0
        )

    def _get_color_name(self, h: int, s: int, v: int) -> str:
        if v < 30:
            return "Black"
        if v > 240 and s < 30:
            return "White"
        if s < 30:
            if v < 80:
                return "Dark Gray"
            elif v < 150:
                return "Gray"
            elif v < 200:
                return "Light Gray"
            else:
                return "White"

        hue_name = "Red"
        for hue_val, name in HUE_NAMES:
            if h >= hue_val:
                hue_name = name

        sat_prefix = ""
        for sat_val, prefix in SAT_PREFIXES:
            if s >= sat_val:
                sat_prefix = prefix

        val_prefix = ""
        for val_val, prefix in VAL_PREFIXES:
            if v >= val_val:
                val_prefix = prefix

        if val_prefix in ("Black", "White"):
            return val_prefix
        if sat_prefix == "Gray":
            if v < 80:
                return "Dark Gray"
            elif v < 150:
                return "Gray"
            else:
                return "Light Gray"

        name = f"{val_prefix}{sat_prefix}{hue_name}"
        return name.strip()

    def _circular_mean(self, hue_values: np.ndarray) -> int:
        if hue_values.size == 0:
            return 0

        angles = hue_values.astype(np.float32) * (2 * np.pi / 180.0)
        sin_mean = float(np.mean(np.sin(angles)))
        cos_mean = float(np.mean(np.cos(angles)))

        if np.isclose(sin_mean, 0.0) and np.isclose(cos_mean, 0.0):
            return int(np.median(hue_values))

        mean_angle = np.arctan2(sin_mean, cos_mean)
        if mean_angle < 0:
            mean_angle += 2 * np.pi

        hue = int(round(mean_angle * 180.0 / (2 * np.pi)))
        return max(0, min(180, hue))

    def _hsv_to_bgr(self, h: int, s: int, v: int) -> Tuple[int, int, int]:
        hsv = np.uint8([[[h, s, v]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return (int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2]))

    def draw(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        if not result.success:
            return frame

        output = frame.copy()
        data = result.data
        roi = data.get("roi")

        if roi:
            roi_x, roi_y, roi_w, roi_h = roi

            cv2.rectangle(
                output,
                (roi_x, roi_y),
                (roi_x + roi_w, roi_y + roi_h),
                (255, 255, 255),
                2
            )

            color_info = data.get("color")
            if color_info:
                indicator_size = 60
                indicator_x = roi_x + roi_w + 20
                indicator_y = roi_y + (roi_h - indicator_size) // 2

                cv2.rectangle(
                    output,
                    (indicator_x, indicator_y),
                    (indicator_x + indicator_size, indicator_y + indicator_size),
                    color_info.bgr,
                    -1
                )

                cv2.rectangle(
                    output,
                    (indicator_x, indicator_y),
                    (indicator_x + indicator_size, indicator_y + indicator_size),
                    (255, 255, 255),
                    2
                )

                text = color_info.name
                font_scale = 1.0
                thickness = 2
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

                if text_size[0] > roi_w:
                    font_scale = 0.7
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

                text_x = roi_x + (roi_w - text_size[0]) // 2
                text_y = roi_y - 15

                cv2.putText(
                    output, text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness + 2
                )
                cv2.putText(
                    output, text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness
                )

                hsv = data.get("hsv", (0, 0, 0))
                hsv_text = f"H:{hsv[0]} S:{hsv[1]} V:{hsv[2]}"
                cv2.putText(
                    output, hsv_text,
                    (indicator_x, indicator_y + indicator_size + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1
                )

        return output

    def release(self) -> None:
        self._roi = None
        super().release()
