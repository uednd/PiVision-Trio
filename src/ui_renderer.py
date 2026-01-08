"""
UI rendering module.
Handles display rendering, info panels, and visual feedback.
"""
from typing import Optional, Tuple

import cv2
import numpy as np

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config.settings import DetectionMode, UIConfig, config
from src.detectors.base import DetectionResult


class UIRenderer:
    """
    Renders UI elements and information panels.

    Responsibilities:
    - Render info panel with mode and FPS
    - Render help text
    - Combine frame with detection overlay
    """

    # Mode display names
    MODE_NAMES = {
        DetectionMode.FACE: "Face",
        DetectionMode.COLOR: "Color",
        DetectionMode.GESTURE: "Gesture",
    }

    def __init__(self, ui_config: Optional[UIConfig] = None):
        """
        Initialize UI renderer.

        Args:
            ui_config: UI configuration
        """
        self._config = ui_config or config.ui
        self._fps = 0.0
        self._frame_count = 0
        self._last_time = 0.0
        self._output_buffer: Optional[np.ndarray] = None
        self._panel_base: Optional[np.ndarray] = None
        self._buffer_shape: Optional[Tuple[int, int, int]] = None

    def _ensure_buffers(self, frame_height: int, frame_width: int) -> None:
        panel_h = self._config.panel_height
        shape = (frame_height, frame_width, panel_h)

        if self._buffer_shape == shape:
            return

        self._buffer_shape = shape
        self._output_buffer = np.zeros((frame_height + panel_h, frame_width, 3), dtype=np.uint8)
        self._panel_base = np.zeros((panel_h, frame_width, 3), dtype=np.uint8)
        self._panel_base[:] = self._config.color_bg

        cv2.line(
            self._panel_base,
            (0, 0),
            (frame_width, 0),
            self._config.color_secondary,
            2
        )

        help_text = "[1]Face [2]Color [3]Gesture [Q]Quit"
        cv2.putText(
            self._panel_base, help_text,
            (frame_width - 350, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (128, 128, 128),
            1
        )

    def update_fps(self, current_time: float) -> None:
        """
        Update FPS calculation.

        Args:
            current_time: Current timestamp in seconds
        """
        self._frame_count += 1

        if self._last_time == 0:
            self._last_time = current_time
            return

        elapsed = current_time - self._last_time

        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_time = current_time

    def render_info_panel(
        self,
        frame: np.ndarray,
        mode: DetectionMode,
        result: Optional[DetectionResult] = None
    ) -> np.ndarray:
        """
        Render information panel at bottom of frame.

        Args:
            frame: BGR image frame
            mode: Current detection mode
            result: Detection result for status display

        Returns:
            Frame with info panel
        """
        h, w = frame.shape[:2]
        self._ensure_buffers(h, w)
        output = self._output_buffer
        output[:h, :] = frame
        output[h:, :] = self._panel_base

        # Mode name
        mode_name = self.MODE_NAMES.get(mode, "Unknown")
        mode_text = f"[{mode.value}] {mode_name}"

        cv2.putText(
            output, mode_text,
            (10, h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            self._config.font_scale,
            self._config.color_primary,
            self._config.font_thickness
        )

        # Detection result
        if result and result.message:
            result_text = result.message
            cv2.putText(
                output, result_text,
                (10, h + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                self._config.font_scale * 0.8,
                self._config.color_text,
                1
            )

        # FPS display (right side)
        fps_text = f"FPS: {self._fps:.1f}"
        text_size = cv2.getTextSize(
            fps_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            self._config.font_scale,
            self._config.font_thickness
        )[0]

        cv2.putText(
            output, fps_text,
            (w - text_size[0] - 10, h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            self._config.font_scale,
            self._config.color_secondary,
            self._config.font_thickness
        )

        return output

    def render_mode_indicator(
        self,
        frame: np.ndarray,
        mode: DetectionMode
    ) -> np.ndarray:
        """
        Render mode indicator at top-left of frame.

        Args:
            frame: BGR image frame
            mode: Current detection mode

        Returns:
            Frame with mode indicator
        """
        output = frame

        mode_name = self.MODE_NAMES.get(mode, "Unknown")
        indicator_text = mode_name

        # Background
        text_size = cv2.getTextSize(
            indicator_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            2
        )[0]

        cv2.rectangle(
            output,
            (5, 5),
            (text_size[0] + 15, text_size[1] + 15),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            output, indicator_text,
            (10, text_size[1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self._config.color_primary,
            2
        )

        return output

    @staticmethod
    def render_no_camera(width: int, height: int) -> np.ndarray:
        """
        Render placeholder when camera not available.

        Args:
            width: Frame width
            height: Frame height

        Returns:
            Placeholder frame
        """
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        text = "Camera Not Available"
        text_size = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            2
        )[0]

        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2

        cv2.putText(
            frame, text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

        return frame
