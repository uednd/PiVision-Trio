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
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._fps = 0.0
        self._frame_count = 0
        self._last_time = 0.0
        self._output_buffer: Optional[np.ndarray] = None
        self._panel_base: Optional[np.ndarray] = None
        self._buffer_shape: Optional[Tuple[int, int, int]] = None
        self._panel_height = self._config.panel_height
        self._font_scale = self._config.font_scale
        self._font_scale_secondary = self._config.font_scale * 0.8
        self._help_scale = 0.5
        self._hint_scale = self._config.font_scale
        self._line1_y = 0
        self._line2_y = 0
        self._help_pos = (0, 0)
        self._help_text = "[1]Face [2]Color [3]Gesture [Q]Quit"
        self._mode_colors = {
            DetectionMode.FACE: self._config.color_primary,
            DetectionMode.COLOR: self._config.color_warning,
            DetectionMode.GESTURE: self._config.color_secondary,
        }

    def _ensure_buffers(self, frame_height: int, frame_width: int) -> None:
        panel_scale = max(0.8, min(1.4, frame_height / 480.0))
        panel_h = max(40, int(self._config.panel_height * panel_scale))
        shape = (frame_height, frame_width, panel_h)

        if self._buffer_shape == shape:
            return

        self._buffer_shape = shape
        self._panel_height = panel_h
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

        self._font_scale = self._config.font_scale * panel_scale
        self._font_scale_secondary = self._font_scale * 0.8
        self._help_scale = 0.5 * panel_scale
        self._hint_scale = max(0.7, self._font_scale * 1.1)
        self._line1_y = frame_height + int(panel_h * 0.45)
        self._line2_y = frame_height + int(panel_h * 0.85)

        help_size = cv2.getTextSize(
            self._help_text,
            self._font,
            self._help_scale,
            1
        )[0]
        help_x = max(10, frame_width - help_size[0] - 10)
        self._help_pos = (help_x, self._line2_y)

    @staticmethod
    def _apply_alpha(color: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int]:
        if alpha >= 1.0:
            return color
        return (
            int(color[0] * alpha),
            int(color[1] * alpha),
            int(color[2] * alpha),
        )

    def _draw_text(
        self,
        image: np.ndarray,
        text: str,
        org: Tuple[int, int],
        scale: float,
        color: Tuple[int, int, int],
        thickness: int,
        outline_thickness: int = 2,
        outline_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> None:
        if outline_thickness > 0:
            cv2.putText(
                image, text,
                org,
                self._font,
                scale,
                outline_color,
                thickness + outline_thickness,
                cv2.LINE_AA
            )
        cv2.putText(
            image, text,
            org,
            self._font,
            scale,
            color,
            thickness,
            cv2.LINE_AA
        )

    def get_mode_color(self, mode: DetectionMode) -> Tuple[int, int, int]:
        return self._mode_colors.get(mode, self._config.color_primary)

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
        result: Optional[DetectionResult] = None,
        show_help: bool = False,
        help_alpha: float = 1.0
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
        mode_color = self.get_mode_color(mode)

        self._draw_text(
            output,
            mode_text,
            (10, self._line1_y),
            self._font_scale,
            mode_color,
            self._config.font_thickness
        )

        # Detection result
        if result and result.message:
            result_text = result.message
            self._draw_text(
                output,
                result_text,
                (10, self._line2_y),
                self._font_scale_secondary,
                self._config.color_text,
                1
            )

        # FPS display (right side)
        fps_text = f"FPS: {self._fps:.1f}"
        text_size = cv2.getTextSize(
            fps_text,
            self._font,
            self._font_scale,
            self._config.font_thickness
        )[0]

        self._draw_text(
            output,
            fps_text,
            (w - text_size[0] - 10, self._line1_y),
            self._font_scale,
            self._config.color_secondary,
            self._config.font_thickness
        )

        if show_help:
            help_color = self._apply_alpha((160, 160, 160), help_alpha)
            help_outline = self._apply_alpha((0, 0, 0), help_alpha)
            self._draw_text(
                output,
                self._help_text,
                self._help_pos,
                self._help_scale,
                help_color,
                1,
                outline_color=help_outline
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
        h, w = frame.shape[:2]
        self._ensure_buffers(h, w)

        mode_name = self.MODE_NAMES.get(mode, "Unknown")
        indicator_text = mode_name
        indicator_color = self.get_mode_color(mode)

        text_size = cv2.getTextSize(
            indicator_text,
            self._font,
            self._hint_scale,
            self._config.font_thickness
        )[0]

        self._draw_text(
            output,
            indicator_text,
            (10, text_size[1] + 10),
            self._hint_scale,
            indicator_color,
            self._config.font_thickness
        )

        return output

    def render_hint(
        self,
        frame: np.ndarray,
        text: str,
        color: Tuple[int, int, int],
        alpha: float = 1.0
    ) -> np.ndarray:
        if not text:
            return frame

        h, w = frame.shape[:2]
        self._ensure_buffers(h, w)
        text_color = self._apply_alpha(color, alpha)

        text_size = cv2.getTextSize(
            text,
            self._font,
            self._hint_scale,
            self._config.font_thickness
        )[0]

        text_x = max(10, (w - text_size[0]) // 2)
        text_y = max(20, text_size[1] + 10)

        self._draw_text(
            frame,
            text,
            (text_x, text_y),
            self._hint_scale,
            text_color,
            self._config.font_thickness,
            outline_color=self._apply_alpha((0, 0, 0), alpha)
        )

        return frame

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
