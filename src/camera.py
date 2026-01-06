"""
Camera management module.
Handles camera initialization, frame capture, and resource cleanup.
"""
import cv2
import numpy as np
from typing import Optional, Tuple

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config.settings import CameraConfig


class CameraManager:
    """
    Manages camera device lifecycle and frame capture.

    Responsibilities:
    - Initialize and configure camera device
    - Capture frames
    - Release resources on cleanup
    """

    def __init__(self, config: Optional[CameraConfig] = None):
        """
        Initialize camera manager.

        Args:
            config: Camera configuration. Uses default if not provided.
        """
        self._config = config or CameraConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened = False

    def open(self) -> bool:
        """
        Open camera device with configured parameters.

        Returns:
            True if camera opened successfully, False otherwise.
        """
        if self._is_opened:
            return True

        self._cap = cv2.VideoCapture(self._config.device_id)

        if not self._cap.isOpened():
            return False

        # Configure camera parameters
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)

        # Verify settings were applied
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_width != self._config.width or actual_height != self._config.height:
            print(f"[CameraManager] Warning: Requested {self._config.width}x{self._config.height}, "
                  f"got {actual_width}x{actual_height}")

        self._is_opened = True
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.

        Returns:
            Tuple of (success, frame). Frame is None if read failed.
        """
        if not self._is_opened or self._cap is None:
            return False, None

        ret, frame = self._cap.read()

        if not ret:
            return False, None

        return True, frame

    def release(self) -> None:
        """Release camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_opened = False

    @property
    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self._is_opened

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Get configured frame size (width, height)."""
        return self._config.width, self._config.height

    def __enter__(self) -> 'CameraManager':
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.release()
