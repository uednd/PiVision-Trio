"""
State management module.
Handles detection mode switching and detector lifecycle.
"""
from typing import Dict, Optional

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config.settings import DetectionMode, config
from src.detectors.base import BaseDetector
from src.detectors.face import FaceDetector
from src.detectors.color import ColorDetector
from src.detectors.gesture import GestureDetector


class StateManager:
    """
    Manages detection states and mode switching.

    Responsibilities:
    - Initialize and store detectors
    - Handle mode switching via keyboard
    - Provide current active detector
    """

    def __init__(self):
        """Initialize state manager with all detectors."""
        self._detectors: Dict[DetectionMode, BaseDetector] = {}
        self._current_mode: DetectionMode = DetectionMode.FACE
        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize all detectors.

        Returns:
            True if at least one detector initialized successfully
        """
        # Create detectors
        self._detectors = {
            DetectionMode.FACE: FaceDetector(config.face),
            DetectionMode.COLOR: ColorDetector(config.color),
            DetectionMode.GESTURE: GestureDetector(config.gesture),
        }

        # Initialize all detectors
        success_count = 0
        for mode, detector in self._detectors.items():
            if detector.initialize():
                success_count += 1
                print(f"[StateManager] {detector.name} initialized")
            else:
                print(f"[StateManager] {detector.name} failed to initialize")

        self._initialized = success_count > 0

        if self._initialized:
            print(f"[StateManager] {success_count}/{len(self._detectors)} detectors ready")

        return self._initialized

    @property
    def current_mode(self) -> DetectionMode:
        """Get current detection mode."""
        return self._current_mode

    @property
    def current_detector(self) -> Optional[BaseDetector]:
        """Get current active detector."""
        return self._detectors.get(self._current_mode)

    def switch_mode(self, mode: DetectionMode) -> bool:
        """
        Switch to specified detection mode.

        Args:
            mode: Target detection mode

        Returns:
            True if switch successful
        """
        if mode not in self._detectors:
            return False

        detector = self._detectors[mode]
        if not detector.is_initialized:
            print(f"[StateManager] Cannot switch to {detector.name}: not initialized")
            return False

        self._current_mode = mode
        print(f"[StateManager] Switched to {detector.name}")
        return True

    def handle_key(self, key: int) -> Optional[bool]:
        """
        Handle keyboard input for mode switching.

        Args:
            key: Key code from cv2.waitKey()

        Returns:
            None if key not handled,
            True if mode switched,
            False if quit requested
        """
        if key == -1:
            return None

        key_bindings = config.key_bindings

        if key == key_bindings['quit']:
            return False

        if key == key_bindings['face']:
            self.switch_mode(DetectionMode.FACE)
            return True

        if key == key_bindings['color']:
            self.switch_mode(DetectionMode.COLOR)
            return True

        if key == key_bindings['gesture']:
            self.switch_mode(DetectionMode.GESTURE)
            return True

        return None

    def release(self) -> None:
        """Release all detector resources."""
        for detector in self._detectors.values():
            detector.release()
        self._detectors.clear()
        self._initialized = False
        print("[StateManager] All detectors released")

    def __enter__(self) -> 'StateManager':
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.release()
