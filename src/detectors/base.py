"""
Base detector interface.
All detectors must inherit from this class.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class DetectionResult:
    """
    Standard detection result container.

    Attributes:
        success: Whether detection was successful
        data: Detection-specific data (bounding boxes, colors, gestures, etc.)
        message: Human-readable result message
        confidence: Detection confidence score (0.0 - 1.0)
    """
    success: bool
    data: Any = None
    message: str = ""
    confidence: float = 0.0

    def __bool__(self) -> bool:
        return self.success


class BaseDetector(ABC):
    """
    Abstract base class for all detectors.

    Defines the interface that all detection modules must implement.
    Follows the Open-Closed Principle: open for extension, closed for modification.
    """

    def __init__(self, name: str):
        """
        Initialize detector.

        Args:
            name: Human-readable detector name for display
        """
        self._name = name
        self._is_initialized = False

    @property
    def name(self) -> str:
        """Get detector name."""
        return self._name

    @property
    def is_initialized(self) -> bool:
        """Check if detector is initialized."""
        return self._is_initialized

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize detector resources (models, etc.).

        Returns:
            True if initialization successful, False otherwise.
        """
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Perform detection on a frame.

        Args:
            frame: BGR image frame from camera

        Returns:
            DetectionResult containing detection outcome
        """
        pass

    @abstractmethod
    def draw(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """
        Draw detection results on frame.

        Args:
            frame: BGR image frame to draw on
            result: Detection result from detect()

        Returns:
            Frame with detection visualization
        """
        pass

    def release(self) -> None:
        """
        Release detector resources.
        Override in subclass if cleanup needed.
        """
        self._is_initialized = False

    def __enter__(self) -> 'BaseDetector':
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.release()
