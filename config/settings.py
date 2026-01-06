"""
Configuration settings for PiVision-Trio.
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Tuple


class DetectionMode(Enum):
    """Detection mode enumeration."""
    FACE = auto()
    COLOR = auto()
    GESTURE = auto()


@dataclass
class CameraConfig:
    """Camera configuration."""
    device_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class FaceDetectorConfig:
    """Face detector configuration."""
    # DNN model paths (relative to project root)
    model_file: str = "models/opencv_face_detector.caffemodel"
    config_file: str = "models/opencv_face_detector.prototxt"
    # Detection parameters
    confidence_threshold: float = 0.6
    # Fallback to Haar Cascade if DNN model not found
    haar_cascade: str = "haarcascade_frontalface_default.xml"
    use_dnn: bool = True


@dataclass
class ColorDetectorConfig:
    """Color detector configuration."""
    # Region of interest (center percentage of frame)
    roi_scale: float = 0.1
    # Minimum saturation and value for color detection
    min_saturation: int = 50
    min_value: int = 50


@dataclass
class GestureDetectorConfig:
    """Gesture detector configuration."""
    max_num_hands: int = 2
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5


@dataclass
class UIConfig:
    """UI rendering configuration."""
    # Colors (BGR format)
    color_primary: Tuple[int, int, int] = (0, 255, 0)      # Green
    color_secondary: Tuple[int, int, int] = (255, 255, 0)  # Cyan
    color_warning: Tuple[int, int, int] = (0, 165, 255)    # Orange
    color_text: Tuple[int, int, int] = (255, 255, 255)     # White
    color_bg: Tuple[int, int, int] = (0, 0, 0)             # Black
    # Font settings
    font_scale: float = 0.7
    font_thickness: int = 2
    # Info panel
    panel_height: int = 60


@dataclass
class AppConfig:
    """Main application configuration."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    face: FaceDetectorConfig = field(default_factory=FaceDetectorConfig)
    color: ColorDetectorConfig = field(default_factory=ColorDetectorConfig)
    gesture: GestureDetectorConfig = field(default_factory=GestureDetectorConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    # Key bindings
    key_bindings: Dict[str, int] = field(default_factory=lambda: {
        'face': ord('1'),
        'color': ord('2'),
        'gesture': ord('3'),
        'quit': ord('q'),
    })
    # Window name
    window_name: str = "PiVision-Trio"


# Global configuration instance
config = AppConfig()
