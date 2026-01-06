"""
Detector modules for PiVision-Trio.
"""
from .base import BaseDetector
from .face import FaceDetector
from .color import ColorDetector
from .gesture import GestureDetector

__all__ = ['BaseDetector', 'FaceDetector', 'ColorDetector', 'GestureDetector']
