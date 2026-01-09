#!/usr/bin/env python3
"""
PiVision-Trio: Camera-based tri-state recognition system.

A real-time computer vision application for Raspberry Pi 5 featuring:
- Face Detection (OpenCV DNN / Haar Cascade)
- Color Recognition (HSV Analysis)
- Gesture Recognition (MediaPipe Hands, 0-10)

Usage:
    python src/main.py [--camera DEVICE_ID]

Controls:
    1 - Face Detection mode
    2 - Color Recognition mode
    3 - Gesture Recognition mode
    Q - Quit
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import DetectionMode, config
from src.camera import CameraManager
from src.state_manager import StateManager
from src.ui_renderer import UIRenderer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PiVision-Trio: Camera-based tri-state recognition"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--width", "-W",
        type=int,
        default=640,
        help="Frame width (default: 640)"
    )
    parser.add_argument(
        "--height", "-H",
        type=int,
        default=480,
        help="Frame height (default: 480)"
    )
    return parser.parse_args()


def _min_interval(max_fps: float) -> float:
    if max_fps <= 0:
        return 0.0
    return 1.0 / max_fps


def main() -> int:
    """
    Main application entry point.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    args = parse_args()

    # Update config from arguments
    config.camera.device_id = args.camera
    config.camera.width = args.width
    config.camera.height = args.height

    cv2.setUseOptimized(True)

    print("=" * 50)
    print("PiVision-Trio - Tri-State Recognition System")
    print("=" * 50)
    print(f"Camera: {args.camera}")
    print(f"Resolution: {args.width}x{args.height}")
    print("-" * 50)

    # Initialize components
    camera = CameraManager(config.camera)
    state_manager = StateManager()
    ui_renderer = UIRenderer(config.ui)

    # Open camera
    if not camera.open():
        print("[Error] Failed to open camera")
        return 1

    print("[Camera] Opened successfully")

    # Initialize detectors
    if not state_manager.initialize():
        print("[Error] No detectors available")
        camera.release()
        return 1

    print("-" * 50)
    print("Controls:")
    print("  [1] Face Detection")
    print("  [2] Color Recognition")
    print("  [3] Gesture Recognition")
    print("  [Q] Quit")
    print("=" * 50)

    # Create display window
    cv2.namedWindow(config.window_name, cv2.WINDOW_AUTOSIZE)

    detection_intervals = {
        DetectionMode.FACE: _min_interval(config.performance.face_max_fps),
        DetectionMode.COLOR: _min_interval(config.performance.color_max_fps),
        DetectionMode.GESTURE: _min_interval(config.performance.gesture_max_fps),
    }
    last_detection_time = {mode: 0.0 for mode in detection_intervals}
    last_results = {mode: None for mode in detection_intervals}
    last_mode = state_manager.current_mode
    start_time = time.perf_counter()

    try:
        while True:
            # Capture frame
            ret, frame = camera.read()
            loop_time = time.perf_counter()
            result = None

            if not ret or frame is None:
                # Show placeholder if camera fails
                frame = UIRenderer.render_no_camera(
                    config.camera.width,
                    config.camera.height
                )
            else:
                # Flip frame horizontally (mirror effect)
                frame = cv2.flip(frame, 1)

                # Get current detector
                detector = state_manager.current_detector
                current_mode = state_manager.current_mode

                if current_mode != last_mode:
                    last_results[current_mode] = None
                    last_detection_time[current_mode] = 0.0
                    last_mode = current_mode

                if detector and detector.is_initialized:
                    # Perform detection
                    min_interval = detection_intervals.get(current_mode, 0.0)

                    if min_interval <= 0.0 or (loop_time - last_detection_time[current_mode]) >= min_interval:
                        result = detector.detect(frame)
                        last_results[current_mode] = result
                        last_detection_time[current_mode] = loop_time
                    else:
                        result = last_results.get(current_mode)

                    # Draw detection results
                    if result:
                        frame = detector.draw(frame, result)

                # Add mode indicator
                frame = ui_renderer.render_mode_indicator(
                    frame,
                    state_manager.current_mode
                )

            # Update FPS
            ui_renderer.update_fps(loop_time)

            help_elapsed = loop_time - start_time
            show_help = help_elapsed <= config.ui.help_display_seconds
            help_alpha = 1.0
            if show_help and config.ui.help_fade_seconds > 0.0:
                remaining = config.ui.help_display_seconds - help_elapsed
                if remaining < config.ui.help_fade_seconds:
                    help_alpha = max(0.0, remaining / config.ui.help_fade_seconds)

            # Render info panel
            display_frame = ui_renderer.render_info_panel(
                frame,
                state_manager.current_mode,
                result,
                show_help=show_help,
                help_alpha=help_alpha
            )

            # Show frame
            cv2.imshow(config.window_name, display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            action = state_manager.handle_key(key)

            if action is False:  # Quit requested
                print("\n[Quit] User requested exit")
                break

    except KeyboardInterrupt:
        print("\n[Quit] Keyboard interrupt")

    finally:
        # Cleanup
        state_manager.release()
        camera.release()
        cv2.destroyAllWindows()
        print("[Cleanup] Done")

    return 0


if __name__ == "__main__":
    sys.exit(main())
