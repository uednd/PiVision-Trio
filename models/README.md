# Models Directory

This directory contains pre-trained model files for face detection.

## Included Models

| File | Size | Description |
|------|------|-------------|
| `opencv_face_detector.caffemodel` | ~10MB | SSD-ResNet10 weights |
| `opencv_face_detector.prototxt` | ~28KB | Model architecture |

## Model Source

- **Architecture**: SSD (Single Shot Detector) with ResNet-10 backbone
- **Input Size**: 300x300 pixels
- **Source**: [OpenCV DNN Samples](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)

## Fallback

If these model files are removed or corrupted, the system will automatically fall back to Haar Cascade face detection (built into OpenCV, lower accuracy but no additional files needed).
