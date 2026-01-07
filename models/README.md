# 模型目录

本目录包含用于人脸检测的预训练模型文件。

## 已包含模型

| 文件 | 大小 | 说明 |
| --- | --- | --- |
| `opencv_face_detector.caffemodel` | ~10MB | SSD-ResNet10 权重 |
| `opencv_face_detector.prototxt` | ~28KB | 模型结构 |

## 模型来源

- **架构**：SSD（Single Shot Detector）+ ResNet-10
- **输入尺寸**：300x300 像素
- **来源**：[OpenCV DNN Samples](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)

## 回退策略

如果这些模型文件缺失或损坏，系统会自动回退到 Haar 级联人脸检测
（OpenCV 内置，精度较低但无需额外文件）。
