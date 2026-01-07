# PiVision-Trio

**实时三态视觉识别：人脸 / 颜色 / 手势（0-10）**

> ⚠️ 仅在 Raspberry Pi 5 (Ubuntu) 通过测试，其他平台未验证。

## 功能特性

- 人脸检测：OpenCV DNN（SSD-ResNet10），模型缺失时回退到 Haar 级联。
- 颜色识别：中心 ROI HSV 统计，256+ 颜色命名，过滤低饱和/低亮度像素。
- 手势识别：MediaPipe Hands，按伸展手指数识别 0-10。

## 运行预览

![Face](assets/face.png)

![Color01](assets/color01.png)

![Color02](assets/color02.png)

![Gesture](assets/gesture.png)

## 键位说明

- `1`: 人脸检测
- `2`: 颜色识别
- `3`: 手势识别
- `q`: 退出

## 运行环境

- Python 3.11
- 摄像头设备（USB 或 CSI）

## 本地开发

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/main.py
```

## 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--camera` / `-c` | `0` | 摄像头设备 ID |
| `--width` / `-W` | `640` | 帧宽度 |
| `--height` / `-H` | `480` | 帧高度 |

## 检测器说明

### 人脸检测

- 使用 DNN 模型 `models/opencv_face_detector.*`。
- 模型缺失时回退到 Haar 级联。
- 输出人脸框与置信度（Haar 模式下置信度为固定值）。

### 颜色识别

- 采样中心 ROI，过滤低饱和/低亮度像素，并基于 HSV 统计生成颜色名称。
- 叠加显示颜色块与 HSV 数值。
- `ColorDetectorConfig.min_saturation` 和 `min_value` 参与像素筛选。

### 手势识别

- 依赖 MediaPipe Hands。
- 按伸展手指数量输出 0-10（双手时为总和）。
- 绘制手部关键点与连线，并在右上角显示结果。

## 配置说明

默认配置集中在 `config/settings.py`。

| 模块 | 配置项 | 默认值 | 说明 |
| --- | --- | --- | --- |
| Camera | `device_id` | `0` | 摄像头设备 ID |
| Camera | `width` / `height` | `640` / `480` | 帧尺寸 |
| Camera | `fps` | `30` | 目标帧率 |
| Face | `confidence_threshold` | `0.6` | DNN 置信度阈值 |
| Face | `use_dnn` | `True` | 优先使用 DNN |
| Color | `roi_scale` | `0.1` | ROI 占比 |
| Color | `min_saturation` | `50` | 颜色采样的最低饱和度 |
| Color | `min_value` | `50` | 颜色采样的最低亮度 |
| Gesture | `max_num_hands` | `2` | 最多检测手数 |
| Gesture | `min_detection_confidence` | `0.7` | 检测置信度 |
| Gesture | `min_tracking_confidence` | `0.5` | 跟踪置信度 |
| UI | `panel_height` | `60` | 信息面板高度 |
| App | `key_bindings` | `1/2/3/q` | 模式与退出键位 |

## 模型

模型文件位于 `models/`，来源与授权请见 `models/README.md`。

模型文件由 Git LFS 管理，本地克隆后执行：

```bash
git lfs pull
```

## 许可证

本项目使用 MIT 许可证，详情见 `LICENSE`。
