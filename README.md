# 实时交通分析系统 (Real-Time Traffic Analysis System)

基于 YOLOv12 + ByteTrack 的实时交通分析系统，实现车辆检测、跟踪、碰撞风险预测和智能违规识别。

## 系统要求

- Docker (必须)
- 显示器 (GUI模式需要)

## 快速开始

### 1. 构建Docker镜像

```bash
docker build -t traffic-analysis:latest .
```

### 2. 运行方式

#### 方式A: 批量检测（推荐，无需显示器）

```bash
# Linux/macOS
docker run --rm \
  -v $(pwd):/app \
  -v /path/to/your/videos:/videos \
  traffic-analysis:latest \
  python3 scripts/batch_detect.py -i /videos/your_video.mp4 -o /app/data/results -n 10

# Windows PowerShell
docker run --rm -v ${PWD}:/app -v C:\your\videos:/videos traffic-analysis:latest python3 scripts/batch_detect.py -i /videos/your_video.mp4 -o /app/data/results -n 10
```

参数说明:
- `-i`: 输入视频文件或目录
- `-o`: 输出目录
- `-n`: 抽帧间隔（每N帧保存一次，默认30）

输出文件:
- `frame_*_orig.jpg` - 原始帧
- `frame_*_detect.jpg` - 检测结果（带风险颜色标注）
- `results.json` - 详细数据

#### 方式B: GUI界面（需要显示器）

```bash
# Linux
xhost +local:docker
docker run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/app \
  traffic-analysis:latest \
  python3 -m src.gui.main_window

# Windows (需要VcXsrv)
docker run --rm -e DISPLAY=host.docker.internal:0 -v ${PWD}:/app traffic-analysis:latest python3 -m src.gui.main_window
```

### 3. 运行测试

```bash
# 测试自适应违规检测
docker run --rm -v $(pwd):/app traffic-analysis:latest python3 tests/test_adaptive_violation.py

# 测试碰撞风险预测
docker run --rm -v $(pwd):/app traffic-analysis:latest python3 tests/test_stgat_collision.py

# 碰撞风险演示
docker run --rm -v $(pwd):/app traffic-analysis:latest python3 scripts/demo_collision.py
```

---

## 功能介绍

### 核心功能

| 功能 | 描述 |
|------|------|
| 车辆检测 | YOLOv12 实时检测多种车型 |
| 多目标跟踪 | ByteTrack 跨帧关联，保持轨迹连续 |
| 碰撞风险预测 | LSTM轨迹预测 + TTC分析 |
| 违规检测 | 闯红灯、超速检测 |
| 智能异常识别 | 识别特种车辆、交警，标记异常情况 |

### 创新点

#### 1. 碰撞风险预测系统
- **LSTM 轨迹预测**: 基于历史轨迹预测未来 15 帧位置
- **TTC 计算**: Time-To-Collision 碰撞时间计算
- **跟车距离检测**: 检测同向行驶车辆的安全距离
- **风险等级可视化**:

| 框颜色 | 风险等级 | 含义 |
|--------|----------|------|
| 绿色 | SAFE | 安全 |
| 黄色 | LOW | 低风险（跟车距离偏近）|
| 橙色 | MEDIUM | 中风险 |
| 红色 | HIGH | 高风险 |
| 紫色 | CRITICAL | 危急（即将碰撞）|

#### 2. 自适应违规检测
智能识别特殊情况，避免误判：

| 情况 | 处理方式 | 截图保存位置 |
|------|----------|-------------|
| 普通违规 | 正常记录 | `snapshots/violations/` |
| 附近有特种车辆 | 标记为异常 | `snapshots/anomaly/` |
| 附近有交警指挥 | 标记为异常 | `snapshots/anomaly/` |
| 信号灯故障 | 标记为异常 | `snapshots/anomaly/` |

支持检测的特种车辆：救护车、消防车、警车、工程救险车

#### 3. ST-GAT 车辆交互建模
- 时空图注意力网络 (Spatio-Temporal Graph Attention Network)
- 学习车辆之间的空间关系
- 提升碰撞风险预测准确性

---

## 项目结构

```
yolo_ls/
├── src/                          # 源代码
│   ├── core/                     # 核心模块
│   │   ├── detector.py           # YOLOv12 车辆检测
│   │   ├── tracker.py            # ByteTrack 多目标跟踪
│   │   ├── collision_risk.py     # 碰撞风险预测 (LSTM + TTC)
│   │   ├── adaptive_violation.py # 自适应违规检测
│   │   ├── emergency_vehicle.py  # 特种车辆检测
│   │   ├── stgat.py              # ST-GAT 时空图注意力网络
│   │   ├── feature.py            # 特征提取（颜色、速度、方向）
│   │   └── violation.py          # 基础违规检测
│   ├── video/                    # 视频处理
│   │   ├── stream.py             # 视频流接入（RTSP/文件）
│   │   └── preprocessor.py       # 帧预处理
│   ├── ocr/                      # 车牌识别
│   │   └── plate_reader.py       # OCR 车牌读取
│   ├── database/                 # 数据存储
│   │   └── db.py                 # SQLite 数据库
│   ├── gui/                      # 图形界面
│   │   ├── main_window.py        # PyQt5 主窗口
│   │   └── widgets/              # UI 组件
│   └── utils/                    # 工具函数
│       └── config.py             # 配置管理
│
├── scripts/                      # 工具脚本
│   ├── batch_detect.py           # 批量检测（含碰撞风险）
│   ├── demo_collision.py         # 碰撞风险演示
│   ├── demo_yielding.py          # 避让特种车辆演示
│   ├── train.py                  # 模型训练
│   ├── prepare_dataset.py        # 数据集准备
│   ├── compare_models.py         # 模型对比
│   └── visualize.py              # 可视化工具
│
├── tests/                        # 测试文件
│   ├── test_adaptive_violation.py # 自适应违规检测测试
│   └── test_stgat_collision.py   # ST-GAT 和碰撞预测测试
│
├── models/                       # 模型权重
│   └── yolo12n.pt                # YOLOv12 预训练模型
│
├── data/                         # 数据目录
│   ├── snapshots/                # 违规截图
│   │   ├── violations/           # 正常违规
│   │   └── anomaly/              # 异常情况（待人工复核）
│   ├── test_videos/              # 测试视频
│   └── detection_results/        # 检测结果
│
├── config/                       # 配置文件
│   └── settings.yaml             # 系统配置
│
├── Dockerfile                    # Docker 配置
├── docker-compose.yml            # Docker Compose
├── requirements.txt              # Python 依赖
├── main.py                       # 程序入口
└── README.md                     # 项目说明
```

---

## 核心模块详解

### 1. 车辆检测 (`src/core/detector.py`)
```python
from src.core import VehicleDetector

detector = VehicleDetector(model_path='models/yolo12n.pt', confidence=0.5)
detections = detector.detect_vehicles(frame)
# 返回: [Detection(bbox, confidence, class_name), ...]
```

### 2. 多目标跟踪 (`src/core/tracker.py`)
```python
from src.core import ByteTracker

tracker = ByteTracker(track_thresh=0.5, track_buffer=30)
tracks = tracker.update(detections)
# 返回: [Track(track_id, bbox, class_name), ...]
```

### 3. 碰撞风险预测 (`src/core/collision_risk.py`)
```python
from src.core.collision_risk import CollisionRiskPredictor

predictor = CollisionRiskPredictor(fps=30.0)
risks = predictor.update(tracks)
# 返回: [CollisionRisk(vehicle1_id, vehicle2_id, risk_level, ttc), ...]
```

### 4. 自适应违规检测 (`src/core/adaptive_violation.py`)
```python
from src.core.adaptive_violation import AdaptiveViolationDetector

detector = AdaptiveViolationDetector(speed_limit=60)
detector.set_stop_line(y=400, x_start=100, x_end=500)
record = detector.check_violation(track_id, bbox, speed, frame)
# 返回: ViolationRecord(violation_type, is_anomaly, anomaly_reason, ...)
```

### 5. 特种车辆检测 (`src/core/emergency_vehicle.py`)
```python
from src.core.emergency_vehicle import EmergencyVehicleDetector

ev_detector = EmergencyVehicleDetector()
emergency_vehicles = ev_detector.detect(frame, vehicle_bboxes)
# 返回: [EmergencyVehicle(vehicle_type, bbox, has_warning_light), ...]
```

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 深度学习 | PyTorch, YOLOv12 (Ultralytics) |
| 视频处理 | OpenCV |
| 目标跟踪 | ByteTrack |
| 轨迹预测 | LSTM |
| 图神经网络 | ST-GAT (PyTorch) |
| GUI | PyQt5 |
| 数据库 | SQLite |
| 容器化 | Docker |

---

## 性能指标

| 指标 | 目标值 |
|------|--------|
| 实时处理帧率 | ≥10 fps |
| 车辆检测准确率 | ≥85% |
| 碰撞预测提前时间 | 1-3 秒 |
| 连续运行稳定性 | 72小时无崩溃 |

---

## 作者

GitHub: https://github.com/Zhye26/yolo_ls
