# Real-Time Traffic Analysis System

基于 YOLOv12 + ByteTrack 的实时交通分析系统，实现车辆检测、跟踪、碰撞风险预测和违规识别。

## 系统要求

- Docker (必须)
- 显示器 (GUI模式需要)

## 快速开始

### 1. 构建Docker镜像

```bash
# Linux / macOS / Windows PowerShell
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
- `-n`: 抽帧间隔（每N帧保存一次，默认30，越小越详细）

输出文件:
- `frame_*_orig.jpg` - 原始帧
- `frame_*_detect.jpg` - 检测结果（有风险车辆框变色）
- `results.json` - 详细数据（检测、跟踪、碰撞风险）

#### 方式B: GUI界面（需要显示器）

Linux:
```bash
xhost +local:docker
docker run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/app \
  traffic-analysis:latest \
  python3 -m src.gui.main_window
```

Windows (需要安装VcXsrv或X410):
```powershell
# 先启动X Server (VcXsrv)，然后:
docker run --rm -e DISPLAY=host.docker.internal:0 -v ${PWD}:/app traffic-analysis:latest python3 -m src.gui.main_window
```

### 3. 运行测试

```bash
# 测试所有模块
docker run --rm -v $(pwd):/app traffic-analysis:latest python3 tests/test_stgat_collision.py

# 碰撞风险演示（生成演示视频）
docker run --rm -v $(pwd):/app traffic-analysis:latest python3 scripts/demo_collision.py
```

## 功能特点

### 核心功能
- 车辆检测 (YOLOv12)
- 多目标跟踪 (ByteTrack)
- 特征提取（颜色、速度、方向）
- 违规检测（闯红灯、超速）

### 创新点
1. **自适应违规检测** - 智能识别紧急车辆让行等特殊情况
2. **ST-GAT车辆交互建模** - 时空图注意力网络学习车辆关系
3. **碰撞风险预测** - LSTM轨迹预测 + TTC分析 + 跟车距离检测

## 碰撞风险颜色说明

| 框颜色 | 风险等级 | 含义 |
|--------|----------|------|
| 绿色 | SAFE | 安全 |
| 黄色 | LOW | 低风险（跟车距离偏近）|
| 橙色 | MEDIUM | 中风险 |
| 红色 | HIGH | 高风险 |
| 紫色 | CRITICAL | 危急（即将碰撞）|

## 项目结构

```
yolo_ls/
├── src/
│   ├── core/           # 核心模块（检测、跟踪、碰撞预测）
│   ├── gui/            # PyQt5界面
│   ├── video/          # 视频处理
│   └── database/       # 数据存储
├── scripts/            # 工具脚本
│   ├── batch_detect.py # 批量检测
│   └── demo_collision.py # 碰撞演示
├── models/             # 模型权重
├── data/               # 数据和结果
├── tests/              # 测试脚本
├── Dockerfile          # Docker配置
└── requirements.txt    # Python依赖
```

## 技术栈

- PyTorch, YOLOv12 (Ultralytics)
- OpenCV, ByteTrack
- PyQt5, SQLite

## 作者

GitHub: https://github.com/Zhye26/yolo_ls
