#!/usr/bin/env python3
"""
实时交通分析系统 - 主程序入口

基于 YOLOv12 + ByteTrack 的智能交通监控系统
功能：车辆检测、跟踪、特征提取、违规识别、数据可视化
"""
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def run_gui():
    """运行 GUI 模式"""
    from PyQt5.QtWidgets import QApplication
    from src.gui import MainWindow

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


def run_cli(args):
    """运行命令行模式"""
    import cv2
    from src.video import VideoStream
    from src.core import VehicleDetector, ByteTracker, FeatureExtractor, ViolationDetector
    from src.utils import load_config

    # 加载配置
    config = load_config(args.config)

    # 初始化组件
    video = VideoStream(
        args.source,
        fps=config.get('video', {}).get('fps', 15)
    )
    detector = VehicleDetector(
        model_path=args.model,
        confidence=args.confidence,
        device=args.device
    )
    tracker = ByteTracker(
        track_thresh=config.get('tracker', {}).get('track_thresh', 0.5),
        track_buffer=config.get('tracker', {}).get('track_buffer', 30)
    )
    feature_extractor = FeatureExtractor(
        pixel_to_meter=config.get('feature', {}).get('pixel_to_meter', 0.05),
        fps=config.get('video', {}).get('fps', 15)
    )
    violation_detector = ViolationDetector(
        speed_limit=config.get('violation', {}).get('speed_limit', 60)
    )

    if not video.open():
        print(f"Error: Cannot open video source: {args.source}")
        return 1

    print(f"Processing video: {args.source}")
    print(f"Model: {args.model}, Device: {args.device}")
    print("Press 'q' to quit")

    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1

        # 检测
        detections = detector.detect_vehicles(frame)

        # 跟踪
        tracks = tracker.update(detections)

        # 特征提取
        for track in tracks:
            features = feature_extractor.extract(frame, track.track_id, track.bbox)

            # 违规检测
            violations = violation_detector.check_violations(
                track.track_id, track.center, features.speed, frame
            )

            # 绘制
            x1, y1, x2, y2 = track.bbox
            color = (0, 255, 0)
            if violations:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track.track_id} {features.color} {features.speed:.1f}km/h"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 显示统计
        cv2.putText(frame, f"Vehicles: {len(tracks)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示
        if not args.headless:
            cv2.imshow("Traffic Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 保存输出
        if args.output:
            # TODO: 实现视频输出
            pass

    video.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="实时交通分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--gui', action='store_true',
        help='启动图形界面模式'
    )
    parser.add_argument(
        '--source', type=str, default='0',
        help='视频源（摄像头ID、RTSP地址或视频文件路径）'
    )
    parser.add_argument(
        '--model', type=str, default='yolov8n.pt',
        help='YOLO 模型路径'
    )
    parser.add_argument(
        '--confidence', type=float, default=0.5,
        help='检测置信度阈值'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'],
        help='运行设备'
    )
    parser.add_argument(
        '--config', type=str, default='config/settings.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='输出视频路径'
    )
    parser.add_argument(
        '--headless', action='store_true',
        help='无头模式（不显示窗口）'
    )

    args = parser.parse_args()

    if args.gui:
        run_gui()
    else:
        sys.exit(run_cli(args))


if __name__ == '__main__':
    main()
