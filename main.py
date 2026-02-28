#!/usr/bin/env python3
"""
实时交通分析系统 - 主程序入口

基于 YOLOv12 + ByteTrack 的智能交通监控系统
功能：车辆检测、跟踪、特征提取、违规识别、数据可视化
"""
import os
import sys

# 在导入任何Qt相关模块之前设置环境变量，避免OpenCV Qt插件冲突
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''  # 清除OpenCV的Qt路径
os.environ.pop('QT_PLUGIN_PATH', None)  # 移除可能的冲突路径

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
    """运行命令行模式（使用自适应违规检测）"""
    import cv2
    from src.video import VideoStream
    from src.core import VehicleDetector, ByteTracker, FeatureExtractor, AdaptiveViolationDetector
    from src.database import Database
    from src.ocr import PlateReader
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
    violation_detector = AdaptiveViolationDetector(
        speed_limit=config.get('violation', {}).get('speed_limit', 60),
        snapshot_dir=config.get('violation', {}).get('snapshot_dir', 'data/snapshots'),
        emergency_distance=config.get('violation', {}).get('emergency_distance', 300)
    )
    plate_reader = PlateReader(
        model_path=config.get('ocr', {}).get('model_path', 'models/plate_ocr.pt'),
        use_gpu=args.device != 'cpu'
    )
    database = Database()

    if not video.open():
        print(f"Error: Cannot open video source: {args.source}")
        return 1

    writer = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = video.get_frame_size()
        fps = video.get_fps()
        if fps <= 0:
            fps = config.get('video', {}).get('fps', 15)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"Warning: Cannot open output writer: {output_path}")
            writer = None

    print(f"Processing video: {args.source}")
    print(f"Model: {args.model}, Device: {args.device}")
    if writer:
        print(f"Output video: {args.output}")
    print("自适应违规检测已启用（支持特种车辆避让免责）")
    print("Press 'q' to quit")

    frame_count = 0
    plate_cache = {}
    seen_tracks = set()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1

        # 检测与跟踪
        detections = detector.detect_vehicles(frame)
        tracks = tracker.update(detections)

        # 交通灯和人员检测（尽力而为，兼容仅车辆模型）
        person_bboxes = []
        light_bbox = None
        try:
            persons = detector.detect_persons(frame)
            person_bboxes = [det.bbox for det in persons]
        except Exception:
            pass

        try:
            traffic_lights = detector.detect_traffic_lights(frame)
            if traffic_lights:
                light_bbox = max(traffic_lights, key=lambda d: d.confidence).bbox
        except Exception:
            pass

        # 更新违规检测上下文
        vehicle_bboxes = [t.bbox for t in tracks]
        violation_detector.update(
            frame,
            vehicle_bboxes,
            person_bboxes=person_bboxes,
            light_bbox=light_bbox
        )

        # 特征提取、车牌识别、违规检测
        for track in tracks:
            plate_number = plate_cache.get(track.track_id)
            if frame_count % 10 == 0:
                plate_result = plate_reader.read(frame, track.bbox)
                if plate_result:
                    plate_number = plate_result.plate_number
                    plate_cache[track.track_id] = plate_number

            features = feature_extractor.extract(frame, track.track_id, track.bbox)

            # 维护车辆记录，支持车牌检索
            if track.track_id in seen_tracks:
                database.update_vehicle(
                    track_id=track.track_id,
                    speed=features.speed,
                    direction=features.direction.value,
                    plate_number=plate_number,
                    vehicle_type=track.class_name,
                    color=features.color,
                )
            else:
                database.add_vehicle(
                    track_id=track.track_id,
                    plate_number=plate_number,
                    vehicle_type=track.class_name,
                    color=features.color,
                    speed=features.speed,
                    direction=features.direction.value,
                )
                seen_tracks.add(track.track_id)

            record = violation_detector.check_violation(
                track_id=track.track_id,
                bbox=track.bbox,
                speed=features.speed,
                frame=frame,
                plate_number=plate_number,
            )

            if record:
                database.add_violation(
                    track_id=record.track_id,
                    violation_type=record.violation_type.value,
                    location=record.location,
                    speed=record.speed,
                    plate_number=record.plate_number,
                    snapshot_path=record.snapshot_path,
                    record_id=record.record_id,
                    is_exempted=record.is_anomaly,
                    exemption_reason=record.anomaly_reason.value if record.is_anomaly else None,
                    exemption_details=", ".join(record.nearby_objects) if record.nearby_objects else None,
                    nearby_emergency_vehicles=record.nearby_objects,
                )

        # 绘制标注
        annotated = violation_detector.draw_annotations(frame)
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            plate = plate_cache.get(track.track_id)
            if plate:
                cv2.putText(
                    annotated,
                    plate,
                    (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

        stats = violation_detector.get_statistics()
        cv2.putText(
            annotated,
            f"Vehicles: {len(tracks)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            annotated,
            f"Violations: {stats['actual_violations']} | Exempted: {stats['exempted_count']}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        if writer:
            writer.write(annotated)

        if not args.headless:
            cv2.imshow("Traffic Analysis - Adaptive Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    final_stats = violation_detector.get_statistics()
    print(f"\n=== 处理完成 ===")
    print(f"处理帧数: {frame_count}")
    print(f"总违规数: {final_stats['total_violations']}")
    print(f"实际违规: {final_stats['actual_violations']}")
    print(f"特殊情况(免责): {final_stats['exempted_count']}")
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
        '--model', type=str, default='models/yolo12n_vehicle.pt',
        help='YOLOv12 model path'
    )
    parser.add_argument(
        '--confidence', type=float, default=0.2,
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
