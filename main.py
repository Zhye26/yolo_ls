#!/usr/bin/env python3
"""
实时交通分析系统 - 主程序入口

基于 YOLOv12 + ByteTrack 的智能交通监控系统
功能：车辆检测、跟踪、特征提取、违规识别、数据可视化
"""
import json
import os
import sys
from collections import Counter

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


def _serialize_risks(collision_risks):
    """序列化碰撞风险为JSON可写结构"""
    serialized = []
    for risk in collision_risks:
        serialized.append({
            'vehicle1_id': int(risk.vehicle1_id),
            'vehicle2_id': int(risk.vehicle2_id),
            'risk_level': risk.risk_level.value,
            'time_to_collision': float(risk.time_to_collision),
            'confidence': float(risk.confidence),
            'collision_point': (
                [float(risk.collision_point[0]), float(risk.collision_point[1])]
                if risk.collision_point
                else None
            ),
        })
    return serialized


def run_cli(args):
    """运行命令行模式（使用自适应违规检测）"""
    import cv2
    from src.video import VideoStream
    from src.core import (
        VehicleDetector,
        ByteTracker,
        FeatureExtractor,
        AdaptiveViolationDetector,
        VehicleInteractionGraph,
        CollisionRiskPredictor,
        RiskLevel,
    )
    from src.database import Database
    from src.ocr import PlateReader
    from src.utils import load_config

    # 加载配置
    config = load_config(args.config)
    risk_cfg = config.get('risk', {})

    if args.enable_risk:
        risk_enabled = True
    elif args.disable_risk:
        risk_enabled = False
    else:
        risk_enabled = risk_cfg.get('enabled', True)

    collision_model_path = args.collision_model or risk_cfg.get('collision_model_path')
    stgat_model_path = args.stgat_model or risk_cfg.get('stgat_model_path')

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

    interaction_graph = None
    collision_predictor = None
    if risk_enabled:
        interaction_graph = VehicleInteractionGraph(
            distance_threshold=risk_cfg.get('interaction_distance', 200),
            temporal_window=risk_cfg.get('temporal_window', 10),
            model_path=stgat_model_path,
            device=args.device,
        )
        collision_predictor = CollisionRiskPredictor(
            history_length=risk_cfg.get('history_length', 10),
            prediction_horizon=risk_cfg.get('prediction_horizon', 15),
            fps=config.get('video', {}).get('fps', 15),
            collision_threshold=risk_cfg.get('collision_threshold', 150.0),
            ttc_thresholds=risk_cfg.get('ttc_thresholds'),
            model_path=collision_model_path,
            device=args.device,
        )

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

    result_writer = None
    if args.risk_output:
        result_path = Path(args.risk_output)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_writer = result_path.open('w', encoding='utf-8')

    print(f"Processing video: {args.source}")
    print(f"Model: {args.model}, Device: {args.device}")
    if writer:
        print(f"Output video: {args.output}")
    if result_writer:
        print(f"Output structured results: {args.risk_output}")
    print("自适应违规检测已启用（支持特种车辆避让免责）")
    print(f"碰撞风险检测: {'启用' if risk_enabled else '关闭'}")
    print("Press 'q' to quit")

    frame_count = 0
    plate_cache = {}
    seen_tracks = set()
    risk_level_counter = Counter()
    max_active_risks = 0
    video_fps = video.get_fps()
    if video_fps <= 0:
        video_fps = config.get('video', {}).get('fps', 15)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1

        # 检测与跟踪
        detections = detector.detect_vehicles(frame)
        tracks = tracker.update(detections)
        track_data = [{'track_id': t.track_id, 'bbox': t.bbox} for t in tracks]

        interaction_embeddings = {}
        collision_risks = []
        risk_summary = {'total_risks': 0, 'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'min_ttc': -1}
        if risk_enabled and interaction_graph and collision_predictor:
            interaction_embeddings = interaction_graph.update(track_data)
            collision_risks = collision_predictor.update(track_data)
            risk_summary = collision_predictor.get_risk_summary(collision_risks)
            for risk in collision_risks:
                risk_level_counter[risk.risk_level.value] += 1
            max_active_risks = max(max_active_risks, len(collision_risks))

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

        frame_tracks = []
        frame_violations = []

        # 特征提取、车牌识别、违规检测
        for track in tracks:
            plate_number = plate_cache.get(track.track_id)
            if frame_count % 10 == 0:
                plate_result = plate_reader.read(frame, track.bbox)
                if plate_result:
                    plate_number = plate_result.plate_number
                    plate_cache[track.track_id] = plate_number

            features = feature_extractor.extract(frame, track.track_id, track.bbox)
            frame_tracks.append({
                'track_id': int(track.track_id),
                'class_name': track.class_name,
                'bbox': [int(v) for v in track.bbox],
                'speed_kmh': float(features.speed),
                'direction': features.direction.value,
                'color': features.color,
                'plate_number': plate_number,
            })

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
                frame_violations.append({
                    'record_id': record.record_id,
                    'track_id': int(record.track_id),
                    'violation_type': record.violation_type.value,
                    'is_anomaly': bool(record.is_anomaly),
                    'anomaly_reason': record.anomaly_reason.value if record.is_anomaly else None,
                    'speed_kmh': float(record.speed) if record.speed is not None else None,
                    'location': [int(record.location[0]), int(record.location[1])],
                    'plate_number': record.plate_number,
                    'snapshot_path': record.snapshot_path,
                })
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

        if frame_count % 30 == 0 and frame_tracks:
            avg_speed = sum(item['speed_kmh'] for item in frame_tracks) / len(frame_tracks)
            direction_counter = Counter(item['direction'] for item in frame_tracks)
            dominant_direction = direction_counter.most_common(1)[0][0]
            database.add_traffic_flow(
                vehicle_count=len(frame_tracks),
                avg_speed=avg_speed,
                direction=dominant_direction,
            )

        # 绘制标注
        annotated = violation_detector.draw_annotations(frame)
        if risk_enabled and collision_predictor and collision_risks:
            annotated = collision_predictor.draw_predictions(annotated, collision_risks, track_data)

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

        if risk_enabled:
            cv2.putText(
                annotated,
                f"Risk C/H/M/L: {risk_summary.get('critical', 0)}/{risk_summary.get('high', 0)}/{risk_summary.get('medium', 0)}/{risk_summary.get('low', 0)}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 200, 0),
                2,
            )

        if writer:
            writer.write(annotated)

        if result_writer:
            payload = {
                'frame_index': frame_count,
                'timestamp_sec': round(frame_count / max(video_fps, 1e-6), 3),
                'vehicle_count': len(frame_tracks),
                'violation_count': len(frame_violations),
                'interaction_embeddings': len(interaction_embeddings),
                'tracks': frame_tracks,
                'violations': frame_violations,
                'collision_risks': _serialize_risks(collision_risks),
                'risk_summary': {
                    'total_risks': int(risk_summary.get('total_risks', 0)),
                    'critical': int(risk_summary.get('critical', 0)),
                    'high': int(risk_summary.get('high', 0)),
                    'medium': int(risk_summary.get('medium', 0)),
                    'low': int(risk_summary.get('low', 0)),
                    'min_ttc': float(risk_summary.get('min_ttc', -1)),
                    'highest_risk_pair': risk_summary.get('highest_risk_pair'),
                },
            }
            result_writer.write(json.dumps(payload, ensure_ascii=False) + '\n')

        if not args.headless:
            cv2.imshow("Traffic Analysis - Adaptive Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video.release()
    if writer:
        writer.release()
    if result_writer:
        result_writer.close()
    cv2.destroyAllWindows()

    final_stats = violation_detector.get_statistics()
    print(f"\n=== 处理完成 ===")
    print(f"处理帧数: {frame_count}")
    print(f"总违规数: {final_stats['total_violations']}")
    print(f"实际违规: {final_stats['actual_violations']}")
    print(f"特殊情况(免责): {final_stats['exempted_count']}")
    if risk_enabled:
        print(
            "累计风险事件(C/H/M/L): "
            f"{risk_level_counter[RiskLevel.CRITICAL.value]}/"
            f"{risk_level_counter[RiskLevel.HIGH.value]}/"
            f"{risk_level_counter[RiskLevel.MEDIUM.value]}/"
            f"{risk_level_counter[RiskLevel.LOW.value]}"
        )
        print(f"峰值并发风险数: {max_active_risks}")

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
        '--risk-output', type=str, default=None,
        help='输出逐帧结构化结果(JSONL)路径'
    )
    parser.add_argument(
        '--collision-model', type=str, default=None,
        help='碰撞预测模型权重路径(.pt)'
    )
    parser.add_argument(
        '--stgat-model', type=str, default=None,
        help='ST-GAT模型权重路径(.pt)'
    )
    risk_group = parser.add_mutually_exclusive_group()
    risk_group.add_argument(
        '--enable-risk', action='store_true',
        help='强制启用碰撞风险检测'
    )
    risk_group.add_argument(
        '--disable-risk', action='store_true',
        help='强制关闭碰撞风险检测'
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
