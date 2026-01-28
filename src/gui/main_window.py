"""
PyQt5 Main Window - Real-Time Traffic Analysis System
Supports adaptive violation detection, ST-GAT interaction modeling,
and collision risk prediction.
"""
import cv2
import numpy as np
from collections import deque
from datetime import datetime
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QTableWidget,
    QTableWidgetItem, QTabWidget, QGroupBox, QLineEdit,
    QComboBox, QSpinBox, QStatusBar, QSplitter, QMessageBox,
    QCheckBox, QProgressBar, QFrame
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont
from typing import Optional, List, Dict
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.video import VideoStream
from src.core import VehicleDetector, ByteTracker, FeatureExtractor
from src.core.adaptive_violation import AdaptiveViolationDetector, ViolationRecord, ExemptionReason
from src.core.stgat import VehicleInteractionGraph
from src.core.collision_risk import CollisionRiskPredictor, RiskLevel
from src.ocr import PlateReader
from src.database import Database


class StatisticsCanvas(FigureCanvas):
    """Matplotlib canvas for statistics charts"""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor='#1a1a2e')
        super().__init__(self.fig)
        self.setParent(parent)

        # Data storage
        self.time_data = deque(maxlen=60)
        self.vehicle_count_data = deque(maxlen=60)
        self.speed_data = deque(maxlen=100)
        self.violation_counts: Dict[str, int] = {}
        self.vehicle_type_counts: Dict[str, int] = {}

        # Create subplots
        self.ax_flow = self.fig.add_subplot(2, 2, 1)
        self.ax_violation = self.fig.add_subplot(2, 2, 2)
        self.ax_speed = self.fig.add_subplot(2, 2, 3)
        self.ax_type = self.fig.add_subplot(2, 2, 4)

        self._setup_style()
        self.fig.tight_layout(pad=2.0)

    def _setup_style(self):
        """Setup chart style"""
        for ax in [self.ax_flow, self.ax_violation, self.ax_speed, self.ax_type]:
            ax.set_facecolor('#2d2d44')
            ax.tick_params(colors='white', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#4a4a6a')
            ax.title.set_color('white')

        self.ax_flow.set_title('Traffic Flow', fontsize=10)
        self.ax_violation.set_title('Violation Types', fontsize=10)
        self.ax_speed.set_title('Speed Distribution', fontsize=10)
        self.ax_type.set_title('Vehicle Types', fontsize=10)

    def update_data(self, vehicle_count: int, speeds: List[float],
                    violations: Dict[str, int], vehicle_types: Dict[str, int]):
        """Update chart data"""
        self.time_data.append(datetime.now().strftime('%H:%M:%S'))
        self.vehicle_count_data.append(vehicle_count)

        for speed in speeds:
            if speed > 0:
                self.speed_data.append(speed)

        for vtype, count in violations.items():
            self.violation_counts[vtype] = self.violation_counts.get(vtype, 0) + count

        for vtype, count in vehicle_types.items():
            self.vehicle_type_counts[vtype] = self.vehicle_type_counts.get(vtype, 0) + count

        self._redraw()

    def _redraw(self):
        """Redraw all charts"""
        # Traffic flow
        self.ax_flow.clear()
        self._setup_ax(self.ax_flow, 'Traffic Flow')
        if self.vehicle_count_data:
            x = list(range(len(self.vehicle_count_data)))
            self.ax_flow.plot(x, list(self.vehicle_count_data), color='#00ff88', linewidth=2)
            self.ax_flow.fill_between(x, list(self.vehicle_count_data), alpha=0.3, color='#00ff88')
            self.ax_flow.set_ylabel('Count', fontsize=8, color='white')

        # Violation pie
        self.ax_violation.clear()
        self._setup_ax(self.ax_violation, 'Violation Types')
        if self.violation_counts:
            labels = list(self.violation_counts.keys())
            sizes = list(self.violation_counts.values())
            colors = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff'][:len(labels)]
            self.ax_violation.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                                  textprops={'color': 'white', 'fontsize': 7})
        else:
            self.ax_violation.text(0.5, 0.5, 'No Data', ha='center', va='center', color='gray', fontsize=10)

        # Speed histogram
        self.ax_speed.clear()
        self._setup_ax(self.ax_speed, 'Speed Distribution')
        if self.speed_data:
            self.ax_speed.hist(list(self.speed_data), bins=15, color='#4d96ff', alpha=0.7, edgecolor='white')
            self.ax_speed.axvline(x=60, color='#ff6b6b', linestyle='--', linewidth=1.5)
            self.ax_speed.set_xlabel('km/h', fontsize=8, color='white')
        else:
            self.ax_speed.text(0.5, 0.5, 'No Data', ha='center', va='center', color='gray', fontsize=10)

        # Vehicle type bar
        self.ax_type.clear()
        self._setup_ax(self.ax_type, 'Vehicle Types')
        if self.vehicle_type_counts:
            types = list(self.vehicle_type_counts.keys())
            counts = list(self.vehicle_type_counts.values())
            colors = ['#6bcb77', '#4d96ff', '#ffd93d', '#ff6b6b'][:len(types)]
            self.ax_type.bar(types, counts, color=colors)
            self.ax_type.tick_params(axis='x', labelrotation=15)
        else:
            self.ax_type.text(0.5, 0.5, 'No Data', ha='center', va='center', color='gray', fontsize=10)

        self.fig.tight_layout(pad=2.0)
        self.draw()

    def _setup_ax(self, ax, title: str):
        """Setup axis style"""
        ax.set_facecolor('#2d2d44')
        ax.tick_params(colors='white', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#4a4a6a')
        ax.set_title(title, fontsize=10, color='white')

    def reset(self):
        """Reset all data"""
        self.time_data.clear()
        self.vehicle_count_data.clear()
        self.speed_data.clear()
        self.violation_counts.clear()
        self.vehicle_type_counts.clear()
        self._redraw()


class VideoThread(QThread):
    """Video processing thread"""
    frame_ready = pyqtSignal(np.ndarray, list, list, list, list, dict)  # Added plate_results
    stats_updated = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, source: str, config: dict):
        super().__init__()
        self.source = source
        self.config = config
        self.running = False

        self.video_stream: Optional[VideoStream] = None
        self.detector: Optional[VehicleDetector] = None
        self.tracker: Optional[ByteTracker] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.violation_detector: Optional[AdaptiveViolationDetector] = None
        self.interaction_graph: Optional[VehicleInteractionGraph] = None
        self.collision_predictor: Optional[CollisionRiskPredictor] = None
        self.plate_reader: Optional[PlateReader] = None

    def run(self):
        """Run video processing"""
        try:
            self.video_stream = VideoStream(
                self.source,
                fps=self.config.get('fps', 15)
            )
            if not self.video_stream.open():
                self.error.emit("Cannot open video source")
                return

            self.detector = VehicleDetector(
                model_path=self.config.get('model_path', 'yolo12n.pt'),
                confidence=self.config.get('confidence', 0.5),
                device=self.config.get('device', 'cpu')
            )

            self.tracker = ByteTracker(
                track_thresh=self.config.get('track_thresh', 0.5),
                track_buffer=self.config.get('track_buffer', 30)
            )

            self.feature_extractor = FeatureExtractor(
                pixel_to_meter=self.config.get('pixel_to_meter', 0.05),
                fps=self.config.get('fps', 15)
            )

            self.violation_detector = AdaptiveViolationDetector(
                speed_limit=self.config.get('speed_limit', 60),
                snapshot_dir=self.config.get('snapshot_dir', 'data/snapshots'),
                emergency_distance=self.config.get('emergency_distance', 300)
            )

            # Initialize ST-GAT and Collision Risk Predictor
            self.interaction_graph = VehicleInteractionGraph(
                distance_threshold=self.config.get('interaction_distance', 200)
            )
            self.collision_predictor = CollisionRiskPredictor(
                fps=self.config.get('fps', 15)
            )

            # Initialize Plate Reader
            self.plate_reader = PlateReader(
                model_path=self.config.get('plate_model_path', 'models/plate_ocr.pt'),
                use_gpu=self.config.get('device', 'cpu') != 'cpu'
            )

            if self.config.get('stop_line'):
                sl = self.config['stop_line']
                self.violation_detector.set_stop_line(sl['y'], sl['x_start'], sl['x_end'])

            self.running = True
            frame_count = 0

            for frame in self.video_stream.frames():
                if not self.running:
                    break

                frame_count += 1

                detections = self.detector.detect_vehicles(frame)
                tracks = self.tracker.update(detections)
                vehicle_bboxes = [t.bbox for t in tracks]
                self.violation_detector.update(frame, vehicle_bboxes)

                # Prepare track data for ST-GAT and collision prediction
                track_data = [
                    {'track_id': t.track_id, 'bbox': t.bbox}
                    for t in tracks
                ]

                # Update interaction graph
                interaction_embeddings = self.interaction_graph.update(track_data)

                # Predict collision risks
                collision_risks = self.collision_predictor.update(track_data)

                features_list = []
                violations = []
                plate_results = {}  # track_id -> plate_number

                for track in tracks:
                    features = self.feature_extractor.extract(
                        frame, track.track_id, track.bbox
                    )
                    features_list.append(features)

                    record = self.violation_detector.check_violation(
                        track_id=track.track_id,
                        bbox=track.bbox,
                        speed=features.speed,
                        frame=frame
                    )
                    if record:
                        violations.append(record)

                    # 车牌识别（每10帧识别一次以减少计算量）
                    if frame_count % 10 == 0 and self.plate_reader:
                        plate_result = self.plate_reader.read(frame, track.bbox)
                        if plate_result:
                            plate_results[track.track_id] = plate_result.plate_number

                # Draw annotations including collision risks
                annotated_frame = self.violation_detector.draw_annotations(frame)
                if collision_risks:
                    annotated_frame = self.collision_predictor.draw_predictions(
                        annotated_frame, collision_risks, track_data
                    )

                self.frame_ready.emit(annotated_frame, tracks, features_list, violations, collision_risks, plate_results)

                if frame_count % 30 == 0:
                    stats = self.violation_detector.get_statistics()
                    stats['emergency_vehicles'] = len(self.violation_detector.current_emergency_vehicles)
                    # Add collision risk stats
                    risk_summary = self.collision_predictor.get_risk_summary(collision_risks)
                    stats['collision_risks'] = risk_summary
                    self.stats_updated.emit(stats)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")
        finally:
            if self.video_stream:
                self.video_stream.release()

    def stop(self):
        """Stop processing"""
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    """Main Window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Traffic Analysis - Adaptive Violation Detection")
        self.setGeometry(100, 100, 1500, 950)

        self.video_thread: Optional[VideoThread] = None
        self.database = Database()
        self.current_frame: Optional[np.ndarray] = None

        self.config = {
            'fps': 15,
            'model_path': 'models/yolo12n_vehicle.pt',
            'plate_model_path': 'models/plate_ocr.pt',
            'confidence': 0.5,
            'device': 'cuda',
            'track_thresh': 0.5,
            'track_buffer': 30,
            'pixel_to_meter': 0.05,
            'speed_limit': 60,
            'snapshot_dir': 'data/snapshots',
            'emergency_distance': 300,
            'stop_line': None
        }

        self._init_ui()

    def _init_ui(self):
        """Initialize UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel: Video display
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(900, 650)
        self.video_label.setStyleSheet("background-color: #1a1a2e; border-radius: 8px;")
        self.video_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.video_label)

        # Control buttons
        btn_layout = QHBoxLayout()
        self.btn_open = QPushButton("Open Video")
        self.btn_camera = QPushButton("Open Camera")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        for btn in [self.btn_open, self.btn_camera, self.btn_stop]:
            btn.setMinimumHeight(35)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #4a4a6a;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background-color: #5a5a7a;
                }
                QPushButton:disabled {
                    background-color: #3a3a4a;
                    color: #888;
                }
            """)

        self.btn_open.clicked.connect(self._open_video)
        self.btn_camera.clicked.connect(self._open_camera)
        self.btn_stop.clicked.connect(self._stop_video)

        btn_layout.addWidget(self.btn_open)
        btn_layout.addWidget(self.btn_camera)
        btn_layout.addWidget(self.btn_stop)
        left_layout.addLayout(btn_layout)

        # Right panel: Info tabs
        right_panel = QTabWidget()
        right_panel.setMaximumWidth(550)

        # ===== Real-time Info Tab =====
        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)

        stats_group = QGroupBox("Real-time Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self.label_vehicle_count = QLabel("Vehicles: 0")
        self.label_emergency_count = QLabel("Emergency Vehicles: 0")
        self.label_avg_speed = QLabel("Avg Speed: 0 km/h")

        violation_frame = QFrame()
        violation_frame.setStyleSheet("background-color: #2d2d44; border-radius: 5px; padding: 5px;")
        vf_layout = QVBoxLayout(violation_frame)
        self.label_total_violations = QLabel("Total Violations: 0")
        self.label_actual_violations = QLabel("Actual Violations: 0")
        self.label_actual_violations.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        self.label_exempted = QLabel("Exempted (Special Cases): 0")
        self.label_exempted.setStyleSheet("color: #ffd93d; font-weight: bold;")
        vf_layout.addWidget(self.label_total_violations)
        vf_layout.addWidget(self.label_actual_violations)
        vf_layout.addWidget(self.label_exempted)

        stats_layout.addWidget(self.label_vehicle_count)
        stats_layout.addWidget(self.label_emergency_count)
        stats_layout.addWidget(self.label_avg_speed)
        stats_layout.addWidget(violation_frame)

        # Collision Risk Display
        risk_frame = QFrame()
        risk_frame.setStyleSheet("background-color: #2d3d44; border-radius: 5px; padding: 5px;")
        rf_layout = QVBoxLayout(risk_frame)
        self.label_collision_risk = QLabel("Collision Risk: SAFE")
        self.label_collision_risk.setStyleSheet("color: #00ff00; font-weight: bold;")
        self.label_min_ttc = QLabel("Min TTC: --")
        self.label_risk_count = QLabel("Active Risks: 0")
        rf_layout.addWidget(self.label_collision_risk)
        rf_layout.addWidget(self.label_min_ttc)
        rf_layout.addWidget(self.label_risk_count)
        stats_layout.addWidget(risk_frame)

        info_layout.addWidget(stats_group)

        # Vehicle list
        vehicle_group = QGroupBox("Detected Vehicles")
        vehicle_layout = QVBoxLayout(vehicle_group)
        self.vehicle_table = QTableWidget()
        self.vehicle_table.setColumnCount(6)
        self.vehicle_table.setHorizontalHeaderLabels(
            ["ID", "Type", "Color", "Speed(km/h)", "Direction", "Plate"]
        )
        self.vehicle_table.horizontalHeader().setStretchLastSection(True)
        vehicle_layout.addWidget(self.vehicle_table)
        info_layout.addWidget(vehicle_group)

        right_panel.addTab(info_tab, "Real-time Info")

        # ===== Violation Records Tab =====
        violation_tab = QWidget()
        violation_layout = QVBoxLayout(violation_tab)

        filter_layout = QHBoxLayout()
        self.cb_show_exempted = QCheckBox("Show Exempted")
        self.cb_show_exempted.setChecked(True)
        self.cb_only_exempted = QCheckBox("Only Exempted")
        filter_layout.addWidget(self.cb_show_exempted)
        filter_layout.addWidget(self.cb_only_exempted)
        filter_layout.addStretch()
        violation_layout.addLayout(filter_layout)

        self.violation_table = QTableWidget()
        self.violation_table.setColumnCount(7)
        self.violation_table.setHorizontalHeaderLabels(
            ["Time", "Type", "Plate", "Speed", "Status", "Reason", "Details"]
        )
        self.violation_table.horizontalHeader().setStretchLastSection(True)
        violation_layout.addWidget(self.violation_table)

        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter plate number...")
        btn_search = QPushButton("Search")
        btn_search.clicked.connect(self._search_plate)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(btn_search)
        violation_layout.addLayout(search_layout)

        right_panel.addTab(violation_tab, "Violations")

        # ===== Special Cases Info Tab =====
        exemption_tab = QWidget()
        exemption_layout = QVBoxLayout(exemption_tab)

        info_text = QLabel("""
<h3>Adaptive Violation Detection - Special Cases</h3>
<p>This system intelligently identifies special situations and marks them as exempted:</p>

<h4>1. Yielding to Emergency Vehicles</h4>
<p style="color: #ffd93d;">When ambulance, fire truck, or police car is detected nearby,
violations (running red light, lane crossing) are marked as "Yielding to Emergency".</p>

<h4>2. Traffic Light Malfunction</h4>
<p style="color: #ffd93d;">When traffic light shows abnormal status (no signal or irregular flashing),
related violations are marked as "Signal Malfunction".</p>

<h4>3. Other Special Cases</h4>
<ul>
<li>Police Direction</li>
<li>Emergency Avoidance</li>
<li>Road Construction Detour</li>
</ul>

<p><b>Note:</b> All special cases are recorded with snapshots.
Snapshot filenames include timestamp for later manual review.</p>
        """)
        info_text.setWordWrap(True)
        info_text.setStyleSheet("padding: 10px;")
        exemption_layout.addWidget(info_text)
        exemption_layout.addStretch()

        right_panel.addTab(exemption_tab, "Special Cases")

        # ===== Statistics Tab =====
        stats_tab = QWidget()
        stats_tab_layout = QVBoxLayout(stats_tab)
        self.stats_canvas = StatisticsCanvas(stats_tab)
        stats_tab_layout.addWidget(self.stats_canvas)

        btn_reset_stats = QPushButton("Reset Statistics")
        btn_reset_stats.clicked.connect(self._reset_statistics)
        btn_reset_stats.setStyleSheet("""
            QPushButton {
                background-color: #4a4a6a;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #5a5a7a;
            }
        """)
        stats_tab_layout.addWidget(btn_reset_stats)

        right_panel.addTab(stats_tab, "Statistics")

        # ===== Settings Tab =====
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)

        detect_group = QGroupBox("Detection Settings")
        detect_layout = QVBoxLayout(detect_group)

        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.spin_confidence = QSpinBox()
        self.spin_confidence.setRange(1, 100)
        self.spin_confidence.setValue(50)
        self.spin_confidence.setSuffix("%")
        conf_layout.addWidget(self.spin_confidence)
        detect_layout.addLayout(conf_layout)

        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed Limit:"))
        self.spin_speed_limit = QSpinBox()
        self.spin_speed_limit.setRange(1, 200)
        self.spin_speed_limit.setValue(60)
        self.spin_speed_limit.setSuffix(" km/h")
        speed_layout.addWidget(self.spin_speed_limit)
        detect_layout.addLayout(speed_layout)

        emergency_layout = QHBoxLayout()
        emergency_layout.addWidget(QLabel("Emergency Distance:"))
        self.spin_emergency_dist = QSpinBox()
        self.spin_emergency_dist.setRange(50, 1000)
        self.spin_emergency_dist.setValue(300)
        self.spin_emergency_dist.setSuffix(" px")
        emergency_layout.addWidget(self.spin_emergency_dist)
        detect_layout.addLayout(emergency_layout)

        settings_layout.addWidget(detect_group)

        stopline_group = QGroupBox("Stop Line Settings")
        stopline_layout = QVBoxLayout(stopline_group)

        self.cb_enable_stopline = QCheckBox("Enable Stop Line Detection")
        stopline_layout.addWidget(self.cb_enable_stopline)

        sl_y_layout = QHBoxLayout()
        sl_y_layout.addWidget(QLabel("Y Position:"))
        self.spin_sl_y = QSpinBox()
        self.spin_sl_y.setRange(0, 2000)
        self.spin_sl_y.setValue(400)
        sl_y_layout.addWidget(self.spin_sl_y)
        stopline_layout.addLayout(sl_y_layout)

        sl_x_layout = QHBoxLayout()
        sl_x_layout.addWidget(QLabel("X Range:"))
        self.spin_sl_x1 = QSpinBox()
        self.spin_sl_x1.setRange(0, 2000)
        self.spin_sl_x1.setValue(100)
        self.spin_sl_x2 = QSpinBox()
        self.spin_sl_x2.setRange(0, 2000)
        self.spin_sl_x2.setValue(500)
        sl_x_layout.addWidget(self.spin_sl_x1)
        sl_x_layout.addWidget(QLabel("-"))
        sl_x_layout.addWidget(self.spin_sl_x2)
        stopline_layout.addLayout(sl_x_layout)

        settings_layout.addWidget(stopline_group)
        settings_layout.addStretch()

        right_panel.addTab(settings_tab, "Settings")

        # Add to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        main_layout.addWidget(splitter)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready - Adaptive Violation Detection System")

    def _open_video(self):
        """Open video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if file_path:
            self._start_processing(file_path)

    def _open_camera(self):
        """Open camera"""
        self._start_processing("0")

    def _start_processing(self, source: str):
        """Start processing"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()

        self.config['confidence'] = self.spin_confidence.value() / 100
        self.config['speed_limit'] = self.spin_speed_limit.value()
        self.config['emergency_distance'] = self.spin_emergency_dist.value()

        if self.cb_enable_stopline.isChecked():
            self.config['stop_line'] = {
                'y': self.spin_sl_y.value(),
                'x_start': self.spin_sl_x1.value(),
                'x_end': self.spin_sl_x2.value()
            }
        else:
            self.config['stop_line'] = None

        self.video_thread = VideoThread(source, self.config)
        self.video_thread.frame_ready.connect(self._on_frame_ready)
        self.video_thread.stats_updated.connect(self._on_stats_updated)
        self.video_thread.error.connect(self._on_error)
        self.video_thread.start()

        self.btn_open.setEnabled(False)
        self.btn_camera.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.statusBar.showMessage("Processing...")

    def _stop_video(self):
        """Stop processing"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None

        self.btn_open.setEnabled(True)
        self.btn_camera.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.statusBar.showMessage("Stopped")

    def _on_frame_ready(self, frame: np.ndarray, tracks: list,
                        features_list: list, violations: list,
                        collision_risks: list = None, plate_results: dict = None):
        """Process frame"""
        self.current_frame = frame.copy()

        # 保存车牌识别结果
        if plate_results:
            if not hasattr(self, '_plate_cache'):
                self._plate_cache = {}
            self._plate_cache.update(plate_results)

        for i, track in enumerate(tracks):
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 构建标签
            if i < len(features_list):
                f = features_list[i]
                label = f"ID:{track.track_id} {f.color} {f.speed:.1f}km/h"
            else:
                label = f"ID:{track.track_id} {track.class_name}"

            # 添加车牌信息
            plate = getattr(self, '_plate_cache', {}).get(track.track_id)
            if plate:
                label += f" [{plate}]"
                # 在车辆下方显示车牌号
                cv2.putText(frame, plate, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self._display_frame(frame)
        self.label_vehicle_count.setText(f"Vehicles: {len(tracks)}")

        if features_list:
            avg_speed = sum(f.speed for f in features_list) / len(features_list)
            self.label_avg_speed.setText(f"Avg Speed: {avg_speed:.1f} km/h")

        # Update collision risk display
        if collision_risks:
            highest_risk = collision_risks[0] if collision_risks else None
            if highest_risk:
                risk_colors = {
                    RiskLevel.SAFE: "#00ff00",
                    RiskLevel.LOW: "#ffff00",
                    RiskLevel.MEDIUM: "#ffa500",
                    RiskLevel.HIGH: "#ff0000",
                    RiskLevel.CRITICAL: "#ff00ff"
                }
                color = risk_colors.get(highest_risk.risk_level, "#00ff00")
                self.label_collision_risk.setText(f"Collision Risk: {highest_risk.risk_level.value.upper()}")
                self.label_collision_risk.setStyleSheet(f"color: {color}; font-weight: bold;")

                if highest_risk.time_to_collision > 0:
                    self.label_min_ttc.setText(f"Min TTC: {highest_risk.time_to_collision:.1f}s")
                else:
                    self.label_min_ttc.setText("Min TTC: --")

                self.label_risk_count.setText(f"Active Risks: {len(collision_risks)}")
        else:
            self.label_collision_risk.setText("Collision Risk: SAFE")
            self.label_collision_risk.setStyleSheet("color: #00ff00; font-weight: bold;")
            self.label_min_ttc.setText("Min TTC: --")
            self.label_risk_count.setText("Active Risks: 0")

        self.vehicle_table.setRowCount(len(tracks))
        for i, track in enumerate(tracks):
            self.vehicle_table.setItem(i, 0, QTableWidgetItem(str(track.track_id)))
            self.vehicle_table.setItem(i, 1, QTableWidgetItem(track.class_name))

            if i < len(features_list):
                f = features_list[i]
                self.vehicle_table.setItem(i, 2, QTableWidgetItem(f.color))
                self.vehicle_table.setItem(i, 3, QTableWidgetItem(f"{f.speed:.1f}"))
                self.vehicle_table.setItem(i, 4, QTableWidgetItem(f.direction.value))
            else:
                self.vehicle_table.setItem(i, 2, QTableWidgetItem("-"))
                self.vehicle_table.setItem(i, 3, QTableWidgetItem("-"))
                self.vehicle_table.setItem(i, 4, QTableWidgetItem("-"))

            # 添加车牌信息
            plate = getattr(self, '_plate_cache', {}).get(track.track_id, "-")
            self.vehicle_table.setItem(i, 5, QTableWidgetItem(plate if plate else "-"))

        for record in violations:
            self._add_violation_record(record)

        # Update statistics charts (every 30 frames to reduce overhead)
        if hasattr(self, '_frame_counter'):
            self._frame_counter += 1
        else:
            self._frame_counter = 0

        if self._frame_counter % 30 == 0:
            speeds = [f.speed for f in features_list if f.speed > 0]
            violation_dict = {}
            for v in violations:
                vtype = v.violation_type.value
                violation_dict[vtype] = violation_dict.get(vtype, 0) + 1

            vehicle_types = {}
            for track in tracks:
                vtype = track.class_name
                vehicle_types[vtype] = vehicle_types.get(vtype, 0) + 1

            self.stats_canvas.update_data(
                vehicle_count=len(tracks),
                speeds=speeds,
                violations=violation_dict,
                vehicle_types=vehicle_types
            )

    def _on_stats_updated(self, stats: dict):
        """Update statistics"""
        self.label_emergency_count.setText(f"Emergency Vehicles: {stats.get('emergency_vehicles', 0)}")
        self.label_total_violations.setText(f"Total Violations: {stats.get('total_violations', 0)}")
        self.label_actual_violations.setText(f"Actual Violations: {stats.get('actual_violations', 0)}")
        self.label_exempted.setText(f"Exempted (Special Cases): {stats.get('exempted_count', 0)}")

        # Update collision risk stats
        risk_stats = stats.get('collision_risks', {})
        if risk_stats:
            total_risks = risk_stats.get('total_risks', 0)
            critical = risk_stats.get('critical', 0)
            high = risk_stats.get('high', 0)
            self.label_risk_count.setText(f"Active Risks: {total_risks} (Critical: {critical}, High: {high})")

    def _display_frame(self, frame: np.ndarray):
        """Display frame"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        scaled = qt_image.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(QPixmap.fromImage(scaled))

    def _add_violation_record(self, record: ViolationRecord):
        """Add violation record to table"""
        if self.cb_only_exempted.isChecked() and not record.is_exempted:
            return
        if not self.cb_show_exempted.isChecked() and record.is_exempted:
            return

        row = self.violation_table.rowCount()
        self.violation_table.insertRow(row)

        self.violation_table.setItem(
            row, 0, QTableWidgetItem(record.timestamp.strftime("%H:%M:%S"))
        )
        self.violation_table.setItem(
            row, 1, QTableWidgetItem(record.violation_type.value)
        )
        self.violation_table.setItem(
            row, 2, QTableWidgetItem(record.plate_number or "-")
        )

        speed_str = f"{record.speed:.1f}" if record.speed else "-"
        self.violation_table.setItem(row, 3, QTableWidgetItem(speed_str))

        status_item = QTableWidgetItem("Exempted" if record.is_exempted else "Violation")
        if record.is_exempted:
            status_item.setBackground(QColor("#ffd93d"))
            status_item.setForeground(QColor("#000"))
        else:
            status_item.setBackground(QColor("#ff6b6b"))
            status_item.setForeground(QColor("#fff"))
        self.violation_table.setItem(row, 4, status_item)

        reason = ""
        if record.is_exempted:
            reason_map = {
                ExemptionReason.YIELD_TO_EMERGENCY: "Yield to Emergency",
                ExemptionReason.POLICE_DIRECTION: "Police Direction",
                ExemptionReason.EMERGENCY_AVOIDANCE: "Emergency Avoidance",
                ExemptionReason.SIGNAL_MALFUNCTION: "Signal Malfunction",
                ExemptionReason.ROAD_CONSTRUCTION: "Road Construction",
                ExemptionReason.NONE: "",
            }
            reason = reason_map.get(record.exemption_reason, "")
        self.violation_table.setItem(row, 5, QTableWidgetItem(reason))

        details = record.exemption_details if record.is_exempted else f"Location: {record.location}"
        self.violation_table.setItem(row, 6, QTableWidgetItem(details))

        self.database.add_violation(
            track_id=record.track_id,
            violation_type=record.violation_type.value,
            location=record.location,
            speed=record.speed,
            plate_number=record.plate_number,
            snapshot_path=record.snapshot_path,
            record_id=record.record_id,
            is_exempted=record.is_exempted,
            exemption_reason=record.exemption_reason.value if record.is_exempted else None,
            exemption_details=record.exemption_details,
            nearby_emergency_vehicles=record.nearby_emergency_vehicles
        )

    def _search_plate(self):
        """Search plate"""
        plate = self.search_input.text().strip()
        if not plate:
            return

        results = self.database.search_by_plate(plate)
        if results:
            QMessageBox.information(
                self, "Search Results",
                f"Found {len(results)} records"
            )
        else:
            QMessageBox.information(self, "Search Results", "No matching records found")

    def _reset_statistics(self):
        """Reset statistics charts"""
        self.stats_canvas.reset()
        self._frame_counter = 0
        self.statusBar.showMessage("Statistics reset")

    def _on_error(self, error: str):
        """Handle error"""
        QMessageBox.critical(self, "Error", error)
        self._stop_video()

    def closeEvent(self, event):
        """Close event"""
        self._stop_video()
        event.accept()


def main():
    """Main function"""
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
