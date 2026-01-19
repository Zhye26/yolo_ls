"""PyQt5 主窗口"""
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QTableWidget,
    QTableWidgetItem, QTabWidget, QGroupBox, QLineEdit,
    QComboBox, QSpinBox, QStatusBar, QSplitter, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
from typing import Optional
import sys
sys.path.insert(0, str(__file__).rsplit('/src/', 1)[0])

from src.video import VideoStream
from src.core import VehicleDetector, ByteTracker, FeatureExtractor, ViolationDetector
from src.database import Database


class VideoThread(QThread):
    """视频处理线程"""
    frame_ready = pyqtSignal(np.ndarray, list, list)  # 帧, 跟踪结果, 违规
    error = pyqtSignal(str)

    def __init__(self, source: str, config: dict):
        super().__init__()
        self.source = source
        self.config = config
        self.running = False

        # 初始化组件
        self.video_stream: Optional[VideoStream] = None
        self.detector: Optional[VehicleDetector] = None
        self.tracker: Optional[ByteTracker] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.violation_detector: Optional[ViolationDetector] = None

    def run(self):
        """运行视频处理"""
        try:
            # 初始化
            self.video_stream = VideoStream(
                self.source,
                fps=self.config.get('fps', 15)
            )
            if not self.video_stream.open():
                self.error.emit("无法打开视频源")
                return

            self.detector = VehicleDetector(
                model_path=self.config.get('model_path', 'yolov8n.pt'),
                confidence=self.config.get('confidence', 0.5),
                device=self.config.get('device', 'cuda')
            )
            self.tracker = ByteTracker(
                track_thresh=self.config.get('track_thresh', 0.5),
                track_buffer=self.config.get('track_buffer', 30)
            )
            self.feature_extractor = FeatureExtractor(
                pixel_to_meter=self.config.get('pixel_to_meter', 0.05),
                fps=self.config.get('fps', 15)
            )
            self.violation_detector = ViolationDetector(
                speed_limit=self.config.get('speed_limit', 60)
            )

            self.running = True
            for frame in self.video_stream.frames():
                if not self.running:
                    break

                # 检测
                detections = self.detector.detect_vehicles(frame)

                # 跟踪
                tracks = self.tracker.update(detections)

                # 特征提取和违规检测
                violations = []
                for track in tracks:
                    features = self.feature_extractor.extract(
                        frame, track.track_id, track.bbox
                    )
                    v = self.violation_detector.check_violations(
                        track.track_id, track.center,
                        features.speed, frame
                    )
                    violations.extend(v)

                self.frame_ready.emit(frame, tracks, violations)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if self.video_stream:
                self.video_stream.release()

    def stop(self):
        """停止处理"""
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("实时交通分析系统")
        self.setGeometry(100, 100, 1400, 900)

        # 组件
        self.video_thread: Optional[VideoThread] = None
        self.database = Database()
        self.current_frame: Optional[np.ndarray] = None

        # 配置
        self.config = {
            'fps': 15,
            'model_path': 'yolov8n.pt',
            'confidence': 0.5,
            'device': 'cuda',
            'track_thresh': 0.5,
            'track_buffer': 30,
            'pixel_to_meter': 0.05,
            'speed_limit': 60
        }

        self._init_ui()

    def _init_ui(self):
        """初始化界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧：视频显示
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.video_label)

        # 控制按钮
        btn_layout = QHBoxLayout()
        self.btn_open = QPushButton("打开视频")
        self.btn_camera = QPushButton("打开摄像头")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)

        self.btn_open.clicked.connect(self._open_video)
        self.btn_camera.clicked.connect(self._open_camera)
        self.btn_stop.clicked.connect(self._stop_video)

        btn_layout.addWidget(self.btn_open)
        btn_layout.addWidget(self.btn_camera)
        btn_layout.addWidget(self.btn_stop)
        left_layout.addLayout(btn_layout)

        # 右侧：信息面板
        right_panel = QTabWidget()
        right_panel.setMaximumWidth(500)

        # 实时信息标签页
        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)

        # 统计信息
        stats_group = QGroupBox("实时统计")
        stats_layout = QVBoxLayout(stats_group)
        self.label_vehicle_count = QLabel("车辆数量: 0")
        self.label_avg_speed = QLabel("平均速度: 0 km/h")
        self.label_violation_count = QLabel("违规数量: 0")
        stats_layout.addWidget(self.label_vehicle_count)
        stats_layout.addWidget(self.label_avg_speed)
        stats_layout.addWidget(self.label_violation_count)
        info_layout.addWidget(stats_group)

        # 车辆列表
        vehicle_group = QGroupBox("检测车辆")
        vehicle_layout = QVBoxLayout(vehicle_group)
        self.vehicle_table = QTableWidget()
        self.vehicle_table.setColumnCount(5)
        self.vehicle_table.setHorizontalHeaderLabels(
            ["ID", "类型", "颜色", "速度", "方向"]
        )
        vehicle_layout.addWidget(self.vehicle_table)
        info_layout.addWidget(vehicle_group)

        right_panel.addTab(info_tab, "实时信息")

        # 违规记录标签页
        violation_tab = QWidget()
        violation_layout = QVBoxLayout(violation_tab)
        self.violation_table = QTableWidget()
        self.violation_table.setColumnCount(5)
        self.violation_table.setHorizontalHeaderLabels(
            ["时间", "类型", "车牌", "速度", "位置"]
        )
        violation_layout.addWidget(self.violation_table)

        # 搜索
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入车牌号搜索...")
        btn_search = QPushButton("搜索")
        btn_search.clicked.connect(self._search_plate)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(btn_search)
        violation_layout.addLayout(search_layout)

        right_panel.addTab(violation_tab, "违规记录")

        # 设置标签页
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)

        # 检测设置
        detect_group = QGroupBox("检测设置")
        detect_layout = QVBoxLayout(detect_group)

        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度阈值:"))
        self.spin_confidence = QSpinBox()
        self.spin_confidence.setRange(1, 100)
        self.spin_confidence.setValue(50)
        conf_layout.addWidget(self.spin_confidence)
        detect_layout.addLayout(conf_layout)

        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("速度限制 (km/h):"))
        self.spin_speed_limit = QSpinBox()
        self.spin_speed_limit.setRange(1, 200)
        self.spin_speed_limit.setValue(60)
        speed_layout.addWidget(self.spin_speed_limit)
        detect_layout.addLayout(speed_layout)

        settings_layout.addWidget(detect_group)
        settings_layout.addStretch()

        right_panel.addTab(settings_tab, "设置")

        # 添加到主布局
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        main_layout.addWidget(splitter)

        # 状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")

    def _open_video(self):
        """打开视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov);;所有文件 (*)"
        )
        if file_path:
            self._start_processing(file_path)

    def _open_camera(self):
        """打开摄像头"""
        self._start_processing("0")

    def _start_processing(self, source: str):
        """开始处理"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()

        self.config['confidence'] = self.spin_confidence.value() / 100
        self.config['speed_limit'] = self.spin_speed_limit.value()

        self.video_thread = VideoThread(source, self.config)
        self.video_thread.frame_ready.connect(self._on_frame_ready)
        self.video_thread.error.connect(self._on_error)
        self.video_thread.start()

        self.btn_open.setEnabled(False)
        self.btn_camera.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.statusBar.showMessage("正在处理...")

    def _stop_video(self):
        """停止处理"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None

        self.btn_open.setEnabled(True)
        self.btn_camera.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.statusBar.showMessage("已停止")

    def _on_frame_ready(self, frame: np.ndarray, tracks: list, violations: list):
        """处理帧"""
        self.current_frame = frame.copy()

        # 绘制检测结果
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{track.track_id} {track.class_name}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示帧
        self._display_frame(frame)

        # 更新统计
        self.label_vehicle_count.setText(f"车辆数量: {len(tracks)}")

        # 更新车辆表格
        self.vehicle_table.setRowCount(len(tracks))
        for i, track in enumerate(tracks):
            self.vehicle_table.setItem(i, 0, QTableWidgetItem(str(track.track_id)))
            self.vehicle_table.setItem(i, 1, QTableWidgetItem(track.class_name))
            self.vehicle_table.setItem(i, 2, QTableWidgetItem("-"))
            self.vehicle_table.setItem(i, 3, QTableWidgetItem("-"))
            self.vehicle_table.setItem(i, 4, QTableWidgetItem("-"))

        # 处理违规
        for v in violations:
            self._add_violation_record(v)

    def _display_frame(self, frame: np.ndarray):
        """显示帧"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 缩放以适应标签
        scaled = qt_image.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(QPixmap.fromImage(scaled))

    def _add_violation_record(self, violation):
        """添加违规记录"""
        row = self.violation_table.rowCount()
        self.violation_table.insertRow(row)
        self.violation_table.setItem(
            row, 0, QTableWidgetItem(violation.timestamp.strftime("%H:%M:%S"))
        )
        self.violation_table.setItem(
            row, 1, QTableWidgetItem(violation.violation_type.value)
        )
        self.violation_table.setItem(
            row, 2, QTableWidgetItem(violation.plate_number or "-")
        )
        self.violation_table.setItem(
            row, 3, QTableWidgetItem(f"{violation.speed:.1f}" if violation.speed else "-")
        )
        self.violation_table.setItem(
            row, 4, QTableWidgetItem(f"{violation.location}")
        )

        # 保存到数据库
        self.database.add_violation(
            violation.track_id,
            violation.violation_type.value,
            violation.location,
            violation.speed,
            violation.plate_number
        )

        # 更新统计
        count = self.violation_table.rowCount()
        self.label_violation_count.setText(f"违规数量: {count}")

    def _search_plate(self):
        """搜索车牌"""
        plate = self.search_input.text().strip()
        if not plate:
            return

        results = self.database.search_by_plate(plate)
        if results:
            QMessageBox.information(
                self, "搜索结果",
                f"找到 {len(results)} 条记录"
            )
        else:
            QMessageBox.information(self, "搜索结果", "未找到匹配记录")

    def _on_error(self, error: str):
        """处理错误"""
        QMessageBox.critical(self, "错误", error)
        self._stop_video()

    def closeEvent(self, event):
        """关闭事件"""
        self._stop_video()
        event.accept()
