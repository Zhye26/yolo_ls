"""YOLO 目标检测模块"""
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from ultralytics import YOLO


@dataclass
class Detection:
    """检测结果数据类"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Tuple[int, int]:
        """获取边界框中心点"""
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def area(self) -> int:
        """获取边界框面积"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    def to_tlwh(self) -> Tuple[int, int, int, int]:
        """转换为 (top, left, width, height) 格式"""
        x1, y1, x2, y2 = self.bbox
        return x1, y1, x2 - x1, y2 - y1


class VehicleDetector:
    """车辆检测器"""

    # COCO 数据集中的车辆类别
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }

    # 交通灯类别
    TRAFFIC_LIGHT_CLASS = 9

    def __init__(self, model_path: str = "yolo12n.pt",
                 confidence: float = 0.2,
                 iou_threshold: float = 0.45,
                 device: str = "cuda"):
        """
        初始化检测器

        Args:
            model_path: YOLO 模型路径
            confidence: 置信度阈值
            iou_threshold: NMS IOU 阈值
            device: 运行设备 (cuda/cpu)
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.class_names = self.model.names

    def detect(self, frame: np.ndarray,
               classes: Optional[List[int]] = None) -> List[Detection]:
        """
        检测图像中的目标

        Args:
            frame: BGR 格式的图像帧
            classes: 要检测的类别ID列表，None 表示检测所有车辆类别

        Returns:
            检测结果列表
        """
        if classes is None:
            classes = list(self.VEHICLE_CLASSES.keys())

        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=classes,
            device=self.device,
            verbose=False
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, 'unknown')

                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name
                ))

        return detections

    def detect_vehicles(self, frame: np.ndarray) -> List[Detection]:
        """仅检测车辆"""
        return self.detect(frame, list(self.VEHICLE_CLASSES.keys()))

    def detect_traffic_lights(self, frame: np.ndarray) -> List[Detection]:
        """检测交通灯"""
        return self.detect(frame, [self.TRAFFIC_LIGHT_CLASS])

    def detect_all(self, frame: np.ndarray) -> Tuple[List[Detection], List[Detection]]:
        """检测车辆和交通灯"""
        all_classes = list(self.VEHICLE_CLASSES.keys()) + [self.TRAFFIC_LIGHT_CLASS]
        all_detections = self.detect(frame, all_classes)

        vehicles = [d for d in all_detections if d.class_id in self.VEHICLE_CLASSES]
        traffic_lights = [d for d in all_detections if d.class_id == self.TRAFFIC_LIGHT_CLASS]

        return vehicles, traffic_lights
