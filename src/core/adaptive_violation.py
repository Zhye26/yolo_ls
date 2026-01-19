"""
自适应违规检测模块

创新点：智能识别特殊情况，避免误判
- 避让特种车辆（救护车、消防车、警车）不算违规
- 交警指挥下的行为不算违规
- 紧急避险情况不算违规
- 信号灯故障情况不算违规

特殊情况会被记录但不计入违规，照片以时间命名保存
"""
import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque

from .emergency_vehicle import EmergencyVehicleDetector, EmergencyVehicle, EmergencyVehicleType


class ViolationType(Enum):
    """违规类型"""
    RED_LIGHT = "red_light"      # 闯红灯
    SPEEDING = "speeding"        # 超速
    WRONG_WAY = "wrong_way"      # 逆行
    ILLEGAL_LANE = "illegal_lane"  # 违规变道


class ExemptionReason(Enum):
    """免责原因（特殊情况）"""
    YIELD_TO_EMERGENCY = "yield_to_emergency"    # 避让特种车辆
    POLICE_DIRECTION = "police_direction"        # 交警指挥
    EMERGENCY_AVOIDANCE = "emergency_avoidance"  # 紧急避险
    SIGNAL_MALFUNCTION = "signal_malfunction"    # 信号灯故障
    ROAD_CONSTRUCTION = "road_construction"      # 道路施工
    NONE = "none"                                # 无免责原因


# 免责原因的中文描述
EXEMPTION_DESCRIPTIONS = {
    ExemptionReason.YIELD_TO_EMERGENCY: "避让特种车辆（救护车/消防车/警车）",
    ExemptionReason.POLICE_DIRECTION: "服从交警现场指挥",
    ExemptionReason.EMERGENCY_AVOIDANCE: "紧急避险",
    ExemptionReason.SIGNAL_MALFUNCTION: "信号灯故障",
    ExemptionReason.ROAD_CONSTRUCTION: "道路施工临时改道",
    ExemptionReason.NONE: "无",
}


@dataclass
class ViolationRecord:
    """违规/特殊情况记录"""
    record_id: str                          # 记录ID（时间戳）
    violation_type: ViolationType           # 违规类型
    track_id: int                           # 车辆跟踪ID
    timestamp: datetime                     # 发生时间
    location: Tuple[int, int]               # 位置
    speed: Optional[float] = None           # 速度
    plate_number: Optional[str] = None      # 车牌号
    snapshot_path: Optional[str] = None     # 截图路径
    is_exempted: bool = False               # 是否免责
    exemption_reason: ExemptionReason = ExemptionReason.NONE  # 免责原因
    exemption_details: str = ""             # 免责详情
    nearby_emergency_vehicles: List[str] = field(default_factory=list)  # 附近特种车辆


class TrafficLightDetector:
    """交通灯状态检测器"""

    def __init__(self):
        # 红灯 HSV 范围
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])
        # 绿灯 HSV 范围
        self.green_lower = np.array([35, 100, 100])
        self.green_upper = np.array([85, 255, 255])
        # 黄灯 HSV 范围
        self.yellow_lower = np.array([15, 100, 100])
        self.yellow_upper = np.array([35, 255, 255])

        # 信号灯状态历史（用于检测故障）
        self.state_history = deque(maxlen=30)  # 保存最近30帧的状态

    def detect_state(self, frame: np.ndarray,
                     bbox: Tuple[int, int, int, int]) -> str:
        """检测交通灯状态"""
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return 'unknown'

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        total = roi.shape[0] * roi.shape[1]
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = red_mask1 | red_mask2
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)

        red_ratio = np.sum(red_mask > 0) / total
        green_ratio = np.sum(green_mask > 0) / total
        yellow_ratio = np.sum(yellow_mask > 0) / total

        threshold = 0.1
        if red_ratio > threshold and red_ratio > green_ratio and red_ratio > yellow_ratio:
            state = 'red'
        elif green_ratio > threshold and green_ratio > red_ratio and green_ratio > yellow_ratio:
            state = 'green'
        elif yellow_ratio > threshold:
            state = 'yellow'
        else:
            state = 'unknown'

        self.state_history.append(state)
        return state

    def is_malfunctioning(self) -> bool:
        """
        检测信号灯是否故障

        故障特征：
        1. 长时间unknown状态
        2. 快速闪烁（非正常黄灯闪烁）
        """
        if len(self.state_history) < 10:
            return False

        recent = list(self.state_history)[-10:]

        # 检查是否长时间unknown
        unknown_count = recent.count('unknown')
        if unknown_count >= 8:
            return True

        # 检查是否异常闪烁（状态变化过于频繁）
        changes = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
        if changes >= 6:  # 10帧内变化6次以上
            return True

        return False


class StopLine:
    """停止线"""

    def __init__(self, y: int, x_start: int, x_end: int):
        self.y = y
        self.x_start = x_start
        self.x_end = x_end

    def is_crossed(self, prev_center: Tuple[int, int],
                   curr_center: Tuple[int, int]) -> bool:
        """判断是否越过停止线"""
        if not (self.x_start <= curr_center[0] <= self.x_end):
            return False
        if prev_center[1] < self.y <= curr_center[1]:
            return True
        return False


class AdaptiveViolationDetector:
    """
    自适应违规检测器

    创新点：
    1. 智能识别特种车辆，避让特种车辆时的违规行为自动免责
    2. 检测信号灯故障，故障期间的违规自动免责
    3. 支持多种免责原因的记录和管理
    4. 特殊情况单独记录，便于后续人工复核
    """

    def __init__(self,
                 speed_limit: float = 60.0,
                 stop_line: Optional[StopLine] = None,
                 snapshot_dir: str = "data/snapshots",
                 emergency_distance: int = 300):
        """
        初始化自适应违规检测器

        Args:
            speed_limit: 速度限制 (km/h)
            stop_line: 停止线对象
            snapshot_dir: 截图保存目录
            emergency_distance: 特种车辆影响距离（像素）
        """
        self.speed_limit = speed_limit
        self.stop_line = stop_line
        self.snapshot_dir = Path(snapshot_dir)
        self.emergency_distance = emergency_distance

        # 创建截图目录
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        (self.snapshot_dir / "violations").mkdir(exist_ok=True)
        (self.snapshot_dir / "exempted").mkdir(exist_ok=True)

        # 子模块
        self.traffic_light_detector = TrafficLightDetector()
        self.emergency_detector = EmergencyVehicleDetector()

        # 状态
        self.track_history: Dict[int, deque] = {}
        self.recorded_violations: Dict[int, set] = {}
        self.current_light_state = 'unknown'
        self.current_emergency_vehicles: List[EmergencyVehicle] = []

        # 统计
        self.total_violations = 0
        self.exempted_count = 0

    def set_stop_line(self, y: int, x_start: int, x_end: int):
        """设置停止线"""
        self.stop_line = StopLine(y, x_start, x_end)

    def update(self, frame: np.ndarray,
               vehicle_bboxes: List[Tuple[int, int, int, int]],
               light_bbox: Optional[Tuple[int, int, int, int]] = None):
        """
        更新检测器状态

        Args:
            frame: 当前帧
            vehicle_bboxes: 所有车辆边界框
            light_bbox: 交通灯边界框
        """
        # 更新交通灯状态
        if light_bbox:
            self.current_light_state = self.traffic_light_detector.detect_state(
                frame, light_bbox
            )

        # 检测特种车辆
        self.current_emergency_vehicles = self.emergency_detector.detect(
            frame, vehicle_bboxes
        )

    def check_violation(self,
                        track_id: int,
                        bbox: Tuple[int, int, int, int],
                        speed: float,
                        frame: np.ndarray,
                        plate_number: Optional[str] = None) -> Optional[ViolationRecord]:
        """
        检查单个车辆的违规行为

        Args:
            track_id: 跟踪ID
            bbox: 车辆边界框
            speed: 当前速度
            frame: 当前帧
            plate_number: 车牌号

        Returns:
            违规记录（如果有违规）
        """
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # 初始化跟踪历史
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=10)
            self.recorded_violations[track_id] = set()

        history = self.track_history[track_id]

        # 检测违规
        violation_type = None
        violation_speed = None

        # 1. 检查超速
        if speed > self.speed_limit:
            if ViolationType.SPEEDING not in self.recorded_violations[track_id]:
                violation_type = ViolationType.SPEEDING
                violation_speed = speed

        # 2. 检查闯红灯
        if (violation_type is None and
            self.stop_line and
            len(history) > 0 and
            self.current_light_state == 'red'):

            prev_center = history[-1]
            if self.stop_line.is_crossed(prev_center, center):
                if ViolationType.RED_LIGHT not in self.recorded_violations[track_id]:
                    violation_type = ViolationType.RED_LIGHT

        # 更新历史
        history.append(center)

        # 如果没有违规，返回None
        if violation_type is None:
            return None

        # 检查是否有免责原因
        is_exempted, exemption_reason, exemption_details, nearby_evs = \
            self._check_exemption(bbox, violation_type)

        # 生成记录
        timestamp = datetime.now()
        record_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")

        # 保存截图
        snapshot_path = self._save_snapshot(
            frame, record_id, is_exempted, bbox, violation_type
        )

        # 创建记录
        record = ViolationRecord(
            record_id=record_id,
            violation_type=violation_type,
            track_id=track_id,
            timestamp=timestamp,
            location=center,
            speed=violation_speed,
            plate_number=plate_number,
            snapshot_path=snapshot_path,
            is_exempted=is_exempted,
            exemption_reason=exemption_reason,
            exemption_details=exemption_details,
            nearby_emergency_vehicles=nearby_evs
        )

        # 更新统计
        self.total_violations += 1
        if is_exempted:
            self.exempted_count += 1
        else:
            self.recorded_violations[track_id].add(violation_type)

        return record

    def _check_exemption(self, vehicle_bbox: Tuple[int, int, int, int],
                         violation_type: ViolationType) -> Tuple[bool, ExemptionReason, str, List[str]]:
        """
        检查是否有免责原因

        Returns:
            (是否免责, 免责原因, 详情描述, 附近特种车辆列表)
        """
        nearby_evs = []

        # 1. 检查是否在避让特种车辆
        if self.current_emergency_vehicles:
            for ev in self.current_emergency_vehicles:
                # 计算距离
                vx = (vehicle_bbox[0] + vehicle_bbox[2]) // 2
                vy = (vehicle_bbox[1] + vehicle_bbox[3]) // 2
                ex = (ev.bbox[0] + ev.bbox[2]) // 2
                ey = (ev.bbox[1] + ev.bbox[3]) // 2

                distance = np.sqrt((vx - ex)**2 + (vy - ey)**2)

                if distance < self.emergency_distance:
                    ev_name = self._get_emergency_vehicle_name(ev.vehicle_type)
                    nearby_evs.append(ev_name)

            if nearby_evs:
                details = f"检测到附近有特种车辆: {', '.join(nearby_evs)}，车辆可能在避让"
                return True, ExemptionReason.YIELD_TO_EMERGENCY, details, nearby_evs

        # 2. 检查信号灯是否故障
        if violation_type == ViolationType.RED_LIGHT:
            if self.traffic_light_detector.is_malfunctioning():
                details = "检测到信号灯可能存在故障（状态异常）"
                return True, ExemptionReason.SIGNAL_MALFUNCTION, details, nearby_evs

        return False, ExemptionReason.NONE, "", nearby_evs

    def _get_emergency_vehicle_name(self, ev_type: EmergencyVehicleType) -> str:
        """获取特种车辆中文名称"""
        names = {
            EmergencyVehicleType.AMBULANCE: "救护车",
            EmergencyVehicleType.FIRE_TRUCK: "消防车",
            EmergencyVehicleType.POLICE_CAR: "警车",
            EmergencyVehicleType.RESCUE_VEHICLE: "工程救险车",
            EmergencyVehicleType.UNKNOWN: "特种车辆",
        }
        return names.get(ev_type, "特种车辆")

    def _save_snapshot(self, frame: np.ndarray, record_id: str,
                       is_exempted: bool, bbox: Tuple[int, int, int, int],
                       violation_type: ViolationType) -> str:
        """
        保存违规截图

        文件名格式: YYYYMMDD_HHMMSS_ffffff_类型.jpg
        """
        # 在图像上标注
        annotated = frame.copy()
        x1, y1, x2, y2 = bbox

        # 绘制边界框
        color = (0, 255, 255) if is_exempted else (0, 0, 255)  # 黄色=免责，红色=违规
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # 添加文字标注
        label = f"{violation_type.value}"
        if is_exempted:
            label += " [EXEMPTED]"
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 添加时间戳
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, time_str, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 保存
        subdir = "exempted" if is_exempted else "violations"
        filename = f"{record_id}_{violation_type.value}.jpg"
        filepath = self.snapshot_dir / subdir / filename

        cv2.imwrite(str(filepath), annotated)
        return str(filepath)

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "total_violations": self.total_violations,
            "exempted_count": self.exempted_count,
            "actual_violations": self.total_violations - self.exempted_count,
            "exemption_rate": self.exempted_count / max(1, self.total_violations),
        }

    def clear_track(self, track_id: int):
        """清除跟踪记录"""
        if track_id in self.track_history:
            del self.track_history[track_id]
        if track_id in self.recorded_violations:
            del self.recorded_violations[track_id]

    def draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        """在帧上绘制标注"""
        annotated = frame.copy()

        # 绘制停止线
        if self.stop_line:
            color = (0, 0, 255) if self.current_light_state == 'red' else (0, 255, 0)
            cv2.line(annotated,
                     (self.stop_line.x_start, self.stop_line.y),
                     (self.stop_line.x_end, self.stop_line.y),
                     color, 2)

        # 绘制交通灯状态
        light_colors = {'red': (0, 0, 255), 'green': (0, 255, 0),
                        'yellow': (0, 255, 255), 'unknown': (128, 128, 128)}
        cv2.circle(annotated, (30, 60), 15,
                   light_colors.get(self.current_light_state, (128, 128, 128)), -1)

        # 标注特种车辆
        for ev in self.current_emergency_vehicles:
            x1, y1, x2, y2 = ev.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 3)
            label = self._get_emergency_vehicle_name(ev.vehicle_type)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # 显示统计
        stats = self.get_statistics()
        info = f"Violations: {stats['actual_violations']} | Exempted: {stats['exempted_count']}"
        cv2.putText(annotated, info, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated
