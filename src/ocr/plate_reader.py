"""车牌 OCR 识别模块"""
import cv2
import numpy as np
import re
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class PlateResult:
    """车牌识别结果"""
    plate_number: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # 车牌位置


class PlateDetector:
    """车牌检测器（基于颜色和形态学）"""

    def __init__(self):
        # 蓝色车牌 HSV 范围
        self.blue_lower = np.array([100, 80, 80])
        self.blue_upper = np.array([130, 255, 255])
        # 黄色车牌 HSV 范围
        self.yellow_lower = np.array([15, 80, 80])
        self.yellow_upper = np.array([40, 255, 255])
        # 绿色车牌 HSV 范围（新能源）
        self.green_lower = np.array([35, 80, 80])
        self.green_upper = np.array([85, 255, 255])

    def detect(self, frame: np.ndarray,
               vehicle_bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """
        在车辆区域内检测车牌

        Args:
            frame: BGR 图像
            vehicle_bbox: 车辆边界框

        Returns:
            车牌边界框或 None
        """
        x1, y1, x2, y2 = vehicle_bbox
        # 车牌通常在车辆下半部分
        roi_y1 = y1 + (y2 - y1) // 2
        roi = frame[roi_y1:y2, x1:x2]

        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 检测蓝色、黄色、绿色车牌
        masks = [
            cv2.inRange(hsv, self.blue_lower, self.blue_upper),
            cv2.inRange(hsv, self.yellow_lower, self.yellow_upper),
            cv2.inRange(hsv, self.green_lower, self.green_upper),
        ]
        mask = masks[0] | masks[1] | masks[2]

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_rect = None
        best_score = 0

        for cnt in contours:
            rect = cv2.boundingRect(cnt)
            rx, ry, rw, rh = rect

            # 车牌宽高比约为 3:1 到 4:1
            aspect_ratio = rw / rh if rh > 0 else 0
            if 2.5 < aspect_ratio < 5.0 and rw > 60 and rh > 15:
                area = rw * rh
                if area > best_score:
                    best_score = area
                    best_rect = (x1 + rx, roi_y1 + ry, x1 + rx + rw, roi_y1 + ry + rh)

        return best_rect


class PlateOCR:
    """车牌 OCR 识别器"""

    def __init__(self, use_gpu: bool = True):
        """
        初始化 OCR

        Args:
            use_gpu: 是否使用 GPU
        """
        self.ocr = None
        self.use_gpu = use_gpu
        self._init_ocr()

    def _init_ocr(self):
        """初始化 PaddleOCR"""
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                lang='ch',
                use_textline_orientation=True
            )
        except ImportError:
            print("Warning: PaddleOCR not installed, OCR disabled")
            self.ocr = None

    def recognize(self, frame: np.ndarray,
                  plate_bbox: Tuple[int, int, int, int]) -> Optional[PlateResult]:
        """
        识别车牌号

        Args:
            frame: BGR 图像
            plate_bbox: 车牌边界框

        Returns:
            PlateResult 或 None
        """
        if self.ocr is None:
            return None

        x1, y1, x2, y2 = plate_bbox
        plate_img = frame[y1:y2, x1:x2]

        if plate_img.size == 0:
            return None

        # 预处理
        plate_img = cv2.resize(plate_img, (140, 40))

        try:
            result = self.ocr.ocr(plate_img, cls=True)
            if result and result[0]:
                text = ''
                conf = 0.0
                for line in result[0]:
                    if line[1]:
                        text += line[1][0]
                        conf = max(conf, line[1][1])

                # 清理和验证车牌号
                plate_number = self._clean_plate(text)
                if plate_number:
                    return PlateResult(
                        plate_number=plate_number,
                        confidence=conf,
                        bbox=plate_bbox
                    )
        except Exception:
            pass

        return None

    def _clean_plate(self, text: str) -> Optional[str]:
        """
        清理和验证车牌号

        Args:
            text: OCR 识别的原始文本

        Returns:
            清理后的车牌号或 None
        """
        # 移除空格和特殊字符
        text = re.sub(r'[^\u4e00-\u9fa5A-Z0-9]', '', text.upper())

        # 中国车牌格式验证
        # 普通车牌：省份简称 + 字母 + 5位字母数字
        # 新能源车牌：省份简称 + 字母 + 6位
        pattern = r'^[\u4e00-\u9fa5][A-Z][A-Z0-9]{5,6}$'

        if re.match(pattern, text):
            return text

        return None


class PlateReader:
    """车牌识别器（整合检测和 OCR）"""

    def __init__(self, use_gpu: bool = True):
        self.detector = PlateDetector()
        self.ocr = PlateOCR(use_gpu)

    def read(self, frame: np.ndarray,
             vehicle_bbox: Tuple[int, int, int, int]) -> Optional[PlateResult]:
        """
        读取车牌

        Args:
            frame: BGR 图像
            vehicle_bbox: 车辆边界框

        Returns:
            PlateResult 或 None
        """
        # 检测车牌位置
        plate_bbox = self.detector.detect(frame, vehicle_bbox)
        if plate_bbox is None:
            return None

        # OCR 识别
        return self.ocr.recognize(frame, plate_bbox)
