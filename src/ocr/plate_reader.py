"""车牌 OCR 识别模块"""
import cv2
import numpy as np
import re
from typing import Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms


# 中国车牌字符集
CHARS = [
    '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
    '新', '警', '学', '挂',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '-'  # blank for CTC
]
IDX2CHAR = {idx: char for idx, char in enumerate(CHARS)}
BLANK_IDX = len(CHARS) - 1


class CRNN(nn.Module):
    """CRNN 车牌识别模型"""

    def __init__(self, num_classes: int = len(CHARS), hidden_size: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, (2, 1), 1, 0), nn.ReLU(),
        )
        self.rnn = nn.LSTM(512, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        conv = conv.squeeze(2).permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv)
        output = self.fc(rnn_out)
        return output.permute(1, 0, 2)


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
    """车牌 OCR 识别器（使用 CRNN 模型）"""

    def __init__(self, model_path: str = "models/plate_ocr.pt", use_gpu: bool = True):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """加载 CRNN 模型"""
        path = Path(model_path)
        if not path.exists():
            # 尝试相对于项目根目录
            path = Path(__file__).resolve().parents[2] / model_path
        if path.exists():
            self.model = CRNN().to(self.device)
            self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            self.model.eval()
            print(f"Loaded plate OCR model from {path}")
        else:
            print(f"Warning: Plate OCR model not found at {model_path}")

    def _decode(self, preds: torch.Tensor) -> Tuple[str, float]:
        """CTC 解码"""
        preds_softmax = torch.softmax(preds, dim=2)
        preds_max, preds_idx = preds_softmax.max(2)
        preds_idx = preds_idx.permute(1, 0).cpu().numpy()[0]
        preds_max = preds_max.permute(1, 0).cpu().numpy()[0]

        chars, confs = [], []
        prev = -1
        for i, p in enumerate(preds_idx):
            if p != prev and p != BLANK_IDX:
                chars.append(IDX2CHAR.get(p, ''))
                confs.append(preds_max[i])
            prev = p

        text = ''.join(chars)
        conf = float(np.mean(confs)) if confs else 0.0
        return text, conf

    def recognize(self, frame: np.ndarray,
                  plate_bbox: Tuple[int, int, int, int]) -> Optional[PlateResult]:
        if self.model is None:
            return None

        x1, y1, x2, y2 = plate_bbox
        plate_img = frame[y1:y2, x1:x2]
        if plate_img.size == 0:
            return None

        # BGR to RGB
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)

        try:
            img_tensor = self.transform(plate_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                preds = self.model(img_tensor)
            text, conf = self._decode(preds)

            plate_number = self._clean_plate(text)
            if plate_number:
                return PlateResult(plate_number=plate_number, confidence=conf, bbox=plate_bbox)
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

    def __init__(self, model_path: str = "models/plate_ocr.pt", use_gpu: bool = True):
        self.detector = PlateDetector()
        self.ocr = PlateOCR(model_path, use_gpu)

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
