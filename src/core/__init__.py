"""核心模块"""
from .detector import VehicleDetector, Detection
from .tracker import ByteTracker, Track
from .feature import FeatureExtractor, VehicleFeatures, Direction, ColorAnalyzer, SpeedCalculator
from .violation import ViolationDetector, Violation, ViolationType, StopLine

__all__ = [
    'VehicleDetector', 'Detection',
    'ByteTracker', 'Track',
    'FeatureExtractor', 'VehicleFeatures', 'Direction', 'ColorAnalyzer', 'SpeedCalculator',
    'ViolationDetector', 'Violation', 'ViolationType', 'StopLine'
]
