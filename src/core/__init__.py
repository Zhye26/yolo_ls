"""Core modules"""
from .detector import VehicleDetector, Detection
from .tracker import ByteTracker, Track
from .feature import FeatureExtractor, VehicleFeatures, Direction, ColorAnalyzer, SpeedCalculator
from .violation import ViolationDetector, Violation, ViolationType, StopLine
from .emergency_vehicle import EmergencyVehicleDetector, EmergencyVehicle, EmergencyVehicleType
from .adaptive_violation import (
    AdaptiveViolationDetector, ViolationRecord, ExemptionReason,
    EXEMPTION_DESCRIPTIONS, ViolationType as AdaptiveViolationType
)
from .stgat import VehicleInteractionGraph, SpatioTemporalGAT
from .collision_risk import CollisionRiskPredictor, CollisionRisk, RiskLevel

__all__ = [
    'VehicleDetector', 'Detection',
    'ByteTracker', 'Track',
    'FeatureExtractor', 'VehicleFeatures', 'Direction', 'ColorAnalyzer', 'SpeedCalculator',
    'ViolationDetector', 'Violation', 'ViolationType', 'StopLine',
    'EmergencyVehicleDetector', 'EmergencyVehicle', 'EmergencyVehicleType',
    'AdaptiveViolationDetector', 'ViolationRecord', 'ExemptionReason',
    'EXEMPTION_DESCRIPTIONS', 'AdaptiveViolationType',
    'VehicleInteractionGraph', 'SpatioTemporalGAT',
    'CollisionRiskPredictor', 'CollisionRisk', 'RiskLevel'
]
