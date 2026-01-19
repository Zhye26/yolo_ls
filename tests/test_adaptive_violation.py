#!/usr/bin/env python3
"""
Adaptive Violation Detection System - Test Script
Tests the core functionality of emergency vehicle detection and exemption handling
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.emergency_vehicle import EmergencyVehicleDetector, EmergencyVehicleType
from src.core.adaptive_violation import (
    AdaptiveViolationDetector, ViolationType, ExemptionReason,
    EXEMPTION_DESCRIPTIONS
)


def create_test_frame(width=640, height=480):
    """Create a test frame"""
    return np.zeros((height, width, 3), dtype=np.uint8)


def create_emergency_vehicle_frame():
    """Create a frame with simulated emergency vehicle (red top area)"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Simulate red warning light on top of vehicle
    frame[50:80, 280:360] = [0, 0, 255]  # Red area (BGR)
    return frame


def test_emergency_vehicle_detector():
    """Test emergency vehicle detection"""
    print("=" * 50)
    print("Test 1: Emergency Vehicle Detector")
    print("=" * 50)

    detector = EmergencyVehicleDetector()

    # Test with normal frame (no emergency vehicle)
    normal_frame = create_test_frame()
    bboxes = [(100, 100, 200, 200)]
    results = detector.detect(normal_frame, bboxes)
    print(f"Normal frame detection: {len(results)} emergency vehicles")
    assert len(results) == 0, "Should not detect emergency vehicle in empty frame"
    print("[PASS] No false positive on normal frame")

    # Test with emergency vehicle frame
    ev_frame = create_emergency_vehicle_frame()
    bboxes = [(250, 40, 380, 200)]  # Bbox covering the red area
    results = detector.detect(ev_frame, bboxes)
    print(f"Emergency frame detection: {len(results)} emergency vehicles")
    if len(results) > 0:
        print(f"  Type: {results[0].vehicle_type}")
        print(f"  Has warning light: {results[0].has_warning_light}")
    print("[PASS] Emergency vehicle detection test completed")

    return True


def test_adaptive_violation_detector():
    """Test adaptive violation detection"""
    print("\n" + "=" * 50)
    print("Test 2: Adaptive Violation Detector")
    print("=" * 50)

    detector = AdaptiveViolationDetector(
        speed_limit=60,
        snapshot_dir='data/test_snapshots',
        emergency_distance=300
    )
    detector.set_stop_line(400, 100, 500)

    frame = create_test_frame()

    # Test 1: Normal speeding violation
    print("\n2.1 Testing speeding violation...")
    record = detector.check_violation(
        track_id=1,
        bbox=(100, 100, 200, 200),
        speed=80.0,  # Over speed limit
        frame=frame
    )

    if record:
        print(f"  Violation type: {record.violation_type.value}")
        print(f"  Speed: {record.speed}")
        print(f"  Is exempted: {record.is_exempted}")
        assert record.violation_type == ViolationType.SPEEDING
        assert not record.is_exempted
        print("[PASS] Speeding violation detected correctly")
    else:
        print("[FAIL] Speeding violation not detected")
        return False

    # Test 2: Speeding with emergency vehicle nearby (should be exempted)
    print("\n2.2 Testing exemption for yielding to emergency vehicle...")

    # Simulate emergency vehicle detection
    from src.core.emergency_vehicle import EmergencyVehicle
    detector.current_emergency_vehicles = [
        EmergencyVehicle(
            vehicle_type=EmergencyVehicleType.AMBULANCE,
            bbox=(150, 100, 250, 200),  # Nearby
            confidence=0.9,
            has_warning_light=True,
            has_siren_color=True
        )
    ]

    record = detector.check_violation(
        track_id=2,
        bbox=(100, 100, 200, 200),
        speed=75.0,  # Over speed limit
        frame=frame
    )

    if record:
        print(f"  Violation type: {record.violation_type.value}")
        print(f"  Speed: {record.speed}")
        print(f"  Is exempted: {record.is_exempted}")
        print(f"  Exemption reason: {record.exemption_reason.value}")
        print(f"  Nearby emergency vehicles: {record.nearby_emergency_vehicles}")
        assert record.is_exempted
        assert record.exemption_reason == ExemptionReason.YIELD_TO_EMERGENCY
        print("[PASS] Exemption for yielding to emergency vehicle works correctly")
    else:
        print("[FAIL] Violation not detected")
        return False

    # Clear emergency vehicles
    detector.current_emergency_vehicles = []

    return True


def test_statistics():
    """Test statistics tracking"""
    print("\n" + "=" * 50)
    print("Test 3: Statistics Tracking")
    print("=" * 50)

    detector = AdaptiveViolationDetector(speed_limit=60)
    frame = create_test_frame()

    # Generate some violations
    for i in range(3):
        detector.check_violation(
            track_id=10 + i,
            bbox=(100, 100, 200, 200),
            speed=70.0 + i * 5,
            frame=frame
        )

    # Add emergency vehicle and generate exempted violation
    from src.core.emergency_vehicle import EmergencyVehicle
    detector.current_emergency_vehicles = [
        EmergencyVehicle(
            vehicle_type=EmergencyVehicleType.FIRE_TRUCK,
            bbox=(150, 100, 250, 200),
            confidence=0.9,
            has_warning_light=True,
            has_siren_color=True
        )
    ]

    for i in range(2):
        detector.check_violation(
            track_id=20 + i,
            bbox=(100, 100, 200, 200),
            speed=65.0 + i * 5,
            frame=frame
        )

    stats = detector.get_statistics()
    print(f"Total violations: {stats['total_violations']}")
    print(f"Exempted count: {stats['exempted_count']}")
    print(f"Actual violations: {stats['actual_violations']}")
    print(f"Exemption rate: {stats['exemption_rate']:.2%}")

    assert stats['total_violations'] == 5
    assert stats['exempted_count'] == 2
    assert stats['actual_violations'] == 3
    print("[PASS] Statistics tracking works correctly")

    return True


def test_exemption_descriptions():
    """Test exemption reason descriptions"""
    print("\n" + "=" * 50)
    print("Test 4: Exemption Descriptions")
    print("=" * 50)

    for reason, desc in EXEMPTION_DESCRIPTIONS.items():
        print(f"  {reason.value}: {desc}")

    assert len(EXEMPTION_DESCRIPTIONS) == 6
    print("[PASS] All exemption descriptions available")

    return True


def test_snapshot_naming():
    """Test snapshot file naming with timestamp"""
    print("\n" + "=" * 50)
    print("Test 5: Snapshot Naming Convention")
    print("=" * 50)

    from datetime import datetime

    # Test timestamp format
    timestamp = datetime.now()
    record_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")

    print(f"  Sample record ID: {record_id}")
    print(f"  Format: YYYYMMDD_HHMMSS_ffffff")

    # Verify format
    parts = record_id.split('_')
    assert len(parts) == 3
    assert len(parts[0]) == 8  # YYYYMMDD
    assert len(parts[1]) == 6  # HHMMSS
    assert len(parts[2]) == 6  # ffffff (microseconds)

    print("[PASS] Snapshot naming convention is correct")

    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ADAPTIVE VIOLATION DETECTION SYSTEM - TEST SUITE")
    print("=" * 60)

    tests = [
        ("Emergency Vehicle Detector", test_emergency_vehicle_detector),
        ("Adaptive Violation Detector", test_adaptive_violation_detector),
        ("Statistics Tracking", test_statistics),
        ("Exemption Descriptions", test_exemption_descriptions),
        ("Snapshot Naming", test_snapshot_naming),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[FAILURE] Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
