import numpy as np

import mcp_server.yolo_detector as yolo_module


class _FakeDetector:
    instances = 0

    def __init__(self, _model_path=None, confidence=None, _device=None):
        type(self).instances += 1
        self.confidence = confidence if confidence is not None else 0.35

    def detect(self, _image, _target, **_kwargs):
        return {"found": True}

    def detect_all(self, _image):
        return []

    @staticmethod
    def is_loaded():
        return True


def test_lazy_yolo_does_not_construct_model_until_detection(monkeypatch):
    _FakeDetector.instances = 0
    monkeypatch.setattr(yolo_module, "YoloDetector", _FakeDetector)

    detector = yolo_module.LazyYoloDetector(confidence=0.42)
    assert _FakeDetector.instances == 0
    assert detector.is_loaded() is False
    assert detector.confidence == 0.42

    result = detector.detect(np.zeros((8, 8, 3), dtype=np.uint8), "block")
    assert result["found"] is True
    assert _FakeDetector.instances == 1
    assert detector.is_loaded() is True

    detector.detect(np.zeros((8, 8, 3), dtype=np.uint8), "block")
    assert _FakeDetector.instances == 1


def test_confidence_update_is_preserved_before_and_after_load(monkeypatch):
    _FakeDetector.instances = 0
    monkeypatch.setattr(yolo_module, "YoloDetector", _FakeDetector)
    detector = yolo_module.LazyYoloDetector()
    detector.confidence = 0.5
    detector.detect_all(np.zeros((8, 8, 3), dtype=np.uint8))
    assert detector.confidence == 0.5
    detector.confidence = 0.6
    assert detector.confidence == 0.6
