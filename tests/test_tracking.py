"""Crop/offset maths for tracked detection, against a stub detector."""

from typing import ClassVar

import numpy as np
import pytest

from facestream.faceswap import FaceSwap, FaceTracker

FRAME_SHAPE = (720, 1280)
EMPTY = (np.zeros((0, 5), dtype=np.float32), None)


def detection(bbox, kps):
    return (
        np.array([[*bbox, 0.9]], dtype=np.float32),
        np.array([kps], dtype=np.float32),
    )


class StubDetector:
    """Stands in for SCRFD: records its calls, returns canned detections."""

    # A dynamic ONNX input shape, which is what enables per-call sizes.
    input_shape: ClassVar[list] = [1, 3, "?", "?"]

    def __init__(self, results):
        self.calls = []
        self.results = list(results)

    def detect(self, img, input_size=None, max_num=0, metric="default"):
        self.calls.append({"shape": img.shape[:2], "input_size": input_size})
        return self.results.pop(0)


@pytest.fixture
def frame():
    return np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8)


def make_swapper(detector):
    swapper = object.__new__(FaceSwap)
    swapper.det_model = detector
    swapper._det_is_dynamic = True
    swapper.det_size = (640, 640)
    swapper.track_det_size = (320, 320)
    return swapper


FACE = [500.0, 200.0, 700.0, 460.0]
FACE_KPS = [[560, 300], [660, 300], [610, 360], [570, 420], [650, 420]]


def test_searches_the_whole_frame_when_nothing_is_tracked(frame):
    detector = StubDetector([detection(FACE, FACE_KPS)])
    swapper = make_swapper(detector)
    tracker = FaceTracker()

    bbox, kps = swapper._detect_target(frame, tracker)

    assert detector.calls[0]["shape"] == FRAME_SHAPE
    assert detector.calls[0]["input_size"] == (640, 640)
    assert list(bbox) == FACE
    assert kps[0].tolist() == FACE_KPS[0]
    assert tracker.last_bbox is not None


def test_tracked_frame_searches_a_crop_and_maps_back(frame):
    detector = StubDetector(
        [
            detection(FACE, FACE_KPS),
            # Reported relative to the crop.
            detection(
                [10.0, 6.0, 210.0, 266.0],
                [[70, 106], [170, 106], [120, 166], [80, 226], [160, 226]],
            ),
        ]
    )
    swapper = make_swapper(detector)
    tracker = FaceTracker()

    swapper._detect_target(frame, tracker)
    bbox, kps = swapper._detect_target(frame, tracker)

    x0, y0, x1, y1 = swapper._tracked_roi(np.array(FACE), FRAME_SHAPE)
    assert (x0, y0, x1, y1) == (380, 44, 820, 616)
    assert detector.calls[1]["shape"] == (y1 - y0, x1 - x0)
    assert detector.calls[1]["input_size"] == (320, 320)
    assert list(bbox) == [10 + x0, 6 + y0, 210 + x0, 266 + y0]
    assert kps.tolist() == [
        [70 + x0, 106 + y0],
        [170 + x0, 106 + y0],
        [120 + x0, 166 + y0],
        [80 + x0, 226 + y0],
        [160 + x0, 226 + y0],
    ]


def test_losing_the_face_in_the_crop_falls_back_to_the_frame(frame):
    found = detection(
        [100.0, 100.0, 200.0, 230.0],
        [[120, 140], [180, 140], [150, 170], [125, 200], [175, 200]],
    )
    detector = StubDetector([detection(FACE, FACE_KPS), EMPTY, found])
    swapper = make_swapper(detector)
    tracker = FaceTracker()

    swapper._detect_target(frame, tracker)
    bbox, _ = swapper._detect_target(frame, tracker)

    assert detector.calls[1]["input_size"] == (320, 320)
    assert detector.calls[2]["shape"] == FRAME_SHAPE
    assert list(bbox) == [100, 100, 200, 230]


def test_no_face_anywhere_clears_the_tracker(frame):
    detector = StubDetector([detection(FACE, FACE_KPS), EMPTY, EMPTY, EMPTY])
    swapper = make_swapper(detector)
    tracker = FaceTracker()

    swapper._detect_target(frame, tracker)
    assert swapper._detect_target(frame, tracker) is None
    assert tracker.last_bbox is None

    # And the next frame goes straight to a full-frame search.
    calls_before = len(detector.calls)
    swapper._detect_target(frame, tracker)
    assert len(detector.calls) == calls_before + 1
    assert detector.calls[-1]["input_size"] == (640, 640)


def test_picks_the_largest_face_not_the_leftmost(frame):
    detector = StubDetector(
        [
            (
                np.array(
                    [
                        [10.0, 10.0, 60.0, 70.0, 0.9],  # leftmost, small
                        [400.0, 100.0, 700.0, 480.0, 0.9],  # largest
                    ],
                    dtype=np.float32,
                ),
                np.array(
                    [
                        [[20, 20], [50, 20], [35, 40], [25, 60], [45, 60]],
                        [[480, 200], [620, 200], [550, 300], [500, 400], [600, 400]],
                    ],
                    dtype=np.float32,
                ),
            )
        ]
    )
    swapper = make_swapper(detector)

    bbox, kps = swapper._detect_target(frame, FaceTracker())

    assert list(bbox) == [400, 100, 700, 480]
    assert kps[0].tolist() == [480, 200]


def test_crop_clipped_by_the_frame_edge_still_maps_back(frame):
    detector = StubDetector(
        [
            detection(
                [0.0, 0.0, 120.0, 150.0],
                [[20, 30], [80, 30], [50, 60], [25, 100], [75, 100]],
            ),
            detection(
                [5.0, 5.0, 100.0, 130.0],
                [[20, 30], [80, 30], [50, 60], [25, 100], [75, 100]],
            ),
        ]
    )
    swapper = make_swapper(detector)
    tracker = FaceTracker()

    swapper._detect_target(frame, tracker)
    bbox, _ = swapper._detect_target(frame, tracker)

    assert swapper._tracked_roi(np.array([0.0, 0.0, 120.0, 150.0]), FRAME_SHAPE)[
        :2
    ] == (
        0,
        0,
    )
    assert list(bbox) == [5, 5, 100, 130]


def test_resolution_change_drops_the_tracked_box(frame):
    """Browsers rescale mid-stream; a box in the old scale would misdirect the crop."""
    detector = StubDetector([detection(FACE, FACE_KPS), detection(FACE, FACE_KPS)])
    swapper = make_swapper(detector)
    tracker = FaceTracker()

    swapper._detect_target(frame, tracker)
    assert tracker.last_bbox is not None

    smaller = np.zeros((540, 960, 3), dtype=np.uint8)
    swapper._detect_target(smaller, tracker)

    # Second call searched the whole (smaller) frame rather than a stale crop.
    assert detector.calls[1]["shape"] == (540, 960)
    assert detector.calls[1]["input_size"] == (640, 640)
