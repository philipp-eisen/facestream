"""ProcessFrameTrack: what actually reaches the wire."""

import asyncio
import fractions
from itertools import pairwise

import numpy as np
import pytest
from av.video.frame import VideoFrame

from facestream import track as track_module
from facestream.track import ProcessFrameTrack

RAW = 10  # what the camera sends
SWAPPED = 200  # what a swapped frame looks like


def make_frame(pts, value):
    frame = VideoFrame.from_ndarray(
        np.full((64, 64, 3), value, dtype=np.uint8), format="bgr24"
    )
    frame.pts = pts
    frame.time_base = fractions.Fraction(1, 90000)
    return frame


def value_of(frame):
    return int(frame.to_ndarray(format="bgr24")[0, 0, 0])


class FakeCamera:
    """Delivers frames at a fixed cadence, like an incoming WebRTC track."""

    kind = "video"

    def __init__(self, interval=1 / 30):
        self.interval = interval
        self.pts = 0

    async def recv(self):
        await asyncio.sleep(self.interval)
        self.pts += 3000
        return make_frame(self.pts, RAW)


async def collect(proc, count):
    """Read `count` frames, recording each at the moment it is returned.

    A repeated frame is the same object restamped in place, so holding
    references and reading them afterwards would only ever show the last value.
    """
    values, timestamps = [], []
    for _ in range(count):
        frame = await proc.recv()
        values.append(value_of(frame))
        timestamps.append(frame.pts)
    return values, timestamps


@pytest.mark.asyncio
async def test_never_sends_the_raw_camera_frame():
    """The real face must not appear while the first swap is still running."""

    async def slow_swap(frame):
        await asyncio.sleep(0.25)
        return make_frame(frame.pts, SWAPPED)

    proc = ProcessFrameTrack(FakeCamera(), slow_swap)
    try:
        values, _ = await collect(proc, 12)
    finally:
        proc.stop()

    assert set(values) == {SWAPPED}


@pytest.mark.asyncio
async def test_timestamps_strictly_increase():
    """A repeat or a late swap must never carry a timestamp already sent."""

    async def slow_swap(frame):
        await asyncio.sleep(0.25)
        return make_frame(frame.pts, SWAPPED)

    proc = ProcessFrameTrack(FakeCamera(), slow_swap)
    try:
        _, timestamps = await collect(proc, 12)
    finally:
        proc.stop()

    assert len(set(timestamps)) == len(timestamps)
    assert all(b > a for a, b in pairwise(timestamps))


@pytest.mark.asyncio
async def test_a_hung_swap_does_not_wedge_the_connection(monkeypatch):
    monkeypatch.setattr(track_module, "FIRST_FRAME_TIMEOUT", 0.5)

    async def never_returns(frame):
        await asyncio.Event().wait()

    proc = ProcessFrameTrack(FakeCamera(), never_returns)
    try:
        started = asyncio.get_event_loop().time()
        frame = await asyncio.wait_for(proc.recv(), timeout=5)
        elapsed = asyncio.get_event_loop().time() - started
    finally:
        proc.stop()

    assert 0.4 < elapsed < 2.0
    assert value_of(frame) == RAW


@pytest.mark.asyncio
async def test_a_raising_swap_does_not_kill_the_track(monkeypatch):
    monkeypatch.setattr(track_module, "FIRST_FRAME_TIMEOUT", 0.3)
    calls = {"n": 0}

    async def flaky(frame):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise RuntimeError("boom")
        return make_frame(frame.pts, SWAPPED)

    proc = ProcessFrameTrack(FakeCamera(), flaky)
    try:
        values, _ = await collect(proc, 15)
    finally:
        proc.stop()

    assert values[-1] == SWAPPED
    assert calls["n"] > 3


@pytest.mark.asyncio
async def test_keeps_up_when_the_swap_is_fast():
    async def fast(frame):
        await asyncio.sleep(0.005)
        return make_frame(frame.pts, SWAPPED)

    proc = ProcessFrameTrack(FakeCamera(), fast)
    try:
        values, timestamps = await collect(proc, 30)
    finally:
        proc.stop()

    assert set(values) == {SWAPPED}
    assert proc._repeated_count <= 2
    assert all(b > a for a, b in pairwise(timestamps))
