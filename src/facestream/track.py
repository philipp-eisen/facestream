import asyncio
from typing import Awaitable, Callable

from aiortc import MediaStreamTrack
from av.video.frame import VideoFrame


class ProcessFrameTrack(MediaStreamTrack):
    """
    A MediaStreamTrack that processes frames while trying to maintain low latency
    (by dropping frames if necessary).
    """

    kind = "video"

    def __init__(
        self,
        track: MediaStreamTrack,
        process_frame: Callable[[VideoFrame], Awaitable[VideoFrame]],
    ):
        super().__init__()
        self.track = track
        self.input_queue = asyncio.Queue(maxsize=3)
        self.output_queue = asyncio.Queue(maxsize=1)
        self.processing_task = asyncio.create_task(self._processor())
        self.last_frame = None
        self.frame_counter = 0
        self.process_frame = process_frame

    async def recv(self):
        # Get original frame and feed to processor
        original_frame = await self.track.recv()
        self.frame_counter += 1

        try:
            self.input_queue.put_nowait((self.frame_counter, original_frame))
        except asyncio.QueueFull:
            # Drop oldest frame if queue full
            _ = await self.input_queue.get()
            self.input_queue.put_nowait((self.frame_counter, original_frame))

        try:
            _, processed_frame = self.output_queue.get_nowait()
            self.last_frame = processed_frame
            return processed_frame
        except asyncio.QueueEmpty:
            return self.last_frame if self.last_frame else original_frame

    async def _processor(self):
        while True:
            try:
                frame_num, frame = await asyncio.wait_for(
                    self.input_queue.get(), timeout=0.1
                )

                processed = await self.process_frame(frame)
                if not self.output_queue.empty():
                    _ = self.output_queue.get_nowait()
                self.output_queue.put_nowait((frame_num, processed))

            except (asyncio.QueueEmpty, asyncio.TimeoutError):
                continue

    def stop(self):
        self.processing_task.cancel()
        # Clear queues
        super().stop()
