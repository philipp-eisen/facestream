import asyncio
import base64
import json
import logging
import os
from functools import partial

import cv2
import fastapi
import modal
import numpy as np
from aioice.candidate import Candidate
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.rtcicetransport import candidate_from_aioice
from av.video.frame import VideoFrame

from facestream import config, turn
from facestream.constants import (
    MODEL_CACHE_DIR,
    SECRET_KEY_TURN_API_TOKEN,
    SECRET_KEY_TURN_TOKEN_ID,
)
from facestream.faceswap import FaceSwap, FaceTracker
from facestream.track import ProcessFrameTrack

logger = logging.getLogger(__name__)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsm6", "libxext6")
    .workdir("/root")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir("web", remote_path="/web")
)

app = modal.App(name="facestream", image=web_image)


def turn_secrets() -> list[modal.Secret]:
    """The Cloudflare TURN secret, when TURN is switched on.

    Naming a Modal secret that doesn't exist fails the deploy, so this is opt-in
    through FACESTREAM_TURN rather than something you uncomment in the source.
    """
    if not config.TURN_ENABLED:
        return []

    logger.info("Cloudflare TURN enabled, expecting the 'facestream' secret")
    return [
        modal.Secret.from_name(
            "facestream",
            required_keys=[SECRET_KEY_TURN_TOKEN_ID, SECRET_KEY_TURN_API_TOKEN],
        )
    ]


# All of these are tunable via FACESTREAM_* environment variables at deploy
# time -- see facestream/config.py and the README.
@app.cls(
    gpu=config.GPU,
    scaledown_window=config.SCALEDOWN_WINDOW,
    cpu=config.CPU,
    timeout=config.TIMEOUT,
    min_containers=config.MIN_CONTAINERS,
    buffer_containers=config.BUFFER_CONTAINERS,
    max_containers=config.MAX_CONTAINERS,
    region=config.REGION,
    # Forward the runtime knobs so the values set on the deploying machine are
    # the ones the container actually reads.
    env=config.runtime_env(),
    volumes={
        MODEL_CACHE_DIR: modal.Volume.from_name(
            "facestream-model-cache", create_if_missing=True
        )
    },
    # Empty unless FACESTREAM_TURN=1, which needs a Cloudflare TURN server --
    # required for most cellular networks. (See README.md for more information)
    secrets=turn_secrets(),
)
@modal.concurrent(max_inputs=config.MAX_CONCURRENT_INPUTS)
class Main:
    @modal.enter()
    def load(self):
        logger.info("Starting with config: %s", config.describe())
        self.faceswap = FaceSwap()
        # Pay for cuDNN algorithm selection now rather than during the first
        # seconds of somebody's stream.
        self.faceswap.warmup()
        self.turn_token_id = os.environ.get(SECRET_KEY_TURN_TOKEN_ID)
        self.turn_api_token = os.environ.get(SECRET_KEY_TURN_API_TOKEN)

    async def process_frame(
        self, frame: VideoFrame, source_face: dict, tracker: FaceTracker
    ):
        out = await self.faceswap.swap_face(
            frame.to_ndarray(format="bgr24"), source_face, tracker
        )

        new_frame = VideoFrame.from_ndarray(out, format="bgr24")  # pyright: ignore[reportArgumentType]
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

    def get_on_track_listener(self, pc: RTCPeerConnection, source_face):
        # Detection state is per stream, not per container.
        tracker = FaceTracker()

        def on_track(track: MediaStreamTrack):
            logger.info("Track received: %s", track.kind)
            if track.kind == "video":
                logger.info("Adding video track")
                new_track = ProcessFrameTrack(
                    track,
                    partial(
                        self.process_frame, source_face=source_face, tracker=tracker
                    ),
                )
                pc.addTrack(new_track)
            elif track.kind == "audio":
                # Send back the same audio track -- could be
                # fun to add
                logger.info("Adding audio track")
                pc.addTrack(track)

            @track.on("ended")
            async def on_ended():
                logger.info("Track ended: %s", track.kind)

        return on_track

    async def get_ice_servers(self):
        # Without a TURN server the stream will most likely not work on
        # cellular networks. See facestream/turn.py.
        return await turn.get_ice_servers(self.turn_token_id, self.turn_api_token)

    @modal.asgi_app()
    def web(self):
        web_app = fastapi.FastAPI()
        web_app.add_route(
            "/", lambda _: fastapi.responses.FileResponse("/web/index.html")
        )
        web_app.add_route(
            "/favicon.ico", lambda _: fastapi.responses.FileResponse("/web/favicon.ico")
        )

        @web_app.get("/healthz")
        def healthz():
            return {"status": "ok", "config": config.describe()}

        @web_app.websocket("/ws")
        async def ws(websocket: fastapi.WebSocket):
            await websocket.accept()
            pc = None
            try:
                pc = RTCPeerConnection()
                source_face = None
                data = None

                while True:
                    try:
                        # The client heartbeats every 15s. Dropping a peer that
                        # has gone quiet is what makes a long session timeout
                        # safe: a browser tab that dies without closing its
                        # socket can't hold a GPU container open for an hour.
                        raw = await asyncio.wait_for(
                            websocket.receive_text(), timeout=config.WS_IDLE_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        logger.info(
                            "No message for %ss. Closing websocket.",
                            config.WS_IDLE_TIMEOUT,
                        )
                        await websocket.close(code=1000)
                        return

                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.error("Received invalid JSON: %s", raw)
                        continue

                    if data.get("type") == "upload_image":
                        logger.info("Received image. Processing...")
                        image = data["image"]

                        image_bytes = base64.b64decode(image)
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        source_face = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        source_face = await self.faceswap.get_one_face(source_face)

                        if source_face is None:
                            logger.info("No face found in the uploaded image")
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "message": "No face found in that image. Try another one.",
                                }
                            )
                            continue

                        logger.info("Got source face")
                        pc.remove_all_listeners()
                        pc.add_listener(
                            "track", self.get_on_track_listener(pc, source_face)
                        )
                        await websocket.send_json(
                            {
                                "type": "readyForStream",
                                "iceServers": await self.get_ice_servers(),
                            }
                        )

                    elif data.get("type") == "offer":
                        if source_face is None:
                            raise Exception("Invalid state: source face not uploaded")

                        await pc.setRemoteDescription(
                            RTCSessionDescription(sdp=data["sdp"], type="offer")
                        )
                        answer = await pc.createAnswer()
                        if answer is None:
                            raise Exception("Failed to create answer")
                        await pc.setLocalDescription(answer)

                        await websocket.send_json(
                            {"type": "answer", "sdp": pc.localDescription.sdp}
                        )

                    elif data.get("type") == "candidate":
                        if source_face is None:
                            raise Exception("Source face needs to be uploaded first")

                        candidate_data = data["candidate"]
                        if candidate_data:
                            candidate = Candidate.from_sdp(candidate_data)
                            rtc_ice_candidate = candidate_from_aioice(candidate)
                            rtc_ice_candidate.sdpMid = data["sdpMid"]
                            rtc_ice_candidate.sdpMLineIndex = data["sdpMLineIndex"]

                            await pc.addIceCandidate(rtc_ice_candidate)

                    elif data.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})

            except fastapi.WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                await websocket.close(code=1000)
                return
            except modal.exception.FunctionTimeoutError:
                logger.info("Function timeout. Closing websocket.")
                await websocket.close(code=1000)
                return

            except Exception as e:
                logger.exception("Error in websocket")
                await websocket.close(code=1000)
                raise e
            finally:
                if pc is not None:
                    await pc.close()
                    pc = None

        return web_app
