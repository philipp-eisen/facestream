import base64
import json
import logging
import os
from functools import partial

import aiohttp
import cv2
import fastapi
import modal
import numpy as np
from aioice.candidate import Candidate
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.rtcicetransport import candidate_from_aioice
from av.video.frame import VideoFrame

from facestream.constants import (
    MODEL_CACHE_DIR,
    SECRET_KEY_TURN_API_TOKEN,
    SECRET_KEY_TURN_TOKEN_ID,
)
from facestream.faceswap import FaceSwap
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


@app.cls(
    gpu="T4",
    container_idle_timeout=60,
    cpu=16,
    timeout=120,
    volumes={
        MODEL_CACHE_DIR: modal.Volume.from_name(
            "facestream-model-cache", create_if_missing=True
        )
    },
    # This secret is only required if you want to use a Cloudflare TURN server
    # - which is required for most cellular networks.
    # (See README.md for more information)
    #
    # secrets=[
    #     modal.Secret.from_name(
    #         "facestream",
    #         required_keys=[SECRET_KEY_TURN_TOKEN_ID, SECRET_KEY_TURN_API_TOKEN],
    #     )
    # ],
    allow_concurrent_inputs=4,
)
class Main:
    @modal.enter()
    def load(self):
        self.faceswap = FaceSwap()
        self.turn_token_id = os.environ.get(SECRET_KEY_TURN_TOKEN_ID)
        self.turn_api_token = os.environ.get(SECRET_KEY_TURN_API_TOKEN)

    async def process_frame(self, frame: VideoFrame, source_face: dict):
        out = await self.faceswap.swap_face(
            frame.to_ndarray(format="bgr24"), source_face
        )

        new_frame = VideoFrame.from_ndarray(out, format="bgr24")  # pyright: ignore[reportArgumentType]
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

    def get_on_track_listener(self, pc: RTCPeerConnection, source_face):
        def on_track(track: MediaStreamTrack):
            logger.info("Track received: %s", track.kind)
            if track.kind == "video":
                logger.info("Adding video track")
                new_track = ProcessFrameTrack(
                    track, partial(self.process_frame, source_face=source_face)
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

    async def _get_cloudflare_ice_servers(self):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://rtc.live.cloudflare.com/v1/turn/keys/{self.turn_token_id}/credentials/generate",
                headers={"Authorization": f"Bearer {self.turn_api_token}"},
                json={"ttl": 86400},
            ) as response:
                data = await response.json()
                return [data["iceServers"]]

    async def get_ice_servers(self):
        # If we have Cloudflare TURN credentials, use them
        # Otherwise, use Google STUN.

        # NOTE: without a TURN server the stream will most likely not
        # work on cellular networks.
        if self.turn_api_token and self.turn_token_id:
            logger.info("Using Cloudflare TURN servers")
            return await self._get_cloudflare_ice_servers()
        else:
            logger.info("Using Google STUN")
            return [{"urls": "stun:stun.l.google.com:19302"}]

    @modal.asgi_app()
    def web(self):
        web_app = fastapi.FastAPI()
        web_app.add_route(
            "/", lambda _: fastapi.responses.FileResponse("/web/index.html")
        )
        web_app.add_route(
            "/favicon.ico", lambda _: fastapi.responses.FileResponse("/web/favicon.ico")
        )

        @web_app.websocket("/ws")
        async def ws(websocket: fastapi.WebSocket):
            await websocket.accept()
            pc = None
            try:
                pc = RTCPeerConnection()

                while True:
                    try:
                        data = json.loads(await websocket.receive_text())
                    except json.JSONDecodeError:
                        logger.error("Received invalid JSON: %s", data)
                        continue

                    if data.get("type") == "upload_image":
                        logger.info("Received image. Processing...")
                        image = data["image"]

                        image_bytes = base64.b64decode(image)
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        source_face = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        source_face = await self.faceswap.get_one_face(source_face)

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
