"""Local stand-in for the Modal deployment.

Speaks the same websocket protocol as facestream.main and runs the real aiortc
peer connection and the real ProcessFrameTrack, but swaps the GPU face swap for
a cheap CPU tint so this can run without CUDA. Used to exercise the browser
client end to end.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

import fastapi
import numpy as np
import uvicorn
from aioice.candidate import Candidate
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.rtcicetransport import candidate_from_aioice
from av.video.frame import VideoFrame

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from facestream.track import ProcessFrameTrack

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stub")

app = fastapi.FastAPI()
received = {"messages": [], "pings": 0, "frames": 0}


@app.get("/")
def index():
    return fastapi.responses.FileResponse(
        Path(__file__).resolve().parents[1] / "web" / "index.html"
    )


@app.get("/received")
def get_received():
    return received


@app.get("/face.jpg")
def face():
    """A stand-in source image (the real presets live on Wikimedia)."""
    import cv2

    img = np.full((400, 400, 3), 200, dtype=np.uint8)
    cv2.circle(img, (200, 200), 120, (180, 160, 150), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return fastapi.responses.Response(buf.tobytes(), media_type="image/jpeg")


async def fake_swap(frame: VideoFrame) -> VideoFrame:
    array = frame.to_ndarray(format="bgr24")
    # Stand in for the swap: mark the frame so the test can tell processed
    # output apart from the raw camera feed.
    array[:, :, 2] = np.minimum(array[:, :, 2].astype(np.int16) + 80, 255).astype(
        np.uint8
    )
    received["frames"] += 1
    out = VideoFrame.from_ndarray(array, format="bgr24")
    out.pts = frame.pts
    out.time_base = frame.time_base
    return out


@app.websocket("/ws")
async def ws(websocket: fastapi.WebSocket):
    await websocket.accept()
    pc = RTCPeerConnection()
    try:
        while True:
            data = json.loads(await asyncio.wait_for(websocket.receive_text(), 60))
            kind = data.get("type")
            received["messages"].append(kind)

            if kind == "kill":
                # Simulate the server hanging up mid stream.
                await websocket.close(code=1000)
                return

            if kind == "ping":
                received["pings"] += 1
                await websocket.send_json({"type": "pong"})

            elif kind == "upload_image":
                logger.info("got image of %d base64 chars", len(data["image"]))
                if data.get("fail"):
                    await websocket.send_json(
                        {"type": "error", "message": "No face found in that image."}
                    )
                    continue

                def on_track(track: MediaStreamTrack):
                    logger.info("track: %s", track.kind)
                    if track.kind == "video":
                        pc.addTrack(ProcessFrameTrack(track, fake_swap))

                pc.remove_all_listeners()
                pc.add_listener("track", on_track)
                await websocket.send_json(
                    {
                        "type": "readyForStream",
                        "iceServers": [{"urls": "stun:stun.l.google.com:19302"}],
                    }
                )

            elif kind == "offer":
                await pc.setRemoteDescription(
                    RTCSessionDescription(sdp=data["sdp"], type="offer")
                )
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await websocket.send_json(
                    {"type": "answer", "sdp": pc.localDescription.sdp}
                )

            elif kind == "candidate":
                if data["candidate"]:
                    candidate = candidate_from_aioice(
                        Candidate.from_sdp(data["candidate"])
                    )
                    candidate.sdpMid = data["sdpMid"]
                    candidate.sdpMLineIndex = data["sdpMLineIndex"]
                    await pc.addIceCandidate(candidate)

    except Exception:
        logger.exception("ws ended")
    finally:
        await pc.close()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8123, log_level="warning")
