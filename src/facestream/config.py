"""Deployment and runtime tuning knobs.

Everything here is read from ``FACESTREAM_*`` environment variables so a
deployment can be tuned without editing code::

    FACESTREAM_GPU=H100 uv run modal deploy -m facestream.main

Two kinds of settings live here:

* *Deploy-time* settings (GPU, cpu, autoscaling, timeouts) are consumed by the
  ``@app.cls(...)`` decorator, which is evaluated on the machine running
  ``modal deploy``.
* *Runtime* settings (detector sizes, idle timeout, ...) are read inside the
  container. They are forwarded there via ``RUNTIME_ENV`` so that the value you
  set locally at deploy time is the value the container sees.
"""

import os

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


def _env_opt_str(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def _env_opt_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if not value:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
# Deploy-time settings (consumed by the @app.cls decorator)
# --------------------------------------------------------------------------- #

# Any GPU string Modal accepts: "T4", "L4", "A10G", "L40S", "A100-40GB",
# "A100-80GB", "H100", "H200", "B200".
#
# This pipeline is latency bound on small models (SCRFD detection at 640x640 and
# inswapper_128 at 128x128), not throughput bound on a large one, so the fastest
# card is not the biggest card. L40S is the sweet spot: Ada clocks and fp16
# throughput give the lowest per-frame latency of the sensibly priced options.
# Going above it mostly buys memory bandwidth this workload never uses.
GPU = _env_str("FACESTREAM_GPU", "L40S")

# WebRTC decode, colour conversion and VP8 encode all happen on the CPU and are
# a real part of the frame budget, so don't starve the container of cores.
CPU = _env_float("FACESTREAM_CPU", 8.0)

# Max lifetime of a single input. For an ASGI app an input is one HTTP request
# or one websocket connection, so this is the hard cap on how long a stream can
# run. The upstream default of 120s exists to protect the public demo's GPU
# budget; a private deployment streaming into OBS wants hours, guarded by
# WS_IDLE_TIMEOUT below instead.
TIMEOUT = _env_int("FACESTREAM_TIMEOUT", 3600)

# How long an idle container sticks around. Long enough that toggling an OBS
# scene reconnects instantly instead of paying another cold start.
SCALEDOWN_WINDOW = _env_int("FACESTREAM_SCALEDOWN_WINDOW", 300)

# Concurrent inputs per container. This must stay above 1: a live websocket
# occupies an input for the whole session, so a container with max_inputs=1
# could not even serve index.html while someone is streaming.
#
# At 2, a page load and one stream fit together, and the next viewer's
# websocket lands on a new container -- so in practice each stream gets a GPU
# to itself. Raise it to share a GPU between streams and spend less, at the
# cost of frame rate when more than one person is connected.
MAX_CONCURRENT_INPUTS = _env_int("FACESTREAM_MAX_CONCURRENT_INPUTS", 2)

# Keep containers warm to skip the ~20s cold start. Costs GPU time while idle,
# so it is opt-in.
MIN_CONTAINERS = _env_int("FACESTREAM_MIN_CONTAINERS", 0)
BUFFER_CONTAINERS = _env_int("FACESTREAM_BUFFER_CONTAINERS", 0)

# Cost ceiling: the maximum number of GPU containers that can run at once.
MAX_CONTAINERS = _env_opt_int("FACESTREAM_MAX_CONTAINERS")

# WebRTC latency is dominated by network round trips, so pinning the deployment
# near your viewers matters more than the GPU does. e.g. "us-east-1", "eu-west-1".
REGION = _env_opt_str("FACESTREAM_REGION")

# Relay media through Cloudflare's TURN service, which most cellular networks
# need. Requires a Modal secret named "facestream" holding TURN_TOKEN_ID and
# TURN_API_TOKEN; the deploy fails if it is missing, hence the opt-in.
TURN_ENABLED = _env_bool("FACESTREAM_TURN", False)

# --------------------------------------------------------------------------- #
# Runtime settings (read inside the container, forwarded via RUNTIME_ENV)
# --------------------------------------------------------------------------- #

# Detector input resolution used when we have to search the whole frame. The
# network input is always this size regardless of frame size, so it is the main
# lever on detection cost. 640 is the insightface default; 320 roughly quarters
# the detection cost at the price of missing small/distant faces.
DET_SIZE = _env_int("FACESTREAM_DET_SIZE", 640)

# Detector input resolution used when tracking a face we already found. The
# crop is tight around the last known face, so a much smaller input still
# resolves it at higher effective detail than a full-frame pass.
TRACK_DET_SIZE = _env_int("FACESTREAM_TRACK_DET_SIZE", 320)

# How far to expand the last bounding box when cropping the next frame. 2.2x
# tolerates a face moving about half its own width between frames.
TRACK_ROI_SCALE = _env_float("FACESTREAM_TRACK_ROI_SCALE", 2.2)

# Reuse the previous frame's face location to crop the search area. Turn off to
# run a full-frame detection on every frame.
FACE_TRACKING = _env_bool("FACESTREAM_FACE_TRACKING", True)

DET_THRESH = _env_float("FACESTREAM_DET_THRESH", 0.5)

# Composite the swapped face only inside its bounding box instead of running
# insightface's full-frame paste-back. Set to 0 to fall back to upstream's
# implementation.
FAST_PASTE_BACK = _env_bool("FACESTREAM_FAST_PASTE_BACK", True)

# Close a websocket that has gone quiet for this long. The client heartbeats
# every 15s, so this only fires on a genuinely dead peer -- which is what makes
# the long TIMEOUT above safe.
WS_IDLE_TIMEOUT = _env_int("FACESTREAM_WS_IDLE_TIMEOUT", 120)

# Log a frame timing summary every N frames. 0 disables it.
STATS_INTERVAL = _env_int("FACESTREAM_STATS_INTERVAL", 150)

# Frames buffered ahead of the swapper. 1 means "always work on the newest
# frame", which is what you want for low latency.
INPUT_QUEUE_SIZE = _env_int("FACESTREAM_INPUT_QUEUE_SIZE", 1)


def runtime_env() -> dict[str, str]:
    """Runtime settings to inject into the container as environment variables."""
    return {
        "FACESTREAM_DET_SIZE": str(DET_SIZE),
        "FACESTREAM_TRACK_DET_SIZE": str(TRACK_DET_SIZE),
        "FACESTREAM_TRACK_ROI_SCALE": str(TRACK_ROI_SCALE),
        "FACESTREAM_FACE_TRACKING": "1" if FACE_TRACKING else "0",
        "FACESTREAM_DET_THRESH": str(DET_THRESH),
        "FACESTREAM_FAST_PASTE_BACK": "1" if FAST_PASTE_BACK else "0",
        "FACESTREAM_WS_IDLE_TIMEOUT": str(WS_IDLE_TIMEOUT),
        "FACESTREAM_STATS_INTERVAL": str(STATS_INTERVAL),
        "FACESTREAM_INPUT_QUEUE_SIZE": str(INPUT_QUEUE_SIZE),
    }


def describe() -> dict[str, object]:
    """Human readable summary of the active configuration."""
    return {
        "gpu": GPU,
        "cpu": CPU,
        "timeout": TIMEOUT,
        "scaledown_window": SCALEDOWN_WINDOW,
        "max_concurrent_inputs": MAX_CONCURRENT_INPUTS,
        "min_containers": MIN_CONTAINERS,
        "region": REGION,
        "turn": TURN_ENABLED,
        "det_size": DET_SIZE,
        "track_det_size": TRACK_DET_SIZE,
        "face_tracking": FACE_TRACKING,
        "fast_paste_back": FAST_PASTE_BACK,
        "ws_idle_timeout": WS_IDLE_TIMEOUT,
    }
