import asyncio
from pathlib import Path

import insightface
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from insightface.app.common import Face
from insightface.model_zoo.inswapper import INSwapper

from facestream.constants import MODEL_CACHE_DIR


class FaceSwap:
    def _download_faceswap_model(self):
        hf_hub_download(
            "hacksider/deep-live-cam",
            "inswapper_128_fp16.onnx",
            local_dir=MODEL_CACHE_DIR,
        )

    def __init__(self):
        self._download_faceswap_model()

        self.faceswap_model: INSwapper = insightface.model_zoo.get_model(  # pyright: ignore[reportAttributeAccessIssue]
            str(Path(MODEL_CACHE_DIR) / "inswapper_128_fp16.onnx"),
            providers=[
                (
                    "CUDAExecutionProvider",
                    {"device_id": torch.cuda.current_device()},
                )
            ],
        )

        self.faceanalysis = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=MODEL_CACHE_DIR,
            providers=[
                (
                    "CUDAExecutionProvider",
                    {"device_id": torch.cuda.current_device()},
                )
            ],
        )

        self.faceanalysis.prepare(ctx_id=0, det_size=(640, 640))

    def _swap_face(self, target: np.ndarray, source_face: dict):
        source_face = Face(source_face)
        target_face = self._get_one_face(target)

        if target_face is None:
            return target

        return self.faceswap_model.get(
            target, target_face, source_face, paste_back=True
        )

    def _get_one_face(self, frame: np.ndarray) -> dict | None:
        try:
            faces = self.faceanalysis.get(frame)
            return min(faces, key=lambda x: x.bbox[0])
        except ValueError:
            return None

    async def get_one_face(self, frame: np.ndarray) -> dict | None:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._get_one_face, frame)
        return result

    async def swap_face(self, frame: np.ndarray, source_face: dict):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._swap_face, frame, source_face)
        return result
