import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import insightface
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from insightface.app.common import Face
from insightface.model_zoo.inswapper import INSwapper

from facestream import config
from facestream.constants import MODEL_CACHE_DIR

logger = logging.getLogger(__name__)

# Keypoints of a roughly centred, ~200px face. Only used to warm up the CUDA
# kernels, the actual positions don't matter.
_WARMUP_KPS = np.array(
    [[560.0, 330.0], [660.0, 330.0], [610.0, 390.0], [570.0, 450.0], [650.0, 450.0]],
    dtype=np.float32,
)


class FaceTracker:
    """Per-stream detection state.

    One ``FaceSwap`` is shared by every session on a container, so the last
    known face position has to live outside it or two streams would steer each
    other's search window.
    """

    __slots__ = ("last_bbox", "last_shape")

    def __init__(self):
        self.last_bbox: np.ndarray | None = None
        self.last_shape: tuple[int, int] | None = None

    def reset(self):
        self.last_bbox = None

    def note_frame_shape(self, shape: tuple[int, int]):
        """Drop the tracked box when the frame size changes.

        Browsers rescale mid-stream as their bandwidth estimate moves, which
        leaves the remembered box in the previous resolution's coordinates.
        """
        if self.last_shape != shape:
            self.last_shape = shape
            self.reset()


def paste_face_back(
    target_img: np.ndarray, bgr_fake: np.ndarray, transform: np.ndarray
) -> np.ndarray:
    """Composite the swapped face back into the frame, inside its bounding box only.

    Two changes versus ``INSwapper.get(..., paste_back=True)``:

    * The warp and the mask blur run on a crop around the face rather than on
      the whole frame. At 720p with a typical webcam framing the crop is about a
      fifth of the frame.
    * Upstream's ``fake_diff`` mask is skipped. It costs a full-frame warp, a
      threshold, a dilate and a blur, and is then thrown away -- upstream
      assigns ``img_mask = img_white`` immediately afterwards.

    The blend mask itself is unchanged: the crop is padded so the erode and blur
    kernels see the same neighbourhood they would at full frame, and those
    kernels are sized from the same measurement. Output matches upstream to
    within sub-pixel resampling rounding (<= 3/255 on a handful of edge pixels).

    Args:
        target_img: the full frame, BGR uint8. Modified in place.
        bgr_fake: the swapped face, as returned with ``paste_back=False``.
        transform: the 2x3 affine mapping frame coordinates to the aligned crop.
    """
    frame_h, frame_w = target_img.shape[:2]
    face_h, face_w = bgr_fake.shape[:2]

    inverse = cv2.invertAffineTransform(transform)

    # Where the aligned crop lands in the frame.
    corners = np.array(
        [[0, 0], [face_w, 0], [face_w, face_h], [0, face_h]], dtype=np.float32
    )
    mapped = corners @ inverse[:, :2].T + inverse[:, 2]
    min_x, min_y = mapped.min(axis=0)
    max_x, max_y = mapped.max(axis=0)

    # Pad the crop so the erode/blur kernels below see the same neighbourhood
    # they would in a full-frame pass. The blur radius works out at roughly
    # face_extent/20, so this leaves a comfortable margin of error; the erode
    # only ever shrinks the mask.
    face_extent = int(np.sqrt(max(max_x - min_x, 1.0) * max(max_y - min_y, 1.0)))
    margin = max(face_extent // 8, 24)

    x0 = max(int(np.floor(min_x)) - margin, 0)
    y0 = max(int(np.floor(min_y)) - margin, 0)
    x1 = min(int(np.ceil(max_x)) + margin, frame_w)
    y1 = min(int(np.ceil(max_y)) + margin, frame_h)
    if x1 <= x0 or y1 <= y0:
        return target_img

    roi_w, roi_h = x1 - x0, y1 - y0

    # Warp straight into region-of-interest coordinates.
    roi_transform = inverse.copy()
    roi_transform[0, 2] -= x0
    roi_transform[1, 2] -= y0

    fake_roi = cv2.warpAffine(bgr_fake, roi_transform, (roi_w, roi_h), borderValue=0.0)
    mask = cv2.warpAffine(
        np.full((face_h, face_w), 255, dtype=np.float32),
        roi_transform,
        (roi_w, roi_h),
        borderValue=0.0,
    )
    mask[mask > 20] = 255

    mask_rows, mask_cols = np.where(mask == 255)
    if mask_rows.size == 0:
        return target_img
    mask_size = int(
        np.sqrt(
            (mask_rows.max() - mask_rows.min()) * (mask_cols.max() - mask_cols.min())
        )
    )

    erode_k = max(mask_size // 10, 10)
    mask = cv2.erode(mask, np.ones((erode_k, erode_k), np.uint8), iterations=1)

    blur_k = max(mask_size // 20, 5)
    mask = cv2.GaussianBlur(mask, (2 * blur_k + 1, 2 * blur_k + 1), 0)
    mask = (mask / 255.0)[:, :, None]

    roi = target_img[y0:y1, x0:x1]
    target_img[y0:y1, x0:x1] = (mask * fake_roi + (1 - mask) * roi).astype(np.uint8)
    return target_img


class FaceSwap:
    def _download_faceswap_model(self):
        hf_hub_download(
            "hacksider/deep-live-cam",
            "inswapper_128_fp16.onnx",
            local_dir=MODEL_CACHE_DIR,
        )

    def _providers(self):
        return [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": torch.cuda.current_device(),
                    # The default is EXHAUSTIVE, which benchmarks every cuDNN
                    # convolution algorithm the first time it sees an input
                    # shape. That shows up as a multi-second stall on the first
                    # frames and again whenever the frame size changes.
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "do_copy_in_default_stream": True,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                },
            )
        ]

    def __init__(self):
        self._download_faceswap_model()

        providers = self._providers()

        self.faceswap_model: INSwapper = insightface.model_zoo.get_model(  # pyright: ignore[reportAttributeAccessIssue]
            str(Path(MODEL_CACHE_DIR) / "inswapper_128_fp16.onnx"),
            providers=providers,
        )

        self.faceanalysis = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=MODEL_CACHE_DIR,
            providers=providers,
        )

        self.det_size = (config.DET_SIZE, config.DET_SIZE)
        self.faceanalysis.prepare(
            ctx_id=0, det_thresh=config.DET_THRESH, det_size=self.det_size
        )
        self.det_model = self.faceanalysis.det_model

        # A per-call detector input size only works if the ONNX graph has a
        # dynamic input shape (buffalo_l's det_10g does). If it is fixed, fall
        # back to whatever the model was prepared with.
        self._det_is_dynamic = isinstance(self.det_model.input_shape[2], str)
        if not self._det_is_dynamic:
            logger.info("Detector has a fixed input shape, ROI tracking disabled")
        self.track_det_size = (config.TRACK_DET_SIZE, config.TRACK_DET_SIZE)

        # One dedicated worker so frames are swapped strictly one at a time.
        # Handing this to the default executor lets several frames enter
        # onnxruntime at once, which trades latency for nothing.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="faceswap"
        )

    # ----------------------------------------------------------------- #
    # Detection
    # ----------------------------------------------------------------- #

    def _detect(self, img: np.ndarray, det_size: tuple[int, int] | None):
        input_size = det_size if self._det_is_dynamic else None
        return self.det_model.detect(img, input_size=input_size, metric="default")

    @staticmethod
    def _largest(bboxes: np.ndarray) -> int:
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        return int(np.argmax(areas))

    @staticmethod
    def _tracked_roi(
        bbox: np.ndarray, frame_shape: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        height, width = frame_shape
        x0, y0, x1, y1 = bbox
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        half_w = (x1 - x0) * config.TRACK_ROI_SCALE / 2
        half_h = (y1 - y0) * config.TRACK_ROI_SCALE / 2
        return (
            max(int(center_x - half_w), 0),
            max(int(center_y - half_h), 0),
            min(int(center_x + half_w), width),
            min(int(center_y + half_h), height),
        )

    def _detect_target(self, frame: np.ndarray, tracker: FaceTracker):
        """Locate the face to replace, returning ``(bbox, kps)`` or ``None``.

        Only the detector runs here. ``FaceAnalysis.get()`` would additionally
        run the two landmark models, gender/age and the ArcFace recogniser on
        every detected face -- four model invocations per frame whose output the
        swapper never reads. It only needs the detector's keypoints.

        When the previous frame had a face, the search is a tight crop around it
        fed to the detector at a smaller input size. The detector always resizes
        its input to a fixed square, so a smaller square is the only thing that
        actually makes detection cheaper -- and on a crop it still resolves the
        face at higher effective detail than a full-frame pass would.
        """
        tracker.note_frame_shape(frame.shape[:2])

        if (
            config.FACE_TRACKING
            and self._det_is_dynamic
            and tracker.last_bbox is not None
        ):
            x0, y0, x1, y1 = self._tracked_roi(tracker.last_bbox, frame.shape[:2])
            crop = frame[y0:y1, x0:x1]
            if crop.size:
                bboxes, kpss = self._detect(crop, self.track_det_size)
                if len(bboxes) and kpss is not None:
                    index = self._largest(bboxes)
                    offset = np.array([x0, y0], dtype=np.float32)
                    bbox = bboxes[index][:4] + np.tile(offset, 2)
                    tracker.last_bbox = bbox
                    return bbox, kpss[index] + offset

        bboxes, kpss = self._detect(frame, self.det_size)
        if not len(bboxes) or kpss is None:
            tracker.last_bbox = None
            return None

        index = self._largest(bboxes)
        tracker.last_bbox = bboxes[index][:4]
        return bboxes[index][:4], kpss[index]

    def _get_one_face(self, frame: np.ndarray) -> Face | None:
        """Full analysis of a still image, for the uploaded source face.

        Unlike the streaming path this does need the recogniser: the swapper
        drives its output off the source face's embedding.
        """
        faces = self.faceanalysis.get(frame)
        if not faces:
            return None
        return max(faces, key=lambda face: np.prod(face.bbox[2:4] - face.bbox[0:2]))

    # ----------------------------------------------------------------- #
    # Swapping
    # ----------------------------------------------------------------- #

    def _swap_face(self, target: np.ndarray, source_face: dict, tracker: FaceTracker):
        detected = self._detect_target(target, tracker)
        if detected is None:
            return target

        bbox, kps = detected
        target_face = Face(bbox=bbox, kps=kps, det_score=1.0)
        if not isinstance(source_face, Face):
            source_face = Face(source_face)

        if not config.FAST_PASTE_BACK:
            return self.faceswap_model.get(
                target, target_face, source_face, paste_back=True
            )

        bgr_fake, transform = self.faceswap_model.get(
            target, target_face, source_face, paste_back=False
        )
        return paste_face_back(target, bgr_fake, transform)

    def _warmup(self, width: int = 1280, height: int = 720):
        """Run one synthetic frame through the pipeline.

        cuDNN algorithm selection and CUDA context setup happen on first use, so
        without this the first few seconds of a stream are noticeably slow.
        """
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        self._detect(frame, self.det_size)
        if self._det_is_dynamic:
            self._detect(frame[:360, :360], self.track_det_size)

        source_face = Face(embedding=np.random.randn(512).astype(np.float32))
        target_face = Face(bbox=np.array([540.0, 300.0, 680.0, 480.0]), kps=_WARMUP_KPS)
        bgr_fake, transform = self.faceswap_model.get(
            frame, target_face, source_face, paste_back=False
        )
        paste_face_back(frame, bgr_fake, transform)

    # ----------------------------------------------------------------- #
    # Async entry points
    # ----------------------------------------------------------------- #

    def warmup(self):
        """Blocking warmup, on the same worker thread that will process frames."""
        self._executor.submit(self._warmup).result()

    async def get_one_face(self, frame: np.ndarray) -> Face | None:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._get_one_face, frame)

    async def swap_face(
        self, frame: np.ndarray, source_face: dict, tracker: FaceTracker
    ):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._swap_face, frame, source_face, tracker
        )
