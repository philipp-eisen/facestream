"""facestream's ROI paste-back must match insightface's full-frame version.

The optimisation only restricts *where* the work happens; the blend itself has
to be the same, or swapped faces would composite differently.
"""

import cv2
import numpy as np
import pytest
from insightface.utils import face_align

from facestream.faceswap import paste_face_back


def upstream_paste_back(img, aimg, bgr_fake, M):
    """Verbatim from insightface's INSwapper.get(paste_back=True).

    Kept as-is, including the fake_diff work that upstream computes and then
    discards, so this stays a faithful reference.
    """
    target_img = img
    fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
    fake_diff = np.abs(fake_diff).mean(axis=2)
    fake_diff[:2, :] = 0
    fake_diff[-2:, :] = 0
    fake_diff[:, :2] = 0
    fake_diff[:, -2:] = 0
    IM = cv2.invertAffineTransform(M)
    img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32)
    bgr_fake = cv2.warpAffine(
        bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0
    )
    img_white = cv2.warpAffine(
        img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0
    )
    fake_diff = cv2.warpAffine(
        fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0
    )
    img_white[img_white > 20] = 255
    fthresh = 10
    fake_diff[fake_diff < fthresh] = 0
    fake_diff[fake_diff >= fthresh] = 255
    img_mask = img_white
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
    mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
    mask_size = int(np.sqrt(mask_h * mask_w))
    k = max(mask_size // 10, 10)
    img_mask = cv2.erode(img_mask, np.ones((k, k), np.uint8), iterations=1)
    fake_diff = cv2.dilate(fake_diff, np.ones((2, 2), np.uint8), iterations=1)
    k = max(mask_size // 20, 5)
    img_mask = cv2.GaussianBlur(img_mask, (2 * k + 1, 2 * k + 1), 0)
    fake_diff = cv2.GaussianBlur(fake_diff, (11, 11), 0)
    img_mask /= 255
    fake_diff /= 255
    img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
    fake_merged = img_mask * bgr_fake + (1 - img_mask) * target_img.astype(np.float32)
    return fake_merged.astype(np.uint8)


def make_kps(cx, cy, size, angle_deg=0.0):
    """Five landmarks (eyes, nose, mouth corners) for a face of `size` px."""
    base = (
        np.array(
            [[-0.30, -0.20], [0.30, -0.20], [0.0, 0.05], [-0.22, 0.30], [0.22, 0.30]],
            dtype=np.float32,
        )
        * size
    )
    theta = np.deg2rad(angle_deg)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    return base @ rotation.T + np.array([cx, cy], dtype=np.float32)


def camera_like_frame(height, width, seed=0):
    """Smooth gradients, the way a real camera frame looks.

    Pixel noise would exaggerate sub-pixel resampling differences well beyond
    anything visible on an actual video frame.
    """
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (height // 40, width // 40, 3), dtype=np.uint8)
    frame = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(frame, (21, 21), 0)


CASES = [
    ("centred", (720, 1280), 640, 360, 200, 0),
    ("small face", (720, 1280), 640, 360, 80, 0),
    ("large face", (720, 1280), 640, 360, 420, 0),
    ("rotated", (720, 1280), 640, 360, 200, 25),
    ("left edge", (720, 1280), 60, 360, 200, 0),
    ("clipped top", (720, 1280), 640, 30, 200, 0),
    ("clipped bottom right", (720, 1280), 1250, 700, 220, -15),
    ("1080p", (1080, 1920), 960, 540, 300, 0),
    ("480p", (480, 640), 320, 240, 160, 10),
]


@pytest.mark.parametrize(
    "name,shape,cx,cy,size,angle", CASES, ids=[case[0] for case in CASES]
)
def test_matches_upstream(name, shape, cx, cy, size, angle):
    frame = camera_like_frame(*shape)
    kps = make_kps(cx, cy, size, angle)
    aligned, transform = face_align.norm_crop2(frame, kps, 128)

    rng = np.random.default_rng(1)
    swapped = cv2.GaussianBlur(
        rng.integers(0, 255, (128, 128, 3), dtype=np.uint8), (9, 9), 0
    )

    reference = upstream_paste_back(frame.copy(), aligned, swapped.copy(), transform)
    ours = paste_face_back(frame.copy(), swapped.copy(), transform)

    difference = np.abs(reference.astype(np.int16) - ours.astype(np.int16))
    # Only sub-pixel resampling rounding should differ: the ROI warp uses a
    # translated matrix, whose constant term rounds differently in OpenCV's
    # fixed-point arithmetic.
    assert difference.max() <= 4, f"{name}: max difference {difference.max()}"
    assert difference.mean() < 0.01, f"{name}: mean difference {difference.mean()}"


def test_leaves_frame_alone_when_the_face_lands_outside():
    frame = camera_like_frame(720, 1280)
    original = frame.copy()
    kps = make_kps(-4000, -4000, 200)
    _, transform = face_align.norm_crop2(frame, kps, 128)
    swapped = np.full((128, 128, 3), 255, dtype=np.uint8)

    result = paste_face_back(frame, swapped, transform)
    assert np.array_equal(result, original)
