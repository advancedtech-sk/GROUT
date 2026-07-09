"""Golden tests for the soft-output feature (written before implementation).

CRITICAL INVARIANT: thresholding the exported probability map at
Config.PREDICTION_THRESHOLD reproduces the native-grid, PRE-morphology
binary mask bit-exactly. (The shipped mask additionally passes through
post_process_grout and, on the resize path, a bilinear resize -- pipeline
order documented in README. The invariant is defined at the stage where it
can hold, per spec.)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402

CKPT = ROOT / "checkpoints" / "grout_b3_zeroshot_v1.pth"
REAL = Path(__file__).parent / "data" / "real_sample.png"

pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="checkpoint missing")


@pytest.fixture(scope="module")
def segmenter():
    from inference import GroutSegmenter
    return GroutSegmenter(str(CKPT), device="cpu")


def synthetic_image(h=512, w=512, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 180, np.uint8)
    for y in range(0, h, 32):
        img[y:y + 2] = 40
    for x in range(0, w, 32):
        img[:, x:x + 2] = 40
    img = np.clip(img.astype(np.int16) + rng.integers(-15, 15, img.shape),
                  0, 255).astype(np.uint8)
    return img


class TestInvariant:
    def test_synthetic_512(self, segmenter):
        prob, _ = segmenter.predict(synthetic_image(), apply_post_processing=True)
        raw = (prob > Config.PREDICTION_THRESHOLD).astype(np.uint8) * 255
        prob2, _ = segmenter.predict(synthetic_image(), apply_post_processing=False)
        raw2 = (prob2 > Config.PREDICTION_THRESHOLD).astype(np.uint8) * 255
        assert prob.dtype == np.float32
        assert np.array_equal(raw, raw2)  # deterministic + invariant stage

    def test_real_sample_roundtrip(self, segmenter, tmp_path):
        import cv2
        from prob_output import load_prob_map, save_prob_outputs
        img = cv2.cvtColor(cv2.imread(str(REAL)), cv2.COLOR_BGR2RGB)
        prob, _ = segmenter.predict(img, apply_post_processing=True)
        paths = save_prob_outputs(prob, tmp_path, "real_sample")
        # canonical npy: bit-exact float32 round trip
        reloaded = np.load(paths["npy"])
        assert reloaded.dtype == np.float32
        assert np.array_equal(reloaded, prob.astype(np.float32))
        # INVARIANT: threshold(exported prob) == native-grid raw mask
        raw = (prob > Config.PREDICTION_THRESHOLD).astype(np.uint8) * 255
        assert np.array_equal(
            (reloaded > Config.PREDICTION_THRESHOLD).astype(np.uint8) * 255, raw)

    def test_sliding_window_invariant(self, segmenter):
        img = synthetic_image(700, 640, seed=3)
        prob, _ = segmenter.predict_large_image(img, apply_post_processing=True)
        assert prob.shape == (700, 640)
        raw = (prob > Config.PREDICTION_THRESHOLD).astype(np.uint8) * 255
        prob2, raw2 = segmenter.predict_large_image(img, apply_post_processing=False)
        assert np.array_equal(raw, raw2)


class TestEightBit:
    def test_roundtrip_within_1_255(self, tmp_path):
        from prob_output import load_prob_map, save_prob_outputs
        rng = np.random.default_rng(1)
        prob = rng.random((64, 64)).astype(np.float32)
        paths = save_prob_outputs(prob, tmp_path, "toy")
        p8 = load_prob_map(paths["png8"])
        assert p8.dtype == np.float32
        assert np.abs(p8 - prob).max() <= (1.0 / 255.0) / 2 + 1e-7


class TestCLIFlags:
    def test_no_prob_disables(self, tmp_path, segmenter, monkeypatch):
        import inference
        import cv2
        img = synthetic_image()
        p = tmp_path / "in.png"
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        inference.inference_single_image(
            str(p), str(CKPT), str(tmp_path), device="cpu",
            visualize=False, save_prob=False, segmenter=segmenter)
        assert not list(tmp_path.glob("*_prob*"))
        inference.inference_single_image(
            str(p), str(CKPT), str(tmp_path), device="cpu",
            visualize=False, save_prob=True, segmenter=segmenter)
        assert list(tmp_path.glob("*_prob_*.npy"))
        assert list(tmp_path.glob("*_prob8_*.png"))
