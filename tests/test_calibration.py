"""Calibration hook tests (written before implementation)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestTemperature:
    def test_t1_noop(self):
        from calibrate import apply_temperature
        logits = np.array([[-3.0, 0.0], [2.0, 5.0]], np.float32)
        p1 = apply_temperature(logits, 1.0)
        base = 1.0 / (1.0 + np.exp(-logits))
        assert np.allclose(p1, base, atol=1e-7)

    @pytest.mark.parametrize("t", [0.25, 0.5, 2.0, 7.3])
    def test_ranking_preserved(self, t):
        """Monotone transform: p sorted by logits must be non-decreasing.
        (argsort equality would be too strict: small T saturates float32
        sigmoid to exact 0/1, creating ties with arbitrary argsort order.)"""
        from calibrate import apply_temperature
        rng = np.random.default_rng(0)
        logits = rng.normal(0, 3, 1000).astype(np.float32)
        p = apply_temperature(logits, t)
        assert np.all(np.diff(p[np.argsort(logits)]) >= 0)

    def test_fit_recovers_scale(self):
        """Generate miscalibrated logits (true T=2); the fit should find
        T near 2 and reduce NLL vs T=1."""
        import torch
        from calibrate import fit_temperature, nll
        rng = np.random.default_rng(42)
        true_logits = rng.normal(0, 2, 20000).astype(np.float32)
        y = (rng.random(20000) < 1 / (1 + np.exp(-true_logits))).astype(np.float32)
        overconfident = true_logits * 2.0    # model reports 2x logits
        t = fit_temperature(overconfident, y)
        assert 1.5 < t < 2.6
        assert nll(overconfident, y, t) < nll(overconfident, y, 1.0) - 1e-4


class TestInferenceHook:
    def test_segmenter_accepts_temperature(self):
        ckpt = Path(__file__).resolve().parents[1] / "checkpoints" / \
            "grout_b3_zeroshot_v1.pth"
        if not ckpt.exists():
            pytest.skip("checkpoint missing")
        from inference import GroutSegmenter
        s1 = GroutSegmenter(str(ckpt), device="cpu", temperature=1.0)
        s2 = GroutSegmenter(str(ckpt), device="cpu", temperature=2.0)
        img = np.full((512, 512, 3), 128, np.uint8)
        p1, _ = s1.predict(img, apply_post_processing=False)
        p2, _ = s2.predict(img, apply_post_processing=False)
        # T>1 pulls probabilities toward 0.5, ranking (weakly) preserved
        assert np.abs(p2 - 0.5).mean() < np.abs(p1 - 0.5).mean()
        flat1, flat2 = p1.ravel(), p2.ravel()
        idx = np.random.default_rng(0).choice(flat1.size, 2000, replace=False)
        order = np.argsort(flat1[idx], kind="stable")
        assert np.all(np.diff(flat2[idx][order]) >= -1e-7)
