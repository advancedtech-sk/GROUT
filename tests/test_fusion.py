"""Fusion utility tests (written before implementation)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return np.log(p / (1.0 - p))


class TestModes:
    def setup_method(self):
        from fuse_probs import fuse_prob_maps
        self.fuse = fuse_prob_maps
        self.a = np.array([[0.0, 0.5], [1.0, 0.9]], np.float32)
        self.b = np.array([[1.0, 0.5], [0.2, 0.0]], np.float32)

    def test_gate(self):
        out = self.fuse(self.a, self.b, mode="gate")
        assert out.dtype == np.float32
        # clamped inputs: 0 -> 1e-6, 1 -> 1-1e-6
        assert out[0, 0] == pytest.approx(1e-6 * (1 - 1e-6), rel=1e-3)
        assert out[0, 1] == pytest.approx(0.25, abs=1e-6)
        assert out[1, 0] == pytest.approx((1 - 1e-6) * 0.2, rel=1e-5)

    def test_logit_prior_half(self):
        out = self.fuse(self.a, self.b, mode="logit", prior=0.5)
        eps = 1e-6
        a = np.clip(self.a, eps, 1 - eps)
        b = np.clip(self.b, eps, 1 - eps)
        expect = sig(logit(a) + logit(b) - logit(np.float32(0.5)))
        assert np.allclose(out, expect, atol=1e-5)
        # symmetric evidence at 0.5,0.5 stays 0.5
        assert out[0, 1] == pytest.approx(0.5, abs=1e-6)

    def test_min(self):
        out = self.fuse(self.a, self.b, mode="min")
        assert out[1, 0] == pytest.approx(0.2, abs=1e-6)
        assert out[0, 0] <= 1e-6 + 1e-9

    def test_mode_required(self):
        with pytest.raises(TypeError):
            self.fuse(self.a, self.b)  # no default mode

    def test_extremes_finite(self):
        z = np.zeros((4, 4), np.float32)
        o = np.ones((4, 4), np.float32)
        for m in ("gate", "logit", "min"):
            out = self.fuse(z, o, mode=m)
            assert np.isfinite(out).all()
            assert (out >= 0).all() and (out <= 1).all()


class TestResample:
    def test_b_resampled_to_a(self):
        from fuse_probs import fuse_prob_maps
        a = np.full((8, 8), 0.5, np.float32)
        b = np.full((4, 4), 0.25, np.float32)
        out = fuse_prob_maps(a, b, mode="gate")
        assert out.shape == (8, 8)
        assert out.dtype == np.float32
        assert np.allclose(out, 0.125, atol=1e-6)

    def test_nearest_flag(self):
        from fuse_probs import fuse_prob_maps
        a = np.full((4, 4), 1.0, np.float32)
        b = np.array([[0.0, 1.0], [1.0, 0.0]], np.float32)
        out = fuse_prob_maps(a, b, mode="min", resample="nearest")
        # nearest keeps hard blocks
        vals = np.unique(np.round(out, 5))
        assert len(vals) <= 2


class TestIO:
    def test_load_png8_png16_npy(self, tmp_path):
        import cv2
        from fuse_probs import load_prob_map
        p = np.linspace(0, 1, 256).reshape(16, 16).astype(np.float32)
        np.save(tmp_path / "m.npy", p)
        cv2.imwrite(str(tmp_path / "m8.png"),
                    np.round(p * 255).astype(np.uint8))
        cv2.imwrite(str(tmp_path / "m16.png"),
                    np.round(p * 65535).astype(np.uint16))
        for f, tol in (("m.npy", 0), ("m8.png", 1 / 255), ("m16.png", 1 / 65535)):
            m = load_prob_map(tmp_path / f)
            assert m.dtype == np.float32
            assert np.abs(m - p).max() <= tol / 2 + 1e-7
