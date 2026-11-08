from __future__ import annotations

import numpy as np

from shared import image_ops


def test_load_depth_decodes_encoded_uint8_without_numpy_overflow(monkeypatch):
    encoded = np.array(
        [[[0, 1, 44], [0, 125, 1], [0, 255, 255]]], dtype=np.uint8
    )
    monkeypatch.setattr(image_ops.cv2, "imread", lambda *_args, **_kwargs: encoded)

    depth = image_ops.load_depth("encoded.png")

    assert depth.dtype == np.uint16
    assert depth.tolist() == [[300, 0, 65535]]


def test_load_depth_preserves_native_uint16(monkeypatch):
    native = np.array([[0, 250, 1000]], dtype=np.uint16)
    monkeypatch.setattr(image_ops.cv2, "imread", lambda *_args, **_kwargs: native)

    depth = image_ops.load_depth("native.png")

    assert depth is native
