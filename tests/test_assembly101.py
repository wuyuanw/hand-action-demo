from pathlib import Path

import numpy as np

from tools.prepare_assembly101 import interpolate_missing, parse_filter, slugify


def test_slugify():
    assert slugify("Pick Up") == "pick_up"
    assert slugify("screw chassis with screwdriver") == "screw_chassis_with_screwdriver"


def test_parse_filter_inline():
    assert parse_filter("pick up, put down, screw") == {"pick up", "put down", "screw"}


def test_interpolate_missing_landmarks():
    a = np.zeros((21, 3), dtype=np.float32)
    b = np.ones((21, 3), dtype=np.float32)
    out = interpolate_missing([a, None, b])
    assert out is not None
    assert out.shape == (3, 21, 3)
    assert np.allclose(out[1], 0.5)


def test_interpolate_all_missing():
    assert interpolate_missing([None, None]) is None
