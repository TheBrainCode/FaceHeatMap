import numpy as np
import pytest

from faceheatmap.config import HeatmapConfig
from faceheatmap.heatmap import HeatmapAccumulator, colorize, overlay_on_image
from faceheatmap.layout import BBox


def test_add_point_accumulates_weight():
    acc = HeatmapAccumulator(200, 100, HeatmapConfig(resolution_scale=1.0))
    acc.add_point(50, 50)
    acc.add_point(50, 50)
    assert acc.total_weight == pytest.approx(2.0)


def test_render_full_peak_is_near_the_dense_point():
    acc = HeatmapAccumulator(200, 100, HeatmapConfig(resolution_scale=1.0, sigma_px=3.0))
    for _ in range(20):
        acc.add_point(150, 20)
    heat = acc.render_full(normalize=True)
    assert heat.shape == (100, 200)
    peak_y, peak_x = np.unravel_index(np.argmax(heat), heat.shape)
    assert abs(peak_x - 150) <= 5
    assert abs(peak_y - 20) <= 5
    assert heat.max() == pytest.approx(1.0)


def test_render_full_empty_accumulator_is_all_zero():
    acc = HeatmapAccumulator(100, 100, HeatmapConfig())
    heat = acc.render_full()
    assert heat.max() == 0.0


def test_points_outside_frame_are_clamped_not_dropped():
    acc = HeatmapAccumulator(100, 50, HeatmapConfig(resolution_scale=1.0))
    acc.add_point(-999, -999)
    acc.add_point(999, 999)
    assert acc.total_weight == pytest.approx(2.0)


def test_render_region_focuses_on_the_requested_crop():
    acc = HeatmapAccumulator(200, 100, HeatmapConfig(resolution_scale=1.0, sigma_px=2.0))
    for _ in range(50):
        acc.add_point(20, 20)  # left half, dense
    acc.add_point(180, 20)  # right half, sparse

    left_region = BBox(0, 0, 100, 100)
    right_region = BBox(100, 0, 200, 100)

    left_heat = acc.render_region(left_region, normalize=False)
    right_heat = acc.render_region(right_region, normalize=False)

    assert left_heat.max() > right_heat.max() * 5
    assert left_heat.shape[1] == pytest.approx(100, abs=1)


def test_render_region_shares_normalization_with_full_frame():
    acc = HeatmapAccumulator(200, 100, HeatmapConfig(resolution_scale=1.0, sigma_px=2.0))
    for _ in range(50):
        acc.add_point(20, 20)
    acc.add_point(180, 20)

    full = acc.render_full(normalize=True)
    right_region = BBox(100, 0, 200, 100)
    right_heat = acc.render_region(right_region, normalize=True)

    assert full.max() == pytest.approx(1.0)
    assert right_heat.max() < 0.5  # the sparse region should look faint relative to global peak


def test_colorize_shape_and_dtype():
    normalized = np.linspace(0, 1, 100).reshape(10, 10)
    color = colorize(normalized, colormap="jet")
    assert color.shape == (10, 10, 3)
    assert color.dtype == np.uint8


def test_overlay_zero_heat_returns_background_unchanged():
    background = np.full((20, 20, 3), 128, dtype=np.uint8)
    normalized = np.zeros((5, 5))
    result = overlay_on_image(background, normalized, alpha=0.55)
    assert np.array_equal(result, background)


def test_overlay_high_heat_shifts_pixels_toward_colormap():
    background = np.full((20, 20, 3), 128, dtype=np.uint8)
    normalized = np.ones((5, 5))
    result = overlay_on_image(background, normalized, alpha=1.0)
    assert not np.array_equal(result, background)
