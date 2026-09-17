"""2D gaze heatmap accumulation and rendering."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from faceheatmap.config import HeatmapConfig
from faceheatmap.layout import BBox


class HeatmapAccumulator:
    """Accumulates gaze points into a low-resolution grid (binned counts),
    blurred into a kernel-density-style heatmap at render time.

    Binning at render time (rather than splatting a full Gaussian per
    point) keeps accumulation O(1) per point and lets the blur radius be
    tuned after the fact without reprocessing the video.
    """

    def __init__(self, width: int, height: int, config: HeatmapConfig | None = None):
        self._config = config or HeatmapConfig()
        self._full_w, self._full_h = width, height
        self._scale = self._config.resolution_scale
        self._grid_w = max(1, round(width * self._scale))
        self._grid_h = max(1, round(height * self._scale))
        self._grid = np.zeros((self._grid_h, self._grid_w), dtype=np.float64)

    @property
    def total_weight(self) -> float:
        return float(self._grid.sum())

    def add_point(self, x: float, y: float, weight: float = 1.0) -> None:
        gx = int(round(x * self._scale))
        gy = int(round(y * self._scale))
        gx = min(max(gx, 0), self._grid_w - 1)
        gy = min(max(gy, 0), self._grid_h - 1)
        self._grid[gy, gx] += weight

    def _blurred_grid(self) -> np.ndarray:
        sigma = max(self._config.sigma_px * self._scale, 1e-6)
        return gaussian_filter(self._grid, sigma=sigma)

    def render_full(self, normalize: bool = True) -> np.ndarray:
        """Returns the full-frame heatmap, values in [0, 1] if normalized."""
        blurred = self._blurred_grid()
        if normalize:
            peak = blurred.max()
            if peak > 0:
                blurred = blurred / peak
        return blurred

    def render_region(self, region: BBox, normalize: bool = True) -> np.ndarray:
        """Returns the heatmap cropped to `region` (in full-frame pixel coords).

        Normalization (if enabled) is computed from the full-frame blurred
        map so intensity is comparable across full-frame and region crops,
        not rescaled to the region's own local peak.
        """
        blurred = self._blurred_grid()
        peak = blurred.max()
        if normalize and peak > 0:
            blurred = blurred / peak

        gx0 = int(np.floor(region.x0 * self._scale))
        gy0 = int(np.floor(region.y0 * self._scale))
        gx1 = int(np.ceil(region.x1 * self._scale))
        gy1 = int(np.ceil(region.y1 * self._scale))
        gx0, gx1 = np.clip([gx0, gx1], 0, self._grid_w)
        gy0, gy1 = np.clip([gy0, gy1], 0, self._grid_h)
        return blurred[gy0:gy1, gx0:gx1]


def colorize(normalized: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """Maps a [0, 1] float array to a BGR uint8 image via a matplotlib colormap."""
    import matplotlib

    cmap = matplotlib.colormaps[colormap]
    rgba = cmap(np.clip(normalized, 0.0, 1.0))  # (H, W, 4) float in [0,1]
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb[..., ::-1]  # RGB -> BGR


def overlay_on_image(
    background_bgr: np.ndarray,
    normalized: np.ndarray,
    alpha: float = 0.55,
    colormap: str = "jet",
) -> np.ndarray:
    """Blends a colorized heatmap onto `background_bgr`, weighted by intensity
    so near-zero heat regions are left close to the original image.
    """
    import cv2

    h, w = background_bgr.shape[:2]
    resized = cv2.resize(normalized, (w, h), interpolation=cv2.INTER_LINEAR)
    color = colorize(resized, colormap)
    weight = (resized * alpha)[..., None]
    blended = background_bgr.astype(np.float64) * (1 - weight) + color.astype(np.float64) * weight
    return blended.astype(np.uint8)
