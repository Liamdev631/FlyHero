"""Turn a game frame into a policy observation: just the instrument's highway.

The game draws one highway per instrument plus a lot of HUD (score, stars, timer,
star-power meter, multiplier badge). Feeding the whole frame to a policy means most
of the input is irrelevant, and worse, the HUD is *shared* between instruments while
the highway is not - so the same pixels would mean different things for different
instruments. Cropping to the instrument's own highway fixes both: the input is the
lanes that actually determine the action, and it is unambiguous per instrument.

This module only does the geometry. It is deliberately dependency-light (numpy +
Pillow) and the region is explicit configuration rather than something guessed at
runtime, because a silently-wrong crop is the kind of bug that produces a model that
trains happily and learns nothing.

NOTE ON REGIONS: the numbers in REGIONS were measured from a real 1024x768 frame of
the built game. A different resolution, window mode, highway tilt or note speed moves
them. Re-derive with `python -m flyhero.vision --frame <png>` (writes a crop next to
the frame) and update the table.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:  # Pillow is present in this environment; keep the import soft for clarity.
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

# Five-fret guitar lane order, matching YARG's fret colours left to right.
LANE_NAMES = ("green", "red", "yellow", "blue", "orange")


@dataclass(frozen=True)
class Region:
    """Pixel bounding box of a note highway, plus the trapezoid's lane geometry.

    ``top``/``bottom`` give the highway's horizontal extent at the top and bottom
    edges of the box, because the highway is drawn in perspective: it is narrow at
    the horizon and wide at the fret line. Lane centres are interpolated between
    them, which is what a policy needs to know where a note will cross.
    """

    x0: int
    y0: int
    x1: int
    y1: int
    top_x0: int
    top_x1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def lane_centres(self, y_frac: float = 0.0) -> list[float]:
        """x centre of each of the 5 lanes, relative to the crop.

        y_frac 0.0 = top of the crop (horizon), 1.0 = bottom (fret line).
        """
        left = self.top_x0 + (self.x0 - self.top_x0) * y_frac
        right = self.top_x1 + (self.x1 - self.top_x1) * y_frac
        span = right - left
        return [left - self.x0 + span * (i + 0.5) / len(LANE_NAMES) for i in range(len(LANE_NAMES))]


# Measured from a 1024x768 frame of the built game (FlyHero, Beginner guitar, 1 player).
# The highway occupies the lower-middle of the screen; the multiplier badge overlaps
# its bottom-centre and the fret buttons sit on the bottom edge.
GUITAR_1024x768 = Region(
    x0=330, y0=415, x1=700, y1=768,
    top_x0=440, top_x1=580,
)

REGIONS: dict[tuple[int, int, str], Region] = {
    (1024, 768, "guitar"): GUITAR_1024x768,
}


def region_for(width: int, height: int, instrument: str = "guitar") -> Region:
    key = (width, height, instrument)
    if key not in REGIONS:
        known = ", ".join(f"{w}x{h}/{i}" for w, h, i in REGIONS)
        raise KeyError(
            f"no highway region for {width}x{height} instrument={instrument!r}. "
            f"Known: {known}. Measure one with `python -m flyhero.vision --frame <png>`."
        )
    return REGIONS[key]


def load_frame(path: str | Path) -> np.ndarray:
    if Image is None:  # pragma: no cover
        raise RuntimeError("Pillow is required to load frames")
    return np.asarray(Image.open(path).convert("RGB"))


def crop_highway(frame: np.ndarray, region: Region) -> np.ndarray:
    """Crop a frame to the instrument's highway."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB frame, got shape {frame.shape}")
    h, w = frame.shape[:2]
    if region.x1 > w or region.y1 > h:
        raise ValueError(
            f"region {region} does not fit frame {w}x{h}; the region was measured on a "
            "different resolution or window size"
        )
    return frame[region.y0:region.y1, region.x0:region.x1]


def to_policy_input(frame: np.ndarray, region: Region, size: tuple[int, int] = (84, 84)) -> np.ndarray:
    """Crop, downsample and normalise to a float32 CHW array in [0, 1].

    CHW matches torch conventions so this feeds a conv net without a re-layout.
    """
    crop = crop_highway(frame, region)
    if Image is None:  # pragma: no cover
        raise RuntimeError("Pillow is required to resize frames")
    # Pillow >= 9.1 moved the constants under Image.Resampling but kept aliases;
    # getattr handles both.
    resampling = getattr(getattr(Image, "Resampling", None), "BILINEAR", 1)
    small = np.asarray(
        Image.fromarray(crop).resize((size[1], size[0]), resampling),
        dtype=np.float32,
    )
    return np.transpose(small / 255.0, (2, 0, 1))


def overlay_region(frame: np.ndarray, region: Region, colour=(255, 0, 255)) -> np.ndarray:
    """Draw the crop box and lane guides on a copy - for eyeballing a region."""
    out = frame.copy()
    out[region.y0:region.y0 + 2, region.x0:region.x1] = colour
    out[region.y1 - 2:region.y1, region.x0:region.x1] = colour
    out[region.y0:region.y1, region.x0:region.x0 + 2] = colour
    out[region.y0:region.y1, region.x1 - 2:region.x1] = colour

    for y_frac in (0.0, 0.5, 1.0):
        y = int(region.y0 + region.height * y_frac)
        for cx in region.lane_centres(y_frac):
            x = int(region.x0 + cx)
            out[max(y - 3, 0):y + 3, max(x - 3, 0):x + 3] = (0, 255, 0)
    return out


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Inspect / verify a highway crop")
    ap.add_argument("--frame", required=True, help="path to a game frame PNG")
    ap.add_argument("--instrument", default="guitar")
    ap.add_argument("--out", default=None, help="where to write the overlay (default: beside the frame)")
    args = ap.parse_args()

    frame = load_frame(args.frame)
    h, w = frame.shape[:2]
    region = region_for(w, h, args.instrument)

    print(f"frame      : {w}x{h}")
    print(f"region     : x {region.x0}..{region.x1}  y {region.y0}..{region.y1} "
          f"({region.width}x{region.height})")
    print(f"lane x @top: {[round(c) for c in region.lane_centres(0.0)]}")
    print(f"lane x @bot: {[round(c) for c in region.lane_centres(1.0)]}")

    crop = crop_highway(frame, region)
    print(f"crop shape : {crop.shape}")
    print(f"policy in  : {to_policy_input(frame, region).shape}")

    base = Path(args.frame)
    out = Path(args.out) if args.out else base.with_name(base.stem + "_overlay.png")
    if Image is not None:
        Image.fromarray(overlay_region(frame, region)).save(out)
        print(f"overlay    : {out}")
        crop_path = base.with_name(base.stem + "_crop.png")
        Image.fromarray(crop).save(crop_path)
        print(f"crop       : {crop_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
