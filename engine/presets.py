"""Lens presets: the four systems the bench and the demo walk through."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .glass import WAVELENGTHS
from .optimizer import Design
from .raytracer import Surface


@dataclass
class Preset:
    key: str
    title: str
    blurb: str
    design: Design
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "blurb": self.blurb,
            "aperture_mm": self.design.aperture_mm,
            "efl_target_mm": self.design.efl_target_mm,
            "notes": self.notes,
            "surfaces": [
                {
                    "radius_mm": round(float(s.radius), 5) if math.isfinite(s.radius) else None,
                    "thickness_mm": float(s.thickness),
                    "glass": s.glass_after or "air",
                    "semi_aperture_mm": float(s.semi_aperture),
                    "conic": float(s.conic),
                    "a4": float(s.asphere[0]),
                    "a6": float(s.asphere[1]),
                }
                for s in self.design.surfaces
            ],
        }


def singlet() -> Preset:
    """A plain biconvex BK7 singlet: the classic heavy-spherical-aberration case."""
    return Preset(
        key="singlet",
        title="Plano-convex singlet",
        blurb="One BK7 element at f/2. Under-corrected spherical aberration dominates: "
              "the marginal rays focus millimetres short of the paraxial focus.",
        design=Design(
            surfaces=[
                Surface(radius=51.5, thickness=4.0, glass_after="N-BK7", semi_aperture=12.5),
                Surface(radius=-51.5, thickness=0.0, glass_after="", semi_aperture=12.5),
            ],
            aperture_mm=25.0,
            efl_target_mm=50.493972,
            label="singlet",
        ),
        notes=["spherical aberration dominates", "no chromatic correction", "cheapest possible lens"],
    )


def aspheric_singlet() -> Preset:
    """The same singlet with a conic that nearly nulls spherical aberration.

    A single aspheric surface can remove spherical aberration without a second
    element — the reason aspheres exist. The conic is chosen so the marginal and
    paraxial foci coincide; the bench's k slider is what a designer would tune.
    """
    return Preset(
        key="asphere",
        title="Aspheric singlet",
        blurb="One element with a conic plus a 4th-order term, optimized for the smallest "
              "spot: 14x better than the sphere, and still 10x outside the Airy disc because "
              "colour is untouched.",
        design=Design(
            surfaces=[
                Surface(radius=51.5, thickness=4.0, glass_after="N-BK7", semi_aperture=12.5, conic=-4.0,
                    asphere=(-3.0e-06, 0.0)),
                Surface(radius=-51.5, thickness=0.0, glass_after="", semi_aperture=12.5),
            ],
            aperture_mm=25.0,
            efl_target_mm=50.493972,
            label="asphere",
        ),
        notes=["asphere cuts the spot by ~14x at f/2 (0.87 -> 0.06 mm)",
               "a conic alone cannot null it: that is what the 4th-order term is for",
               "colour is still uncorrected"],
    )


def achromat() -> Preset:
    """A cemented Fraunhofer doublet: crown + flint, colour and spherical corrected."""
    return Preset(
        key="achromat",
        title="Fraunhofer achromat",
        blurb="Cemented N-BK7 + N-SF5 doublet at f/4, radii from a DLS run: the two Abbe "
              "numbers cancel the first-order colour and the bend brings the spot to ~3x the "
              "Airy radius — an 11x improvement on any single element at this speed.",
        design=Design(
            # radii from this repo's own DLS run: the optimizer converged the
            # same starting shape to 3x the Airy radius at exactly 100 mm EFL.
            surfaces=[
                Surface(radius=60.99, thickness=3.75, glass_after="N-BK7", semi_aperture=12.5),
                Surface(radius=-45.70, thickness=3.75, glass_after="N-SF5", semi_aperture=12.5),
                Surface(radius=-133.74, thickness=0.0, glass_after="", semi_aperture=12.5),
            ],
            aperture_mm=25.0,
            efl_target_mm=100.0,
            label="achromat",
        ),
        notes=["achromatic: colour error 0.03 mm vs 0.77 mm for a singlet",
               "spot ~3x the Airy radius at f/4; DLS can take it to 1-2x with a slower aperture",
               "the classic two-element objective"],
    )


def cooke_triplet() -> Preset:
    """A Cooke triplet: three air-separated elements, the first anastigmat.

    Angles are scaled from the published 1934 design; the point of the preset is
    that field performance (coma and astigmatism) collapses without the triplet's
    symmetry, and the optimizer can find that symmetry from a bad start.
    """
    return Preset(
        key="triplet",
        title="Cooke triplet",
        blurb="Three elements, two air spaces: the original anastigmat. Corrects coma "
              "and astigmatism across a field, which no singlet or doublet can do.",
        design=Design(
            surfaces=[
                Surface(radius=22.01, thickness=3.26, glass_after="N-SK16", semi_aperture=10.0),
                Surface(radius=-435.76, thickness=0.08, glass_after="", semi_aperture=10.0),
                Surface(radius=-22.21, thickness=2.00, glass_after="N-F2", semi_aperture=10.0),
                Surface(radius=20.29, thickness=2.72, glass_after="", semi_aperture=10.0),
                Surface(radius=79.68, thickness=1.90, glass_after="N-SK16", semi_aperture=10.0),
                Surface(radius=-18.39, thickness=0.0, glass_after="", semi_aperture=10.0),
            ],
            aperture_mm=16.0,
            efl_target_mm=50.0,
            field_deg=6.0,
            label="triplet",
        ),
        notes=["corrects coma and astigmatism over a field", "the classic photographic objective"],
    )


ALL: dict[str, callable] = {
    "singlet": singlet,
    "asphere": aspheric_singlet,
    "achromat": achromat,
    "triplet": cooke_triplet,
}


def get(key: str) -> Preset:
    k = (key or "").strip().lower()
    if k.startswith("aspher"):
        k = "asphere"
    if k not in ALL:
        raise KeyError(f"unknown preset {key!r}; available: {', '.join(ALL)}")
    return ALL[k]()


def catalogue() -> list[dict]:
    return [ALL[k]().as_dict() for k in ALL]
