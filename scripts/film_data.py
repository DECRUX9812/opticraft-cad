"""Precomputed scene data for the demo film.

Rendering must be frame-rate independent and fast, so everything expensive —
ray tracing, the DLS run, spot statistics, MTF curves — is computed once here
and handed to the renderer as plain arrays. The renderer then only draws.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.api import design_payload
from engine.glass import WAVELENGTHS
from engine.metrics import (
    best_focus,
    longitudinal_spherical_aberration,
    mtf_from_wavefront,
    polychromatic_rms,
    psf_from_wavefront,
    spot_stats,
    trace_bundle,
    wavefront_from_transverse,
)
from engine.optimizer import Design, merit, optimize
from engine.paraxial import solve as paraxial_solve
from engine.presets import get as get_preset
from engine.raytracer import Surface, trace_fan, trace_ray

FRAUNHOFER = (("F", WAVELENGTHS["F"], (0.36, 0.62, 1.00)),
              ("d", WAVELENGTHS["d"], (0.36, 0.95, 0.52)),
              ("C", WAVELENGTHS["C"], (1.00, 0.42, 0.40)))


def _surfaces_dict(pts):
    return [
        {
            "radius_mm": float(s.radius) if math.isfinite(s.radius) else 0.0,
            "thickness_mm": float(s.thickness),
            "glass": s.glass_after,
            "semi_aperture_mm": float(s.semi_aperture),
            "conic": float(s.conic),
            "a4": float(s.asphere[0]),
            "a6": float(s.asphere[1]),
        }
        for s in pts
    ]


@dataclass
class FrameState:
    """One drawable lens state: geometry plus the rays that go with it."""

    surfaces: list
    paraxial: dict
    fans: dict = field(default_factory=dict)      # label -> list of ray point lists
    spot: dict = field(default_factory=dict)      # label -> spot points + stats
    summary: dict = field(default_factory=dict)


def build_state(pts, aperture_mm, field_deg=0.0, *, fan_samples=15, rings=3, arms=10) -> FrameState:
    px = paraxial_solve(pts)
    st = FrameState(surfaces=pts, paraxial=px.as_dict())
    for label, w, _c in FRAUNHOFER:
        fan = trace_fan(surfaces=pts, wavelength_nm=w, pupil_radius=aperture_mm / 2,
                        image_distance=px.bfd, field_deg=field_deg, samples=fan_samples)
        st.fans[label] = [
            {"points": [list(map(float, p)) for p in r.points], "status": r.status}
            for r in fan
        ]
        bundle = trace_bundle(surfaces=pts, wavelength_nm=w, pupil_radius=aperture_mm / 2,
                              image_distance=px.bfd, field_deg=field_deg, rings=rings, arms=arms)
        s = spot_stats(bundle, wavelength_nm=w, f_number=px.f_number)
        st.spot[label] = s.as_dict()
    poly = polychromatic_rms(pts, aperture_mm=aperture_mm, image_distance=px.bfd,
                             field_deg=field_deg, rings=rings, arms=arms)
    airy = st.spot["d"]["airy_radius_mm"] or 1e-9
    st.summary = {
        "efl_mm": px.efl,
        "f_number": px.f_number,
        "bfd_mm": px.bfd,
        "rms_mm": poly["pooled_rms_mm"],
        "airy_radius_mm": airy,
        "ratio_to_airy": poly["pooled_rms_mm"] / airy,
    }
    return st


def singlet_caustic() -> dict:
    """Act 1: the singlet's marginal vs paraxial focus, as two numbers + rays."""
    p = get_preset("singlet")
    pts = p.design.surfaces
    px = paraxial_solve(pts)
    lsa = longitudinal_spherical_aberration(pts, aperture_mm=p.design.aperture_mm)
    axi = trace_ray(surfaces=pts, wavelength_nm=WAVELENGTHS["d"], pupil_x=0, pupil_y=0.001,
                    image_distance=0.0)
    mar = trace_ray(surfaces=pts, wavelength_nm=WAVELENGTHS["d"], pupil_x=0,
                    pupil_y=p.design.aperture_mm / 2, image_distance=0.0)
    return {
        "paraxial_bdf": abs(px.bfd),
        "exact_bfd": -axi.image_y / axi.slope_y,
        "marginal_focus": -mar.image_y / mar.slope_y,
        "lsa": lsa["lsa_mm"],
        "aperture_mm": p.design.aperture_mm,
        "samples": lsa["samples"],
    }


def aperture_sweep(preset_key: str = "singlet") -> list[dict]:
    """Act 2: a sequence of states as the aperture opens."""
    p = get_preset(preset_key)
    out = []
    for ap in (1.0, 3.0, 6.0, 10.0, 15.0, 20.0, 25.0):
        st = build_state(p.design.surfaces, ap, field_deg=0.0, fan_samples=13)
        out.append({"aperture_mm": ap, "state": st})
    return out


def field_sweep(preset_key: str = "singlet") -> list[dict]:
    """Act 2: field angle opening, where coma and astigmatism appear."""
    p = get_preset(preset_key)
    out = []
    for f in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0):
        st = build_state(p.design.surfaces, p.design.aperture_mm, field_deg=f, fan_samples=13)
        out.append({"field_deg": f, "state": st})
    return out


def optimization_film(iterations: int = 30) -> dict:
    """Act 3: the DLS run, with the rays for every accepted iterate.

    The starting shape is a deliberately bad doublet: right glasses, wrong bend.
    Each logged iterate is rebuilt into full scene state so the film can morph
    the elements and re-trace the bundle as the merit falls.
    """
    base = get_preset("achromat").design
    bad = [Surface(radius=38.0, thickness=3.75, glass_after="N-BK7", semi_aperture=12.5),
           Surface(radius=-120.0, thickness=3.75, glass_after="N-SF5", semi_aperture=12.5),
           Surface(radius=-90.0, thickness=0.0, glass_after="", semi_aperture=12.5)]
    design = Design(surfaces=bad, aperture_mm=25.0, efl_target_mm=100.0, label="dls")
    m0, d0 = merit(design, rings=3, arms=10)
    result = optimize(design, iterations=iterations, rings=3, arms=10)
    m1, d1 = merit(result.design, rings=3, arms=10)

    states = []
    for h in result.history:
        vals = h.get("variables") or []
        if not vals:
            continue
        radii = []
        k = 0
        for s in bad:
            if math.isfinite(s.radius) and s.radius != 0.0:
                curv = vals[k]; k += 1
                radii.append(1.0 / curv if abs(curv) > 1e-9 else s.radius)
            else:
                radii.append(s.radius)
        pts = [Surface(radius=radii[i], thickness=bad[i].thickness,
                       glass_after=bad[i].glass_after, semi_aperture=bad[i].semi_aperture)
               for i in range(len(bad))]
        st = build_state(pts, 25.0, fan_samples=13, rings=2, arms=8)
        states.append({"iteration": h["iteration"], "merit": h["merit_mm2"],
                       "rms_mm": st.summary["rms_mm"], "state": st,
                       "accepted": h.get("accepted")})

    final = build_state(result.design.surfaces, 25.0, fan_samples=15, rings=4, arms=12)
    wfe = wavefront_from_transverse(result.design.surfaces, aperture_mm=25.0,
                                    image_distance=abs(paraxial_solve(result.design.surfaces).bfd),
                                    samples=61)
    psf = psf_from_wavefront(wfe, aperture_mm=25.0, wavelength_nm=WAVELENGTHS["d"],
                             f_number=paraxial_solve(result.design.surfaces).f_number)
    mtf = mtf_from_wavefront(wfe, aperture_mm=25.0, wavelength_nm=WAVELENGTHS["d"],
                             f_number=paraxial_solve(result.design.surfaces).f_number)
    return {
        "states": states,
        "merit_start": m0,
        "merit_end": m1,
        "rms_start": d0.get("rms_mm"),
        "rms_end": d1.get("rms_mm"),
        "ratio_start": d0.get("strehl_ratio"),
        "ratio_end": d1.get("strehl_ratio"),
        "final": final,
        "wfe": wfe,
        "psf": {k: v for k, v in psf.items() if k != "psf"},
        "psf_profile": psf.get("profile", []),
        "mtf": mtf,
        "iterations": result.iterations,
    }


def triplet_state() -> dict:
    """Act 4: the three-element anastigmat, plus its MTF."""
    p = get_preset("triplet")
    pts = p.design.surfaces
    px = paraxial_solve(pts)
    st = build_state(pts, p.design.aperture_mm, field_deg=p.design.field_deg, fan_samples=17)
    bf = best_focus(pts, aperture_mm=p.design.aperture_mm)
    wfe = wavefront_from_transverse(pts, aperture_mm=p.design.aperture_mm,
                                    image_distance=bf["best_focus_mm"], samples=61,
                                    field_deg=0.0)
    psf = psf_from_wavefront(wfe, aperture_mm=p.design.aperture_mm,
                             wavelength_nm=WAVELENGTHS["d"], f_number=px.f_number)
    mtf = mtf_from_wavefront(wfe, aperture_mm=p.design.aperture_mm,
                             wavelength_nm=WAVELENGTHS["d"], f_number=px.f_number)
    return {
        "title": p.title,
        "blurb": p.blurb,
        "state": st,
        "best_focus": bf,
        "wfe": wfe,
        "psf": {k: v for k, v in psf.items() if k != "psf"},
        "psf_profile": psf.get("profile", []),
        "mtf": mtf,
        "surfaces": _surfaces_dict(pts),
    }


def build_all(iterations: int = 30) -> dict:
    print("[film] building scene data ...")
    data = {
        "caustic": singlet_caustic(),
        "aperture_sweep": aperture_sweep(),
        "field_sweep": field_sweep(),
        "dls": optimization_film(iterations=iterations),
        "triplet": triplet_state(),
    }
    print(f"[film] dls: {len(data['dls']['states'])} iterates, "
          f"merit {data['dls']['merit_start']:.3e} -> {data['dls']['merit_end']:.3e}")
    print("[film] done")
    return data
