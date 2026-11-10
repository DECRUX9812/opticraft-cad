"""The workbench's computation layer: one JSON payload per request.

This module is deliberately transport-agnostic so the same functions serve the
web app (over HTTP) and the headless video renderer (called directly). Nothing
here draws anything; it computes optics and returns numbers plus the geometry a
renderer needs.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .glass import GLASSES, WAVELENGTHS, catalogue as glass_catalogue
from .metrics import (
    best_focus,
    chromatic_shift,
    diffraction_cutoff_lp_per_mm,
    longitudinal_spherical_aberration,
    mtf_from_wavefront,
    polychromatic_rms,
    psf_from_wavefront,
    spot_stats,
    trace_bundle,
    wavefront_from_transverse,
)
from .raytracer import trace_fan
from .optimizer import Design, merit, optimize
from .paraxial import solve as paraxial_solve
from .presets import catalogue as preset_catalogue
from .presets import get as get_preset
from .raytracer import Surface, describe, trace_ray


def _surfaces_from(spec: Sequence[dict[str, Any]]) -> list[Surface]:
    out = []
    for i, s in enumerate(spec):
        r = s.get("radius_mm", s.get("radius"))
        radius = float(r) if r is not None else 0.0
        out.append(Surface(
            radius=radius,
            thickness=float(s.get("thickness_mm", s.get("thickness", 0.0)) or 0.0),
            glass_after=str(s.get("glass", s.get("glass_after", "")) or ""),
            semi_aperture=float(s.get("semi_aperture_mm", s.get("semi_aperture", 0.0)) or 0.0),
            conic=float(s.get("conic", 0.0) or 0.0),
            asphere=(float(s.get("a4", 0.0) or 0.0), float(s.get("a6", 0.0) or 0.0)),
        ))
    if not out:
        raise ValueError("surface_spec is empty")
    return out


def design_payload(
    *,
    preset: str | None = None,
    surfaces: Sequence[dict[str, Any]] | None = None,
    aperture_mm: float | None = None,
    field_deg: float = 0.0,
    wavelength_override: float | None = None,
    include_wavefront: bool = True,
) -> dict[str, Any]:
    """Everything the bench and the HUD show for one design."""
    if surfaces:
        pts = _surfaces_from(surfaces)
        semi = max(s.semi_aperture for s in pts)
        aperture = aperture_mm or 2.0 * semi
        title, blurb, notes = "custom prescription", "", []
    elif preset:
        p = get_preset(preset)
        pts = list(p.design.surfaces)
        aperture = aperture_mm or p.design.aperture_mm
        title, blurb, notes = p.title, p.blurb, p.notes
    else:
        raise ValueError("pass either preset or surfaces")

    px = paraxial_solve(pts)
    bf = best_focus(pts, aperture_mm=aperture)
    lsa = longitudinal_spherical_aberration(pts, aperture_mm=aperture)
    colour = chromatic_shift(pts)

    # ray bundles per wavelength: geometry for the canvas AND spot statistics
    bundles: dict[str, Any] = {}
    for label, w in (("F", WAVELENGTHS["F"]), ("d", WAVELENGTHS["d"]), ("C", WAVELENGTHS["C"])):
        fan = trace_fan(surfaces=pts, wavelength_nm=w, pupil_radius=aperture / 2.0,
                        image_distance=px.bfd, field_deg=field_deg, samples=13)
        rays = [{"points": [list(map(float, p)) for p in r.points],
                 "status": r.status, "wavelength_nm": w} for r in fan]
        st = spot_stats(
            trace_bundle(surfaces=pts, wavelength_nm=w, pupil_radius=aperture / 2.0,
                         image_distance=px.bfd, field_deg=field_deg, rings=3, arms=10),
            wavelength_nm=w, f_number=px.f_number)
        bundles[label] = {"wavelength_nm": w, "rays": rays, "spot": st.as_dict()}

    payload: dict[str, Any] = {
        "ok": True,
        "title": title,
        "blurb": blurb,
        "notes": notes,
        "aperture_mm": aperture,
        "field_deg": field_deg,
        "surfaces": describe(pts),
        "paraxial": px.as_dict(),
        "best_focus": bf,
        "lsa": lsa,
        "colour": colour,
        "bundles": bundles,
        "polychromatic": polychromatic_rms(pts, aperture_mm=aperture, image_distance=px.bfd,
                                           field_deg=field_deg),
    }

    if include_wavefront:
        wfe = wavefront_from_transverse(
            pts, aperture_mm=aperture,
            image_distance=bf["best_focus_mm"] if bf.get("ok") else px.bfd,
            samples=61, field_deg=field_deg, f_number=px.f_number)
        psf = psf_from_wavefront(wfe, aperture_mm=aperture, wavelength_nm=WAVELENGTHS["d"],
                                 f_number=px.f_number)
        mtf = mtf_from_wavefront(wfe, aperture_mm=aperture, wavelength_nm=WAVELENGTHS["d"],
                                 f_number=px.f_number)
        payload["wavefront"] = {k: v for k, v in wfe.items() if k != "map"}
        payload["psf"] = {k: v for k, v in psf.items() if k != "psf"}
        payload["psf_profile"] = psf.get("profile", [])
        payload["mtf"] = mtf

    payload["summary"] = {
        "efl_mm": round(px.efl, 3),
        "f_number": round(px.f_number, 2),
        "rms_mm": round(payload["polychromatic"]["pooled_rms_mm"], 5),
        "airy_radius_mm": round(bundles["d"]["spot"]["airy_radius_mm"], 5),
        "ratio_to_airy": round(
            payload["polychromatic"]["pooled_rms_mm"] / max(bundles["d"]["spot"]["airy_radius_mm"], 1e-12), 2),
        "cutoff_lp_per_mm": round(diffraction_cutoff_lp_per_mm(WAVELENGTHS["d"], px.f_number), 1),
        "lsa_mm": lsa["lsa_mm"],
        "colour_shift_mm": colour["shift_mm"],
        "verdict": _verdict(payload["polychromatic"]["pooled_rms_mm"],
                            bundles["d"]["spot"]["airy_radius_mm"]),
    }
    return payload


def _verdict(rms_mm: float, airy_r_mm: float) -> str:
    if not airy_r_mm:
        return "no finite focus"
    ratio = rms_mm / airy_r_mm
    if ratio <= 1.0:
        return "diffraction-limited: the spot is inside the Airy disc"
    if ratio <= 2.0:
        return "near diffraction-limited"
    if ratio <= 5.0:
        return "good, but aberration-dominated: stop down to sharpen it"
    if ratio <= 20.0:
        return "soft: visibly blurred at full aperture"
    return "heavily aberrated: this is a singlet at full aperture"


def presets_payload() -> dict[str, Any]:
    return {"ok": True, "presets": preset_catalogue(), "glasses": glass_catalogue()}


def optimize_payload(
    *,
    preset: str | None = None,
    surfaces: Sequence[dict[str, Any]] | None = None,
    aperture_mm: float | None = None,
    efl_target_mm: float | None = None,
    field_deg: float = 0.0,
    iterations: int = 22,
    on_iteration=None,
) -> dict[str, Any]:
    """Run DLS and return the full iteration trace, for streaming or rendering."""
    if surfaces:
        pts = _surfaces_from(surfaces)
        aperture = aperture_mm or 2.0 * max(s.semi_aperture for s in pts)
    elif preset:
        p = get_preset(preset)
        pts = list(p.design.surfaces)
        aperture = aperture_mm or p.design.aperture_mm
        efl_target_mm = efl_target_mm or p.design.efl_target_mm
        # The presets ship already converged, so a run on them shows nothing.
        # Start from a deliberately mis-bent copy of the same glasses: the demo
        # must show the optimizer *doing* something, and the improvement it
        # reports must be the improvement it actually achieved.
        if preset in ("achromat", "triplet", "asphere"):
            pts = [Surface(radius=(float(surf.radius) * 0.78 if i % 2 == 0 else float(surf.radius) * 1.22),
                           thickness=surf.thickness, glass_after=surf.glass_after,
                           semi_aperture=surf.semi_aperture)
                   for i, surf in enumerate(pts)]
    else:
        raise ValueError("pass either preset or surfaces")

    design = Design(surfaces=pts, aperture_mm=aperture, efl_target_mm=efl_target_mm,
                    field_deg=field_deg, label=preset or "custom")
    before, before_detail = merit(design, rings=3, arms=10)
    result = optimize(design, iterations=iterations, rings=3, arms=10, on_iteration=on_iteration)
    after, after_detail = merit(result.design, rings=3, arms=10)
    return {
        "ok": True,
        "run": result.as_dict(),
        "before": {"merit_mm2": before, "rms_mm": before_detail.get("rms_mm"),
                   "ratio_to_airy": before_detail.get("strehl_ratio"),
                   "surfaces": describe(design.surfaces)},
        "after": {"merit_mm2": after, "rms_mm": after_detail.get("rms_mm"),
                  "ratio_to_airy": after_detail.get("strehl_ratio"),
                  "surfaces": describe(result.design.surfaces)},
        "improvement": round(before / after, 2) if after else None,
    }


def ray_fan_payload(*, surfaces, aperture_mm, field_deg, wavelength_nm, samples=41) -> dict:
    pts = _surfaces_from(surfaces) if isinstance(surfaces, (list, tuple)) and surfaces and isinstance(surfaces[0], dict) else surfaces
    px = paraxial_solve(pts)
    fan = trace_fan(surfaces=pts, wavelength_nm=wavelength_nm, pupil_radius=aperture_mm / 2,
                    image_distance=px.bfd, field_deg=field_deg, samples=samples)
    return {
        "pupil": [r.py for r in fan],
        "aberration": [r.image_y for r in fan],
        "statuses": [r.status for r in fan],
    }
