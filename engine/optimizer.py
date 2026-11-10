"""Damped least-squares lens optimizer (Levenberg–Marquardt).

The merit function is built from the numbers the engine already verifies: RMS
spot radius over the visible band, plus penalties that hold the design to its
specification (focal length, back focal distance, glass thickness, edge
thickness, physical buildability). Curvatures and air spaces are the variables;
the glass is not, because a real design fixes its materials and bends the glass.

Why DLS rather than a black-box search: the merit function is a sum of squares,
so its Jacobian can be assembled numerically once per iteration and the LM
normal equations give a step that is damped when the local model is bad. Each
iteration is one merit evaluation per variable plus one, which is cheap here
because the tracer is analytic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .glass import WAVELENGTHS, glass as get_glass
from .metrics import spot_stats, trace_bundle
from .paraxial import solve as paraxial_solve
from .raytracer import Surface, last_vertex_z

VISIBLE = (WAVELENGTHS["F"], WAVELENGTHS["d"], WAVELENGTHS["C"])


@dataclass
class Design:
    """A lens system plus the specification it must satisfy."""

    surfaces: list[Surface]
    aperture_mm: float
    efl_target_mm: float | None = None
    bfd_target_mm: float | None = None
    field_deg: float = 0.0
    wavelengths: tuple[float, ...] = VISIBLE
    min_edge_mm: float = 0.8
    label: str = "design"

    def copy(self) -> "Design":
        return Design(
            surfaces=[Surface(**{**s.__dict__}) for s in self.surfaces],
            aperture_mm=self.aperture_mm,
            efl_target_mm=self.efl_target_mm,
            bfd_target_mm=self.bfd_target_mm,
            field_deg=self.field_deg,
            wavelengths=self.wavelengths,
            min_edge_mm=self.min_edge_mm,
            label=self.label,
        )


@dataclass
class OptimizerResult:
    design: Design
    merit_start: float
    merit_end: float
    iterations: int
    history: list[dict] = field(default_factory=list)
    stopped_because: str = "converged"

    def as_dict(self) -> dict:
        return {
            "label": self.design.label,
            "merit_start_mm": round(self.merit_start, 8),
            "merit_end_mm": round(self.merit_end, 8),
            "improvement_factor": round(self.merit_start / self.merit_end, 3)
            if self.merit_end else None,
            "iterations": self.iterations,
            "stopped_because": self.stopped_because,
            "history": self.history,
            "surfaces": [
                {
                    "radius_mm": round(float(s.radius), 5),
                    "thickness_mm": round(s.thickness, 5),
                    "glass": s.glass_after or "air",
                    "conic": s.conic,
                    "a4": s.asphere[0],
                    "a6": s.asphere[1],
                }
                for s in self.design.surfaces
            ],
        }


def merit(design: Design, *, rings: int = 3, arms: int = 10) -> tuple[float, dict]:
    """Merit = squared RMS spot pooled over the band, plus the specification terms.

    Returns (merit, detail). The merit is in mm² so its square root reads as a
    spot radius, which is what the HUD displays.
    """
    if not design.surfaces:
        return math.inf, {"error": "no surfaces"}
    try:
        px = paraxial_solve(design.surfaces, WAVELENGTHS["d"])
        image_distance = abs(px.bfd) if math.isfinite(px.bfd) else 1.0
        if not math.isfinite(image_distance) or image_distance <= 0:
            return math.inf, {"error": "the system has no finite focus"}
        pts: list[tuple[float, float]] = []
        per_w = []
        for w in design.wavelengths:
            rays = trace_bundle(
                surfaces=design.surfaces, wavelength_nm=w, pupil_radius=design.aperture_mm / 2.0,
                image_distance=image_distance, field_deg=design.field_deg, rings=rings, arms=arms)
            st = spot_stats(rays, wavelength_nm=w, f_number=px.f_number)
            per_w.append({"wavelength_nm": round(w, 2), "rms_mm": st.rms_radius_mm,
                          "vignetted": st.n_vignetted})
            for x, y in st.points:
                pts.append((x - st.centroid[0], y - st.centroid[1]))
        if not pts:
            return math.inf, {"error": "no ray reached the image plane"}
        rms = math.sqrt(sum(x * x + y * y for x, y in pts) / len(pts))
        m = rms * rms

        penalties = {}
        # hold the focal length if a target is given
        if design.efl_target_mm:
            err = (px.efl - design.efl_target_mm) / design.efl_target_mm
            penalties["efl"] = err * err
            m += penalties["efl"] * 1.0
        if design.bfd_target_mm:
            err = (abs(px.bfd) - design.bfd_target_mm) / max(design.bfd_target_mm, 1e-9)
            penalties["bfd"] = err * err
            m += penalties["bfd"] * 1.0
        # physical buildability: every element must have positive edge thickness
        edge = _min_edge_thickness(design)
        penalties["edge_min_mm"] = edge
        if edge < design.min_edge_mm:
            v = (design.min_edge_mm - edge) / design.min_edge_mm
            penalties["edge_penalty"] = v * v
            m += penalties["edge_penalty"] * 4.0
        for s in design.surfaces[:-1]:
            if s.thickness <= 0.2:
                penalties["thickness_penalty"] = penalties.get("thickness_penalty", 0.0) + 1.0
                m += 1.0

        airy = 1.22 * (WAVELENGTHS["d"] * 1e-6) * abs(px.f_number) if px.f_number else 0.0
        detail = {"rms_mm": rms, "f_number": px.f_number, "efl_mm": px.efl, "bfd_mm": px.bfd,
                  "airy_radius_mm": airy,
                  "strehl_ratio": (2.0 * rms / airy) if airy else None,
                  "per_wavelength": per_w, "penalties": penalties}
        return m, detail
    except Exception as exc:  # noqa: BLE001 - a bad design must not crash the optimizer
        return math.inf, {"error": f"{type(exc).__name__}: {exc}"}


def _min_edge_thickness(design: Design) -> float:
    """Smallest edge thickness in the system, from the actual sag and aperture."""
    semi = design.aperture_mm / 2.0
    worst = math.inf
    z = 0.0
    for i, s in enumerate(design.surfaces[:-1]):
        nxt = design.surfaces[i + 1]
        try:
            sag_this = s.sag(semi)
            sag_next = nxt.sag(semi)
        except ValueError:
            return -math.inf
        edge = s.thickness + sag_this - sag_next
        worst = min(worst, edge)
        z += s.thickness
    return worst


def _variables(design: Design) -> tuple[list[tuple[str, int]], np.ndarray]:
    """The free parameters: every curvature and every air space between elements."""
    names: list[tuple[str, int]] = []
    values: list[float] = []
    for i, s in enumerate(design.surfaces):
        # radius 0 or inf means plane; keep it fixed, a plane is a deliberate choice
        if math.isfinite(s.radius) and s.radius != 0.0:
            names.append(("curvature", i))
            values.append(1.0 / s.radius)
        if i < len(design.surfaces) - 1 and not s.glass_after:
            # an air space is free; a glass thickness is held to keep elements real
            names.append(("thickness", i))
            values.append(s.thickness)
    return names, np.array(values, dtype=float)


def _apply(design: Design, names, values: np.ndarray) -> Design:
    d = design.copy()
    for (kind, i), v in zip(names, values):
        s = d.surfaces[i]
        if kind == "curvature":
            if abs(v) < 1e-9:
                continue
            d.surfaces[i] = Surface(**{**s.__dict__, "radius": float(1.0 / v)})
        else:
            d.surfaces[i] = Surface(**{**s.__dict__, "thickness": float(v)})
    return d


def optimize(
    design: Design,
    *,
    iterations: int = 30,
    rings: int = 3,
    arms: int = 10,
    on_iteration: Callable[[dict], None] | None = None,
    lambda0: float = 1e-2,
    tolerance: float = 1e-8,
) -> OptimizerResult:
    """Levenberg–Marquardt on the merit function, logging every iteration."""
    names, x = _variables(design)
    m0, d0 = merit(design, rings=rings, arms=arms)
    history = [{"iteration": 0, "merit_mm2": m0, "rms_mm": d0.get("rms_mm"),
                "variables": [float(v) for v in x]}]
    if on_iteration:
        on_iteration(history[-1])
    if not names or not math.isfinite(m0):
        return OptimizerResult(design, m0, m0, 0, history, "nothing to optimize or merit is infinite")

    lam = lambda0
    best_x, best_m = x.copy(), m0
    for it in range(1, iterations + 1):
        base, detail = merit(_apply(design, names, x), rings=rings, arms=arms)
        if not math.isfinite(base):
            break
        # numeric Jacobian of the residual (sqrt of merit is not separable, so
        # differentiate the merit itself and form the normal equations by hand)
        J = np.zeros((len(names), len(names)))
        g = np.zeros(len(names))
        step = np.maximum(1e-6, np.abs(x) * 1e-4)
        for k in range(len(names)):
            xp = x.copy(); xp[k] += step[k]
            xm = x.copy(); xm[k] -= step[k]
            mp, _ = merit(_apply(design, names, xp), rings=rings, arms=arms)
            mm, _ = merit(_apply(design, names, xm), rings=rings, arms=arms)
            if not (math.isfinite(mp) and math.isfinite(mm)):
                step[k] *= 0.5
                continue
            g[k] = (mp - mm) / (2.0 * step[k])
            for j in range(k, len(names)):
                xpp = x.copy(); xpp[k] += step[k]; xpp[j] += step[j]
                xmm = x.copy(); xmm[k] -= step[k]; xmm[j] -= step[j]
                xpm = x.copy(); xpm[k] += step[k]; xpm[j] -= step[j]
                xmp = x.copy(); xmp[k] -= step[k]; xmp[j] += step[j]
                h = ((merit(_apply(design, names, xpp), rings=rings, arms=arms)[0]
                      - merit(_apply(design, names, xpm), rings=rings, arms=arms)[0]
                      - merit(_apply(design, names, xmp), rings=rings, arms=arms)[0]
                      + merit(_apply(design, names, xmm), rings=rings, arms=arms)[0])
                     / (4.0 * step[k] * step[j]))
                J[k, j] = h
                J[j, k] = h
            if not np.all(np.isfinite(J)):
                break

        # LM step: (JᵀJ + λ·diag)⁻¹ · (−Jᵀr) with r the merit gradient
        try:
            A = J + lam * np.diag(np.diag(J) + 1e-12)
            delta = np.linalg.solve(A, -g)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue
        # limit the step: a radius must not flip through zero in one iteration
        lim = np.maximum(np.abs(x) * 0.25, 1e-4)
        delta = np.clip(delta, -lim, lim)
        trial = _apply(design, names, x + delta)
        m_trial, detail_trial = merit(trial, rings=rings, arms=arms)
        entry = {"iteration": it, "merit_mm2": m_trial, "rms_mm": detail_trial.get("rms_mm"),
                 "lambda": lam, "step": [float(v) for v in delta],
                 "variables": [float(v) for v in (x + delta)]}
        if math.isfinite(m_trial) and m_trial < base:
            x = x + delta
            lam = max(lam / 3.0, 1e-9)
            if m_trial < best_m:
                best_x, best_m = x.copy(), m_trial
            entry["accepted"] = True
        else:
            lam *= 10.0
            entry["accepted"] = False
        history.append(entry)
        if on_iteration:
            on_iteration(entry)
        if best_m <= tolerance:
            return OptimizerResult(_apply(design, names, best_x), m0, best_m, it, history, "merit at tolerance")
        if lam > 1e8:
            return OptimizerResult(_apply(design, names, best_x), m0, best_m, it, history,
                                   "damping saturated: no descent direction left")
    return OptimizerResult(_apply(design, names, best_x), m0, best_m, iterations, history, "iteration cap")
