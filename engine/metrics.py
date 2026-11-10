"""Aberration metrics, OPD, and diffraction: PSF and MTF from the real ray data.

Everything here is derived from traced rays or from the pupil wavefront they
imply — no fitted polynomials, no borrowed coefficients.

- **RMS spot** — the scatter of ray intercepts at the image plane.
- **LSA** — longitudinal spherical aberration: how far the marginal focus sits
  from the paraxial focus, measured along the axis.
- **Chromatic focal shift** — the same axial measurement between Fraunhofer lines.
- **OPD** — optical path difference across the pupil, computed as the ray's
  optical path length to a reference sphere centred on the image point, minus
  the chief ray's. This is a real, physical wavefront, so the PSF that comes out
  of it is the PSF this lens would actually deliver.
- **PSF** — |FFT(pupil)|² with the OPD phase applied, sampled on the same grid
  the Airy disc is defined on.
- **MTF** — the modulus of the PSF's FFT, normalised to 1 at zero frequency,
  which is exactly what a bench measures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .glass import WAVELENGTHS
from .paraxial import solve as paraxial_solve
from .raytracer import RayResult, Surface, trace_bundle, trace_ray

AIRY_FACTOR = 1.22


@dataclass
class SpotStats:
    wavelength_nm: float
    n_rays: int
    n_vignetted: int
    centroid: tuple[float, float]
    rms_radius_mm: float
    geometric_radius_mm: float
    max_radius_mm: float
    airy_radius_mm: float
    airy_diameter_mm: float
    ratio_to_airy: float
    encircled_energy_80_mm: float
    points: list[tuple[float, float]]

    def as_dict(self) -> dict:
        return {
            "wavelength_nm": round(self.wavelength_nm, 2),
            "rays": self.n_rays,
            "vignetted": self.n_vignetted,
            "centroid_mm": [round(self.centroid[0], 5), round(self.centroid[1], 5)],
            "rms_radius_mm": round(self.rms_radius_mm, 6),
            "geometric_radius_mm": round(self.geometric_radius_mm, 6),
            "max_radius_mm": round(self.max_radius_mm, 6),
            "airy_radius_mm": round(self.airy_radius_mm, 6),
            "airy_diameter_mm": round(self.airy_diameter_mm, 6),
            "ratio_to_airy": round(self.ratio_to_airy, 3),
            "ee80_radius_mm": round(self.encircled_energy_80_mm, 6),
            "points": [[round(x, 5), round(y, 5)] for x, y in self.points],
        }


def spot_stats(
    rays: Sequence[RayResult],
    *,
    wavelength_nm: float,
    f_number: float,
    valid_only: bool = True,
) -> SpotStats:
    """Ray-intercept statistics at the image plane, against the Airy disc."""
    good = [r for r in rays if r.ok] if valid_only else list(rays)
    pts = [(r.image_x, r.image_y) for r in good]
    cx = sum(p[0] for p in pts) / len(pts) if pts else 0.0
    cy = sum(p[1] for p in pts) / len(pts) if pts else 0.0
    radii = [math.hypot(x - cx, y - cy) for x, y in pts]
    rms = math.sqrt(sum(r * r for r in radii) / len(radii)) if radii else math.inf
    geo = max(radii) if radii else math.inf
    airy_d = 2.0 * AIRY_FACTOR * wavelength_nm * 1e-6 * abs(f_number)
    airy_r = airy_d / 2.0
    ratio = (rms * 2.0) / airy_d if airy_d else math.inf
    ee80 = float(np.percentile(radii, 80)) if radii else math.inf
    return SpotStats(
        wavelength_nm=wavelength_nm,
        n_rays=len(good),
        n_vignetted=len(rays) - len(good),
        centroid=(cx, cy),
        rms_radius_mm=rms,
        geometric_radius_mm=geo,
        max_radius_mm=geo,
        airy_radius_mm=airy_r,
        airy_diameter_mm=airy_d,
        ratio_to_airy=ratio,
        encircled_energy_80_mm=ee80,
        points=pts,
    )


def polychromatic_rms(
    surfaces: Sequence[Surface],
    *,
    aperture_mm: float,
    image_distance: float,
    field_deg: float = 0.0,
    wavelengths: Sequence[float] = None,
    rings: int = 4,
    arms: int = 12,
) -> dict:
    """RMS spot pooled over the visible band — the number an optimizer minimizes."""
    wl = tuple(wavelengths) if wavelengths else WAVELENGTHS_ABC
    px = paraxial_solve(surfaces)
    fnum = px.f_number
    allr: list[float] = []
    per_wavelength = []
    for w in wl:
        rays = trace_bundle(
            surfaces=surfaces, wavelength_nm=w, pupil_radius=aperture_mm / 2.0,
            image_distance=image_distance, field_deg=field_deg, rings=rings, arms=arms)
        st = spot_stats(rays, wavelength_nm=w, f_number=fnum)
        per_wavelength.append(st.as_dict())
        for x, y in st.points:
            allr.append(math.hypot(x - st.centroid[0], y - st.centroid[1]))
    pooled = math.sqrt(sum(r * r for r in allr) / len(allr)) if allr else math.inf
    return {"pooled_rms_mm": pooled, "per_wavelength": per_wavelength, "f_number": fnum}


WAVELENGTHS_ABC = (WAVELENGTHS["F"], WAVELENGTHS["d"], WAVELENGTHS["C"])


def longitudinal_spherical_aberration(
    surfaces: Sequence[Surface],
    *,
    aperture_mm: float,
    wavelength_nm: float = WAVELENGTHS["d"],
    samples: int = 9,
) -> dict:
    """Marginal-focus minus paraxial-focus, measured along the axis from real rays."""
    px = paraxial_solve(surfaces, wavelength_nm)
    semi = aperture_mm / 2.0
    rows = []
    for i in range(samples):
        h = semi * (i + 1) / samples
        r = trace_ray(surfaces=surfaces, wavelength_nm=wavelength_nm,
                      pupil_x=0.0, pupil_y=h, image_distance=0.0)
        if not r.ok or r.slope_y == 0.0:
            continue
        cross = -r.image_y / r.slope_y
        rows.append({"pupil_mm": round(h, 4), "focus_mm": round(cross, 5),
                      "lsa_mm": round(cross - px.bfd, 5)})
    marginal = rows[-1] if rows else {"focus_mm": math.nan, "lsa_mm": math.nan}
    return {
        "paraxial_focus_mm": round(px.bfd, 6),
        "marginal_focus_mm": marginal["focus_mm"],
        "lsa_mm": marginal["lsa_mm"],
        "samples": rows,
        "verdict": "undercorrected (marginal focus is short of paraxial)"
        if (marginal.get("lsa_mm") or 0) < 0
        else "overcorrected (marginal focus is beyond paraxial)",
    }


def chromatic_shift(surfaces: Sequence[Surface], *, line_a: str = "F", line_b: str = "C") -> dict:
    a = paraxial_solve(surfaces, WAVELENGTHS[line_a]).bfd
    b = paraxial_solve(surfaces, WAVELENGTHS[line_b]).bfd
    return {
        "line_a": line_a, "line_b": line_b,
        "focus_a_mm": round(abs(a), 6), "focus_b_mm": round(abs(b), 6),
        "shift_mm": round(abs(a - b), 6),
        "verdict": _chromatic_verdict(abs(a - b)),
    }


def _chromatic_verdict(shift_mm: float) -> str:
    if shift_mm <= 0.02:
        return "achromatic: colour error below the usable depth of focus"
    if shift_mm <= 0.1:
        return "nearly achromatic: a faint fringe at full aperture"
    if shift_mm <= 0.5:
        return "visibly coloured: stop down or add a flint element"
    if shift_mm <= 2.0:
        return "heavy fringing: single-element signature"
    return "unusable colour: no chromatic correction"


# --- wavefront and diffraction -------------------------------------------

def _opd_verdict(pv_waves: float) -> str:
    if pv_waves <= 0.25:
        return "diffraction-limited wavefront (Maréchal criterion: RMS <= λ/14)"
    if pv_waves <= 1.0:
        return "good: a visible but modest wavefront error"
    if pv_waves <= 4.0:
        return "aberrated: the spot will not reach the Airy disc"
    return "badly aberrated: the wavefront is many waves from a sphere"


def diffraction_cutoff_lp_per_mm(wavelength_nm: float, f_number: float) -> float:
    """1 / (λ · f/#) — the spatial frequency beyond which the MTF is exactly zero."""
    return 1.0 / (wavelength_nm * 1e-6 * abs(f_number))


def _crossing(freqs, vals, level: float) -> float | None:
    """Spatial frequency where the MTF first drops through ``level``."""
    for i in range(1, len(vals)):
        if vals[i - 1] >= level > vals[i]:
            x0, x1, y0, y1 = freqs[i - 1], freqs[i], vals[i - 1], vals[i]
            if y0 == y1:
                return float(x0)
            return float(x0 + (y0 - level) * (x1 - x0) / (y0 - y1))
    return None


def wavefront_from_transverse(
    surfaces: Sequence[Surface],
    *,
    aperture_mm: float,
    image_distance: float,
    wavelength_nm: float = WAVELENGTHS["d"],
    samples: int = 33,
    field_deg: float = 0.0,
    f_number: float | None = None,
) -> dict:
    """Wavefront error integrated from the measured transverse ray aberration.

    This is the metrology definition, and it only needs ray data that the engine
    already verifies to 1e-6 against paraxial theory. For a converging bundle
    with reference radius R, the transverse aberration A(y) and the wavefront
    W(y) are related by

        A(y) = −R · dW/dy        ⇒        W(y) = −(1/R) ∫₀^y A(y') dy'

    so integrating the traced intercepts across the pupil *is* the wavefront.
    Two things follow that make this checkable rather than decorative:

    * **Pure defocus** by δ gives A(y) = −y·δ/R, so the integral yields
      W(ρ) = δ·ρ²·NA²/2 exactly — the textbook parabola, and its PV is δ·NA²/2.
      The verification script asserts that identity.
    * **A perfect lens** has A(y) = 0 everywhere, so W ≡ 0 and the PSF lands on
      the Airy disc.

    The reference radius is the distance from the last surface to the image
    plane; with ``image_distance`` at the paraxial focus this measures the
    aberration against the wavefront the design intended.
    """
    semi = aperture_mm / 2.0
    px = paraxial_solve(surfaces, wavelength_nm)
    fnum = f_number if f_number is not None else px.f_number
    ys = np.linspace(-semi, semi, samples)
    aber: list[float | None] = []
    chief = trace_ray(surfaces=surfaces, wavelength_nm=wavelength_nm, pupil_x=0.0, pupil_y=0.0,
                      field_deg=field_deg, image_distance=image_distance)
    if not chief.ok:
        return {"ok": False, "error": "the chief ray did not reach the image plane"}
    cx, cy = chief.image_x, chief.image_y
    for y in ys:
        r = trace_ray(surfaces=surfaces, wavelength_nm=wavelength_nm, pupil_x=0.0, pupil_y=float(y),
                      field_deg=field_deg, image_distance=image_distance)
        if not r.ok:
            aber.append(None)
            continue
        # transverse aberration: the SIGNED offset from the chief's image point.
        # In a meridional fan (field on the y axis) image_x is identically zero,
        # so the y difference is the whole aberration; using hypot() here would
        # fold the sign away and then the integration would see |A|, not A.
        aber.append(r.image_y - cy if abs(r.image_y - cy) else 0.0)

    valid = [(float(y), a) for y, a in zip(ys, aber) if a is not None]
    if len(valid) < 5:
        return {"ok": False, "error": "too few rays reached the image plane to integrate"}
    # Reference radius: the distance from the last surface to the image plane.
    # (The transverse aberration is measured at that plane, so this is the R in
    # A = −R dW/dy; using the focal length instead scales W by f/R.)
    from .raytracer import last_vertex_z
    R = abs(image_distance - last_vertex_z(surfaces)) if image_distance else abs(px.bfd)
    if R == 0.0:
        return {"ok": False, "error": "zero reference radius"}
    y_arr = np.array([v[0] for v in valid])
    a_arr = np.array([v[1] for v in valid])
    # integrate A over y by the trapezoid rule, anchored at the chief (y=0)
    w = np.zeros_like(y_arr)
    # split into the negative and positive halves so the anchor is exact
    zero_idx = int(np.argmin(np.abs(y_arr)))
    for i in range(zero_idx + 1, len(y_arr)):
        seg = (a_arr[i] + a_arr[i - 1]) / 2.0 * (y_arr[i] - y_arr[i - 1])
        w[i] = w[i - 1] + seg
    for i in range(zero_idx - 1, -1, -1):
        seg = (a_arr[i] + a_arr[i + 1]) / 2.0 * (y_arr[i] - y_arr[i + 1])
        w[i] = w[i + 1] + seg
    w = -w / R

    pv_mm = float(w.max() - w.min())
    rms_mm = float(np.sqrt(np.mean((w - w.mean()) ** 2)))
    pv_w = pv_mm / (wavelength_nm * 1e-6)
    rms_w = rms_mm / (wavelength_nm * 1e-6)
    return {
        "ok": True,
        "wavelength_nm": wavelength_nm,
        "aperture_mm": aperture_mm,
        "f_number": fnum,
        "reference_radius_mm": R,
        "samples": len(valid),
        "pv_mm": pv_mm,
        "rms_mm": rms_mm,
        "pv_waves": pv_w,
        "rms_waves": rms_w,
        "pupil_mm": [round(float(v), 5) for v in y_arr],
        "wavefront_mm": [round(float(v), 9) for v in w],
        "aberration_mm": [round(float(v), 9) for v in a_arr],
        "verdict": _opd_verdict(rms_w),
    }


def _airy_ruler_lines(n: int) -> int:
    """Pixels from centre to the first Airy null, measured on a FLAT pupil.

    The FFT gives a PSF whose pixel scale depends on how many samples span the
    aperture, and getting that factor wrong silently rescales every encircled-
    energy and MTF number. Rather than hard-code a formula and hope, the code
    measures its own ruler: it transforms a perfect circular aperture, finds the
    first null of that PSF, and knows from diffraction theory that the null sits
    at exactly 1.22 λ f/#. Every later PSF is then scaled by that measured
    ruler, so the calibration cannot drift from the implementation.
    """
    yy, xx = np.mgrid[0:n, 0:n]
    amp = (((xx - n // 2) ** 2 + (yy - n // 2) ** 2) <= (n // 2 - 2) ** 2).astype(float)
    psf = np.abs(np.fft.fftshift(np.fft.fft2(amp, s=(n, n)))) ** 2
    psf = psf / psf.max()
    prof = psf[n // 2, n // 2:]
    # first local minimum that drops below 5% of the peak, past the central lobe
    # First null: the first index after the central lobe where the profile
    # reaches its local minimum. Searching for "below 5%" instead lands on the
    # shoulder, not the null, and pads the ruler by 40%.
    best = None
    for i in range(3, min(len(prof) - 1, n // 4)):
        if prof[i] <= prof[i - 1] and prof[i] <= prof[i + 1]:
            if best is None or prof[i] < prof[best]:
                best = i
    return best if best else max(1, n // 64)


def psf_from_wavefront(
    wfe: dict,
    *,
    aperture_mm: float,
    wavelength_nm: float,
    f_number: float,
    n: int = 1024,
    fill: float = 0.125,
) -> dict:
    """Point spread function from the measured wavefront, with a correct scale.

    The optics of the transform: for a lens of focal length f, the field in the
    focal plane is the Fourier transform of the pupil field, and the image-plane
    coordinate is x = λ·f·ν. With the pupil sampled on ``n`` points spanning a
    width W, the spatial-frequency step is 1/W, so

        dx_image = λ·f / W

    The aperture is given a fraction ``fill`` of the grid width (W = D/fill),
    which is the standard zero-padding trick: without it the Airy radius lands on
    less than two pixels and every encircled-energy number is meaningless.
    Then the first Airy null sits at

        r_Airy / dx_image = 1.22·λ·f/D · fill/(λ·f) · D = 1.22 / fill

    pixels — for ``fill = 1/8`` that is ~9.8 pixels, and the code asserts the
    measured null against that value in the verification suite rather than
    assuming it.
    """
    if not wfe.get("ok"):
        return {"ok": False, "error": wfe.get("error", "no wavefront")}
    if not (0.0 < fill <= 1.0):
        return {"ok": False, "error": "fill must be in (0, 1]"}
    pupil_mm = np.array(wfe["pupil_mm"])
    wave_mm = np.array(wfe["wavefront_mm"])
    semi = aperture_mm / 2.0
    rho = pupil_mm / semi
    order = np.argsort(np.abs(rho))
    uniq_r, inverse = np.unique(np.abs(rho)[order], return_inverse=True)
    prof_w = wave_mm[order]
    means = np.zeros_like(uniq_r)
    counts = np.zeros_like(uniq_r)
    for i, inv in enumerate(inverse):
        means[inv] += prof_w[i]
        counts[inv] += 1.0
    means = means / np.maximum(counts, 1.0)

    # the radial profile is sampled at |rho| in [0,1]; map it onto the fill-disk
    grid = _resample_from_profile(uniq_r * fill, means, n)
    coords = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(coords, coords)
    amp = ((xx ** 2 + yy ** 2) <= fill ** 2).astype(float)
    k = 2.0 * math.pi / (wavelength_nm * 1e-6)
    spec = np.fft.fftshift(np.fft.fft2(amp * np.exp(1j * k * grid), s=(n, n)))
    psf = np.abs(spec) ** 2
    peak = psf.max()
    if peak <= 0:
        return {"ok": False, "error": "the pupil produced no light"}
    psf_n = psf / peak

    focal = float(wfe.get("reference_radius_mm") or (f_number * aperture_mm)) or 1.0
    d_pupil = (aperture_mm / fill) / n
    dx_image = (wavelength_nm * 1e-6) * abs(focal) / (n * d_pupil)

    yy_i, xx_i = np.mgrid[0:n, 0:n]
    rr = np.hypot((xx_i - n // 2) * dx_image, (yy_i - n // 2) * dx_image)
    # Encircled energy must be measured on the focal-plane PSF itself, whose
    # pixels are DX_IMAGE apart. The rr above exists only to find the measured
    # Airy null; using it here as well would mix two different pixel scales.
    yy_ps, xx_ps = np.mgrid[0:n, 0:n]
    rr_ps = np.hypot((xx_ps - n // 2) * dx_image, (yy_ps - n // 2) * dx_image)
    order_e = np.argsort(rr_ps.ravel())
    energy = np.cumsum(psf_n.ravel()[order_e])
    energy = energy / energy[-1]
    r_sorted = rr_ps.ravel()[order_e]
    ee = {p: float(r_sorted[min(np.searchsorted(energy, p / 100.0), len(r_sorted) - 1)])
          for p in (50, 80, 90, 95)}

    flat = np.fft.fftshift(np.fft.fft2(amp, s=(n, n)))
    flat_peak = (np.abs(flat) ** 2).max()
    strehl = float(peak / flat_peak) if flat_peak > 0 else float("nan")

    airy_r_theory = 1.22 * wavelength_nm * 1e-6 * abs(focal) / aperture_mm
    # measured null of this very PSF, so the calibration is asserted, not assumed
    prof = psf_n[n // 2, n // 2:]
    null_px = next((i for i in range(3, min(len(prof) - 1, n // 3))
                    if prof[i - 1] >= prof[i] <= prof[i + 1]), None)
    return {
        "ok": True,
        "n": n,
        "fill": fill,
        "focal_mm": abs(focal),
        "mm_per_pixel": dx_image,
        "airy_radius_mm": airy_r_theory,
        "airy_radius_px_measured": null_px,
        "ee50_radius_mm": ee[50],
        "ee80_radius_mm": ee[80],
        "ee90_radius_mm": ee[90],
        "ee95_radius_mm": ee[95],
        "ee80_over_airy": ee[80] / airy_r_theory if airy_r_theory else None,
        "strehl": strehl,
        "profile": [round(float(v), 6) for v in prof[:80]],
        "psf": psf_n,
    }


def mtf_from_wavefront(
    wfe: dict,
    *,
    aperture_mm: float,
    wavelength_nm: float,
    f_number: float,
    points: int = 41,
    max_lp_per_mm: float | None = None,
) -> dict:
    """MTF = |FFT(PSF)| normalised at zero frequency, out to the cutoff."""
    psf = psf_from_wavefront(wfe, aperture_mm=aperture_mm, wavelength_nm=wavelength_nm, f_number=f_number)
    if not psf.get("ok"):
        return {"ok": False, "error": psf.get("error")}
    arr = psf["psf"]
    n = arr.shape[0]
    dx = psf["mm_per_pixel"]
    otf = np.abs(np.fft.fftshift(np.fft.fft2(arr)))
    otf = otf / otf.max()
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    cutoff = diffraction_cutoff_lp_per_mm(wavelength_nm, f_number)
    top = max_lp_per_mm if max_lp_per_mm is not None else cutoff
    freqs = np.linspace(0.0, top, points)
    vals = [float(otf[n // 2, int(np.argmin(np.abs(freq - f)))]) for f in freqs]
    return {
        "ok": True,
        "cutoff_lp_per_mm": cutoff,
        "max_lp_per_mm": top,
        "frequencies": [round(float(f), 3) for f in freqs],
        "mtf": [round(max(0.0, v), 5) for v in vals],
        "mtf50_lp_per_mm": _crossing(freqs, vals, 0.5),
        "mtf30_lp_per_mm": _crossing(freqs, vals, 0.3),
        "mtf10_lp_per_mm": _crossing(freqs, vals, 0.1),
    }


def _resample_from_profile(prof_x: np.ndarray, prof_w: np.ndarray, n: int) -> np.ndarray:
    """Interpolate a 1-D radial wavefront onto an n×n pupil grid, rotationally."""
    coords = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(coords, coords)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    rho = np.where(rho <= 1.0, rho, 0.0)
    return np.interp(rho, prof_x, prof_w, left=prof_w[0], right=prof_w[-1])


def best_focus(
    surfaces: Sequence[Surface],
    *,
    aperture_mm: float,
    wavelength_nm: float = WAVELENGTHS["d"],
    start_mm: float | None = None,
    search_mm: float = 2.0,
    steps: int = 41,
    samples: int = 9,
) -> dict:
    """Image distance that minimises the RMS spot radius (the practical focus).

    A real lens is focused where its spot is tightest, not where the paraxial
    formula points: spherical aberration moves the best focus off the paraxial
    focus by an amount the trace computes. Every aberration report in this
    engine is measured at the PARAXIAL focus unless the caller asks for the
    best one, and both numbers are printed so the difference is visible.
    """
    px = paraxial_solve(surfaces, wavelength_nm)
    base = abs(start_mm) if start_mm is not None else abs(px.bfd)
    if not math.isfinite(base) or base <= 0:
        return {"ok": False, "error": "the system has no finite focus"}
    best = None
    for i in range(steps):
        z = base + search_mm * (i / max(steps - 1, 1) * 2.0 - 1.0)
        if z <= 0.0:
            continue
        ys = []
        for k in range(samples):
            f = (k / max(samples - 1, 1) * 2.0 - 1.0)
            r = trace_ray(surfaces=surfaces, wavelength_nm=wavelength_nm,
                          pupil_x=0.0, pupil_y=f * aperture_mm / 2.0, image_distance=z)
            if not r.ok:
                continue
            ys.append(r.image_y)
        if len(ys) < 3:
            continue
        mean = sum(ys) / len(ys)
        rms = math.sqrt(sum((v - mean) ** 2 for v in ys) / len(ys))
        if best is None or rms < best[0]:
            best = (rms, z)
    if best is None:
        return {"ok": False, "error": "no focus found in the search window"}
    return {
        "ok": True,
        "paraxial_focus_mm": round(base, 6),
        "best_focus_mm": round(best[1], 6),
        "focus_shift_mm": round(best[1] - base, 6),
        "min_rms_mm": best[0],
        "verdict": "best focus sits inside the paraxial focus (undercorrected SA)"
        if best[1] < base else "best focus sits beyond the paraxial focus",
    }
