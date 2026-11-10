"""Paraxial optics: ABCD ray-transfer matrices, cardinal points, and the
first-order numbers the exact tracer is verified against.

Convention: a ray is the column vector [y, u] where y is the height and u is the
*reduced* angle n·θ. Matrices act left-to-right through the system:

    translation in medium n:  [[1, t/n], [0, 1]]
    refraction at a surface:  [[1, 0], [−φ, 1]]     φ = (n' − n)/R

A positive radius has its centre of curvature on the +z side. The system matrix
M = M_k … M_1 maps a ray at the first surface to a ray at the last vertex; for a
parallel input bundle (u = 0) the axis crossing is at −A/C past the last vertex,
and the effective focal length is EFL = −1/C.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .glass import WAVELENGTHS, glass as get_glass
from .raytracer import Surface, last_vertex_z


Matrix = tuple[tuple[float, float], tuple[float, float]]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    return (
        (a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]),
        (a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]),
    )


IDENTITY: Matrix = ((1.0, 0.0), (0.0, 1.0))


def surface_power(radius: float, n_in: float, n_out: float) -> float:
    """φ = (n' − n)/R. Zero for a plane; a sign flip for a diverging surface."""
    if radius == 0.0 or not math.isfinite(radius):
        return 0.0
    return (n_out - n_in) / radius


def conic_power(surface: Surface, n_in: float, n_out: float) -> float:
    """Conic surfaces keep the same first-order power as their vertex sphere;
    the conic constant and asphere terms change aberrations, not the paraxial
    focus. That is exactly why an asphere can be 'the same lens, just sharper'."""
    return surface_power(surface.radius, n_in, n_out)


@dataclass
class Paraxial:
    matrix: Matrix
    power: float
    efl: float
    bfd: float
    ffd: float
    f_number: float
    aperture_mm: float
    curvature_sum: float

    def as_dict(self) -> dict[str, float | list]:
        return {
            "matrix": [[round(v, 8) for v in row] for row in self.matrix],
            "power_mm_inv": round(self.power, 8),
            "efl_mm": round(self.efl, 6),
            "bfd_mm": round(self.bfd, 6),
            "ffd_mm": round(self.ffd, 6),
            "f_number": round(self.f_number, 4),
            "curvature_sum_mm_inv": round(self.curvature_sum, 6),
        }


def solve(surfaces: Sequence[Surface], wavelength_nm: float | None = None) -> Paraxial:
    """System matrix and cardinal points at one wavelength."""
    w = WAVELENGTHS["d"] if wavelength_nm is None else wavelength_nm
    m = IDENTITY
    n_in = 1.0
    curvature_sum = 0.0
    for i, s in enumerate(surfaces):
        n_out = get_glass(s.glass_after).index(w) if s.glass_after else 1.0
        phi = conic_power(s, n_in, n_out)
        curvature_sum += phi
        m = matmul(((1.0, 0.0), (-phi, 1.0)), m)
        if i < len(surfaces) - 1:
            t = s.thickness
            m = matmul(((1.0, t / n_out), (0.0, 1.0)), m)
        n_in = n_out

    a, c = m[0][0], m[1][0]
    semi = max((s.semi_aperture for s in surfaces), default=0.0)
    aperture = 2.0 * semi
    if c == 0.0:
        return Paraxial(m, 0.0, math.inf, math.inf, math.inf, math.inf, aperture, curvature_sum)
    efl = -1.0 / c
    bfd = -a / c
    ffd = bfd - efl
    f_number = abs(efl) / aperture if aperture else math.inf
    return Paraxial(m, -c, efl, bfd, ffd, f_number, aperture, curvature_sum)


def chief_ray_height(surfaces: Sequence[Surface], field_deg: float, efl: float) -> float:
    """Paraxial chief ray height at the first surface — the field's image offset
    divided by the entrance geometry. Used to place the spot diagram's centre."""
    return abs(efl) * math.tan(math.radians(field_deg))


def magnification(surfaces: Sequence[Surface], object_distance: float, wavelength_nm: float | None = None) -> float:
    """Finite conjugate magnification m = 1/A of the object-to-image matrix."""
    w = WAVELENGTHS["d"] if wavelength_nm is None else wavelength_nm
    base = solve(surfaces, w)
    m = matmul(((1.0, object_distance), (0.0, 1.0)), base.matrix)
    if m[0][0] == 0.0:
        return math.inf
    return 1.0 / m[0][0]


def back_focal_distance_for(surfaces: Sequence[Surface], wavelength_nm: float | None = None) -> float:
    return solve(surfaces, wavelength_nm).bfd


def focus_shift_between(surfaces: Sequence[Surface], w_a: float, w_b: float) -> float:
    """Axial focus difference between two wavelengths — the chromatic error."""
    return abs(solve(surfaces, w_a).bfd - solve(surfaces, w_b).bfd)


def best_focus(surfaces: Sequence[Surface], wavelength_nm: float | None = None) -> float:
    """Image distance that minimises the paraxial focus for this system."""
    return abs(solve(surfaces, wavelength_nm).bfd)
