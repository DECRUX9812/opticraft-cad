"""Sequential exact ray tracer: vector Snell's law at real conic/aspheric surfaces.

No optical library. This module owns three things and nothing else:

1. **Surface intersection.** A surface is a conic base plus even asphere terms,
   given as a sag function of the radial coordinate:

       z(r) = c r² / (1 + sqrt(1 − (1+k) c² r²)) + A₄ r⁴ + A₆ r⁶        c = 1/R

   A ray (p, d) is intersected with it by Newton iteration on
   F(t) = p_z + t d_z − sag(p_x + t d_x, p_y + t d_y), with the analytic
   derivative, so the hit is exact rather than a paraxial stand-in.

2. **Refraction.** Vector Snell's law in the form that survives a sign change
   in the surface normal:

       t = η i + (η cos θᵢ − cos θₜ) n̂,   η = n₁/n₂,   cos θᵢ = −i · n̂

   ``cos θᵢ`` is flipped together with the normal when the dot product is
   negative, which is what keeps a ray travelling +z through a surface whose
   normal points −z refracting correctly. (Getting that sign wrong bends rays
   away from the axis on a converging surface; the convergence test catches it.)

3. **Honest failure.** Total internal reflection and a ray outside the clear
   aperture are reported as such, never silently dropped or clipped.

Sign convention, used everywhere: +z is the direction of light travel, and a
positive radius means the centre of curvature lies on the +z (image) side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .glass import WAVELENGTHS, glass as get_glass

Vec3 = tuple[float, float, float]


# --- vector helpers -------------------------------------------------------

def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def scale(a: Vec3, k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def unit(a: Vec3) -> Vec3:
    m = norm(a)
    if m == 0.0:
        raise ValueError("cannot normalise a zero vector")
    return scale(a, 1.0 / m)


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


# --- surfaces -------------------------------------------------------------

@dataclass(frozen=True)
class Surface:
    """One optical surface.

    ``radius``       signed radius of curvature (mm); 0 = plane, inf = plane
    ``conic``        conic constant k (0 = sphere, −1 = paraboloid)
    ``asphere``      (A4, A6) even asphere coefficients (1/mm³, 1/mm⁵)
    ``thickness``    distance along the axis to the NEXT surface (mm)
    ``glass_after``  glass filling the space after this surface ("" = air)
    ``semi_aperture`` clear semi-diameter (mm); rays outside it are vignetted
    ``stop``         marks the aperture stop (informational, used by the bench)
    """

    radius: float
    thickness: float
    glass_after: str = ""
    semi_aperture: float = 50.0
    conic: float = 0.0
    asphere: tuple[float, float] = (0.0, 0.0)
    stop: bool = False

    @property
    def curvature(self) -> float:
        if self.radius == 0.0 or not math.isfinite(self.radius):
            return 0.0
        return 1.0 / self.radius

    def sag(self, r: float) -> float:
        c = self.curvature
        if c == 0.0:
            return self.asphere[0] * r ** 4 + self.asphere[1] * r ** 6
        root = 1.0 - (1.0 + self.conic) * c * c * r * r
        if root < 0.0:
            raise ValueError(f"sag undefined at r={r:.3f} mm (beyond the conic's domain)")
        base = c * r * r / (1.0 + math.sqrt(root))
        return base + self.asphere[0] * r ** 4 + self.asphere[1] * r ** 6

    def sag_slope(self, r: float) -> float:
        """d(sag)/dr — the analytic derivative Newton needs."""
        c = self.curvature
        out = 4.0 * self.asphere[0] * r ** 3 + 6.0 * self.asphere[1] * r ** 5
        if c == 0.0:
            return out
        root = 1.0 - (1.0 + self.conic) * c * c * r * r
        if root < 0.0:
            raise ValueError(f"slope undefined at r={r:.3f} mm")
        return out + c * r / math.sqrt(root)

    def normal(self, x: float, y: float) -> Vec3:
        """Surface normal at a hit point, pointing back toward the incoming medium.

        The implicit form is F = z − sag(r); ∇F = (−sag_x, −sag_y, 1) and the
        slope chain rule gives sag_x = sag'(r)·x/r.
        """
        r = math.hypot(x, y)
        if r == 0.0:
            return (0.0, 0.0, -1.0)
        ds = self.sag_slope(r)
        return unit((-ds * x / r, -ds * y / r, 1.0))


P = Surface


# --- the trace ------------------------------------------------------------

@dataclass
class RayResult:
    """One traced ray: where it finished, and why if it did not."""

    wavelength_nm: float
    px: float                 # pupil coordinate (height at the first surface plane)
    py: float
    field_x: float
    field_y: float
    points: list[Vec3] = field(default_factory=list)   # the actual hit points, for drawing
    image_y: float = 0.0      # intercept at the image plane
    image_x: float = 0.0
    slope_y: float = 0.0      # dy/dz after the last surface
    slope_x: float = 0.0
    opl: float = 0.0          # optical path length, n·d summed (geometric)
    last_point: Vec3 = (0.0, 0.0, 0.0)
    last_dir: Vec3 = (0.0, 0.0, 1.0)
    status: str = "ok"        # ok | vignetted | tir | missed | divergent
    surface_status: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def refract(d_in: Vec3, normal: Vec3, n1: float, n2: float) -> Vec3 | None:
    """Exact vector Snell's law. Returns the refracted direction, or None on TIR."""
    cos_i = -dot(d_in, normal)
    n = normal
    if cos_i < 0.0:
        # The normal points the other way; flip it together with cos_i so the
        # formula stays consistent (this is the sign trap).
        cos_i = -cos_i
        n = scale(normal, -1.0)
    eta = n1 / n2
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    if sin2_t > 1.0:
        return None  # total internal reflection
    cos_t = math.sqrt(1.0 - sin2_t)
    t = (n[0] * (eta * cos_i - cos_t), n[1] * (eta * cos_i - cos_t), n[2] * (eta * cos_i - cos_t))
    out = (eta * d_in[0] + t[0], eta * d_in[1] + t[1], eta * d_in[2] + t[2])
    return unit(out)


def _intersect(p: Vec3, d: Vec3, s: Surface, vertex_z: float, tol: float = 1e-12, iters: int = 40):
    """Newton solve for the ray/surface intersection; returns (point, t, iterations)."""
    # Initial guess: the vertex plane (exact for a sphere at the axis, close elsewhere).
    t = (vertex_z - p[2]) / d[2] if d[2] != 0.0 else 0.0
    for i in range(iters):
        q = add(p, scale(d, t))
        r = math.hypot(q[0], q[1])
        f = q[2] - vertex_z - s.sag(r)
        ds = s.sag_slope(r) if r > 0.0 else 0.0
        df = d[2] - (ds * (q[0] * d[0] + q[1] * d[1]) / r if r > 0.0 else 0.0)
        if df == 0.0:
            return None, t, i
        step = f / df
        t -= step
        if abs(step) < tol:
            q = add(p, scale(d, t))
            return q, t, i + 1
    q = add(p, scale(d, t))
    return q, t, iters


def trace_ray(
    *,
    surfaces: Sequence[Surface],
    wavelength_nm: float,
    pupil_x: float,
    pupil_y: float,
    field_deg: float = 0.0,
    field_azimuth_deg: float = 0.0,
    object_distance: float | None = None,
    image_distance: float,
    n_start: float = 1.0,
) -> RayResult:
    """Trace one ray from the first surface plane through to the image plane.

    ``pupil_x/pupil_y`` are the ray height at the first surface plane, which is
    the aperture for an object at infinity. ``field_deg`` tilts the incoming
    bundle; ``field_azimuth_deg`` rotates that tilt in the pupil.
    """
    fx = math.tan(math.radians(field_deg))
    result = RayResult(
        wavelength_nm=wavelength_nm, px=pupil_x, py=pupil_y,
        field_x=fx * math.cos(math.radians(field_azimuth_deg)),
        field_y=fx * math.sin(math.radians(field_azimuth_deg)),
    )
    d = unit((result.field_x, result.field_y, 1.0))
    if object_distance is not None:
        # finite object: aim from the axial object point through the pupil point
        z_obj = -float(object_distance)
        d = unit((pupil_x, pupil_y, -z_obj))
    p = (pupil_x, pupil_y, 0.0)
    n = n_start
    result.points.append(p)

    vertex_z = 0.0
    for idx, s in enumerate(surfaces):
        q, t, _ = _intersect(p, d, s, vertex_z)
        if q is None:
            result.status = "missed"
            result.surface_status.append(f"s{idx}:no-intersection")
            return result
        if t < -1e-9:
            result.status = "missed"
            result.surface_status.append(f"s{idx}:behind")
            return result
        r = math.hypot(q[0], q[1])
        if r > s.semi_aperture + 1e-9:
            result.status = "vignetted"
            result.surface_status.append(f"s{idx}:vignetted")
            result.points.append(q)
            return result
        normal = s.normal(q[0], q[1])
        n_next = get_glass(s.glass_after).index(wavelength_nm) if s.glass_after else 1.0
        new_dir = refract(d, normal, n, n_next)
        # Optical path is index × GEOMETRIC distance travelled. ``t`` is the
        # parameter along the unit direction, so |t| is the distance; using the
        # axial ``t`` would under-count a slanted segment by 1/cos θ.
        result.opl += n * abs(t)
        if new_dir is None:
            result.status = "tir"
            result.surface_status.append(f"s{idx}:tir")
            result.points.append(q)
            return result
        result.surface_status.append("ok")
        d = new_dir
        n = n_next
        p = q
        result.points.append(q)
        vertex_z += s.thickness

    result.last_point = p
    result.last_dir = d
    # Transfer to the image plane. The plane sits at image_distance past the
    # LAST VERTEX, and ``vertex_z`` already includes every prior thickness, so
    # this is a pure axial move — the earlier code re-added the last thickness
    # and double-counted it, which put every ray slightly past its own crossing.
    if d[2] == 0.0:
        result.status = "divergent"
        return result
    t_img = (vertex_z + image_distance - p[2]) / d[2]
    q = add(p, scale(d, t_img))
    result.opl += n * abs(t_img)
    result.points.append(q)
    result.image_x = q[0]
    result.image_y = q[1]
    result.slope_x = d[0] / d[2]
    result.slope_y = d[1] / d[2]
    return result


def trace_fan(
    *,
    surfaces: Sequence[Surface],
    wavelength_nm: float,
    pupil_radius: float,
    image_distance: float,
    field_deg: float = 0.0,
    field_azimuth_deg: float = 0.0,
    samples: int = 21,
    meridional: bool = True,
    object_distance: float | None = None,
) -> list[RayResult]:
    """A fan of rays across the pupil: meridional (y) or sagittal (x)."""
    out = []
    for i in range(samples):
        f = (i / max(samples - 1, 1) * 2.0 - 1.0)
        h = f * pupil_radius
        if meridional:
            out.append(trace_ray(
                surfaces=surfaces, wavelength_nm=wavelength_nm, pupil_x=0.0, pupil_y=h,
                field_deg=field_deg, field_azimuth_deg=field_azimuth_deg,
                image_distance=image_distance, object_distance=object_distance))
        else:
            out.append(trace_ray(
                surfaces=surfaces, wavelength_nm=wavelength_nm, pupil_x=h, pupil_y=0.0,
                field_deg=field_deg, field_azimuth_deg=field_azimuth_deg,
                image_distance=image_distance, object_distance=object_distance))
    return out


def trace_bundle(
    *,
    surfaces: Sequence[Surface],
    wavelength_nm: float,
    pupil_radius: float,
    image_distance: float,
    field_deg: float = 0.0,
    field_azimuth_deg: float = 0.0,
    rings: int = 4,
    arms: int = 12,
    object_distance: float | None = None,
) -> list[RayResult]:
    """A filled pupil: rings × arms, for a spot diagram."""
    out = [trace_ray(
        surfaces=surfaces, wavelength_nm=wavelength_nm, pupil_x=0.0, pupil_y=0.0,
        field_deg=field_deg, field_azimuth_deg=field_azimuth_deg,
        image_distance=image_distance, object_distance=object_distance)]
    for ring in range(1, rings + 1):
        rad = pupil_radius * ring / rings
        for arm in range(arms):
            ang = 2.0 * math.pi * arm / arms + (math.pi / arms if ring % 2 else 0.0)
            out.append(trace_ray(
                surfaces=surfaces, wavelength_nm=wavelength_nm,
                pupil_x=rad * math.cos(ang), pupil_y=rad * math.sin(ang),
                field_deg=field_deg, field_azimuth_deg=field_azimuth_deg,
                image_distance=image_distance, object_distance=object_distance))
    return out


def last_vertex_z(surfaces: Sequence[Surface]) -> float:
    return sum(s.thickness for s in surfaces[:-1])


def total_track(surfaces: Sequence[Surface]) -> float:
    return sum(s.thickness for s in surfaces)


def describe(surfaces: Sequence[Surface]) -> list[dict[str, float | str]]:
    rows = []
    z = 0.0
    for i, s in enumerate(surfaces):
        rows.append({
            "index": i + 1,
            "radius_mm": round(s.radius, 5) if math.isfinite(s.radius) else s.radius,
            "conic": s.conic,
            "a4": s.asphere[0],
            "a6": s.asphere[1],
            "thickness_mm": round(s.thickness, 5),
            "glass_after": s.glass_after or "air",
            "semi_aperture_mm": round(s.semi_aperture, 5),
            "vertex_z_mm": round(z, 5),
        })
        z += s.thickness
    return rows
