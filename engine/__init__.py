"""Opticraft engine — first-principles computational optics.

Modules
-------
``glass``      Sellmeier dispersion and the optical-glass catalogue
``raytracer``  exact vector-Snell sequential trace over conic/aspheric surfaces
``paraxial``   ABCD ray-transfer matrices, EFL/BFD, cardinal points
``metrics``    spot statistics, LSA, chromatic shift, wavefront, PSF, MTF
``optimizer``  damped least squares (Levenberg–Marquardt) lens design
``presets``    the four systems the bench and the film walk through
``api``        JSON payloads shared by the web app and the film renderer
"""

from __future__ import annotations

__version__ = "1.0.0"

from .glass import GLASSES, WAVELENGTHS, Glass, catalogue, glass  # noqa: F401
from .raytracer import (  # noqa: F401
    RayResult,
    Surface,
    refract,
    trace_bundle,
    trace_fan,
    trace_ray,
)
from .paraxial import solve as paraxial_solve  # noqa: F401
