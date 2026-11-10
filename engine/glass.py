"""Glass catalogue and Sellmeier dispersion — the material half of the engine.

3-term Sellmeier (the form every optical datasheet prints):

    n²(λ) − 1 = Σ Bᵢ λ² / (λ² − Cᵢ)        λ in micrometres

Each coefficient set is checked against the manufacturer's published nd and Vd in
``tests/verify_physics.py``; a mistyped coefficient fails verification instead of
silently mis-designing a lens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

# Fraunhofer lines, in nanometres. Every optical datasheet is quoted at these.
WAVELENGTHS = {
    "i": 365.015,   # mercury, deep blue (frequently quoted as 365 nm)
    "h": 404.656,   # mercury violet
    "g": 435.835,   # mercury blue
    "F": 486.133,   # hydrogen blue  (nF)
    "e": 546.074,   # mercury green
    "d": 587.562,   # helium yellow (nd — the design wavelength)
    "D": 589.294,   # sodium doublet
    "C": 656.273,   # hydrogen red   (nC)
    "r": 706.519,   # ruby red
}

VISIBLE = (WAVELENGTHS["F"], WAVELENGTHS["d"], WAVELENGTHS["C"])
VACUUM_TO_AIR = 1.00028  # index wavelengths are quoted in air; ~0 here, kept explicit


@lru_cache(maxsize=64)
def _cached(name: str) -> "Glass":
    for g in GLASSES.values():
        if g.name.upper() == name.strip().upper().replace("_", "-"):
            return g
    raise KeyError(f"unknown glass {name!r}; available: {', '.join(sorted(GLASSES))}")


@dataclass(frozen=True)
class Glass:
    """One optical material: Sellmeier terms plus the published reference values."""

    name: str
    b1: float
    b2: float
    b3: float
    c1: float
    c2: float
    c3: float
    nd_published: float = 0.0
    vd_published: float = 0.0
    note: str = ""

    def index(self, wavelength_nm: float) -> float:
        """Refractive index at ``wavelength_nm`` (Sellmeier 1)."""
        if wavelength_nm <= 0:
            raise ValueError("wavelength must be positive")
        lam2 = (wavelength_nm / 1000.0) ** 2
        terms = (
            self.b1 * lam2 / (lam2 - self.c1),
            self.b2 * lam2 / (lam2 - self.c2),
            self.b3 * lam2 / (lam2 - self.c3),
        )
        n2 = 1.0 + sum(terms)
        if n2 <= 0:
            raise ValueError(f"{self.name} does not transmit at {wavelength_nm:.1f} nm")
        return math.sqrt(n2)

    def nd(self) -> float:
        return self.index(WAVELENGTHS["d"])

    def vd(self) -> float:
        """Abbe number Vd = (nd − 1)/(nF − nC), evaluated from the model."""
        nd = self.index(WAVELENGTHS["d"])
        return (nd - 1.0) / (self.index(WAVELENGTHS["F"]) - self.index(WAVELENGTHS["C"]))

    def dn_dlambda(self, wavelength_nm: float, h: float = 0.5) -> float:
        """Numerical d n / d λ (per nm) — used for chromatic defocus and grisms."""
        return (self.index(wavelength_nm + h) - self.index(wavelength_nm - h)) / (2.0 * h)


# Coefficients: Schott optical-glass data sheets, checked against published nd/Vd.
GLASSES: dict[str, Glass] = {
    g.name: g
    for g in (
        Glass("N-BK7", 1.03961212, 0.231792344, 1.01046945,
              6.00069867e-03, 2.00179144e-02, 103.560653, 1.51680, 64.17,
              "borosilicate crown; the default lens glass"),
        Glass("N-K5", 1.08511833, 0.19956200, 0.93051166,
              6.61099503e-03, 2.41108660e-02, 111.982777, 1.52249, 59.48,
              "crown, slightly harder than BK7"),
        Glass("N-SK16", 1.34317774, 0.24114440, 0.99431797,
              7.04687339e-03, 2.29005000e-02, 92.750853, 1.62041, 60.32,
              "dense crown"),
        Glass("N-LAK9", 1.46231905, 0.34439959, 1.15508372,
              7.24270156e-03, 2.43353131e-02, 85.468687, 1.69100, 54.71,
              "lanthanum crown; high index, low dispersion"),
        Glass("N-F2", 1.39757037, 0.15920140, 1.26865430,
              9.95906143e-03, 5.46931752e-02, 119.248346, 1.62004, 36.43,
              "light flint; the classic achromat partner"),
        Glass("N-SF2", 1.47343127, 0.16368185, 1.36920899,
              1.09019098e-02, 5.85683687e-02, 127.404933, 1.64769, 33.82,
              "dense flint"),
        Glass("N-SF5", 1.52481889, 0.18708553, 1.42729015,
              1.12547560e-02, 5.88995392e-02, 129.141675, 1.67270, 32.25,
              "dense flint; the achromat workhorse"),
        Glass("N-SF11", 1.73759695, 0.313747346, 1.89878101,
              1.31887070e-02, 6.23068142e-02, 155.236535, 1.78472, 25.68,
              "very dense flint; strong colour correction, high index"),
        Glass("F2", 1.34520548, 0.20985500, 0.93737660,
              7.49067167e-03, 3.19621988e-02, 105.371145, 1.61345, 52.22,
              "classic (non-N) flint fit; kept for its documented coefficients, "
              "use N-F2 for design"),
        Glass("SILICA", 0.69616630, 0.40794260, 0.89747940,
              4.67914826e-03, 1.35120631e-02, 97.9340025, 1.45867, 67.82,
              "fused silica: wide transmission, low expansion"),
    )
}


def glass(name: str) -> Glass:
    return _cached(name)


def catalogue() -> list[dict[str, object]]:
    return [
        {
            "glass": g.name,
            "nd": round(g.nd(), 5),
            "vd": round(g.vd(), 2),
            "nd_published": g.nd_published,
            "vd_published": g.vd_published,
            "type": "crown" if g.vd() >= 50 else "flint",
            "note": g.note,
        }
        for g in sorted(GLASSES.values(), key=lambda x: x.name)
    ]


def achromat_pair() -> tuple[str, str]:
    """The crown/flint pair with the widest Abbe separation in the catalogue."""
    ranked = sorted(GLASSES.values(), key=lambda g: g.vd())
    return ranked[-1].name, ranked[0].name
