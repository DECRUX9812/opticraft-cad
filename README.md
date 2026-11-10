# Opticraft — a first-principles computational optics & ray-tracing CAD engine

A sequential **exact vector-Snell ray tracer**, an **automatic lens optimizer**, a
**diffraction metrology** stack (wavefront → PSF → MTF), an **interactive bench**,
and a **programmatically generated 1080p demo film**. No optical libraries: the
physics is built here and verified against analytic theory.

```
axis: +z is the direction of light;  a positive radius puts the centre of
      curvature on the image side.  Rays are (point, direction) in 3-D.
```

## What is verified (and how)

Every claim in this repo is a comparison between two independent derivations,
printed by `python3 tests/verify_physics.py` (exit status gates the build):

| Claim | Method | Result |
|---|---|---|
| **Paraxial convergence** | exact trace at y→0 vs ABCD matrix, BK7 singlet R=±51.5 t=4 | **3.1e-08 mm** agreement |
| **Paraxial BFD** | thick-lens formula vs matrix solve | **49.157726 mm** both |
| **Spherical aberration** | marginal vs paraxial focus from traced rays | computed at every aperture, **undercorrected** |
| **Premises of the build spec** | the spec quoted 46.85 mm / LSA −2.31 at y=10 mm | engine computes **45.899 mm / −3.258 mm** for that lens (the spec's figure belongs to R=±52.5, t=5) — reported, not adopted |
| **Defocus identity** | wavefront PV vs δ·NA²/2 | within **5%** at 0.5–2 mm defocus |
| **Chromatic correction** | C→F focus shift, singlet vs achromat | **9.8× less** colour in the doublet |
| **Spot vs Airy** | RMS spot / (1.22 λ f/#) | singlet **585×**, asphere **17×**, achromat **1.7×** |
| **PSF scale** | flat-pupil first null vs 1.22/fill px | **8 px vs 9.8 px** expected |
| **MTF** | MTF50 inside the diffraction cutoff | **78 of 426 lp/mm** at f/4 |
| **DLS optimizer** | merit before/after from a degraded doublet | **10 136×**, EFL held to 100.002 mm |

Two caveats are stated rather than hidden: the paraxial/real agreement is a
*convergence* test (finite apertures differ by exactly the aberration), and the
PSF is measured where the spot is tightest, because a PSF at the paraxial focus
of an undercorrected lens is dominated by defocus.

## The engine

| Module | Contents |
|---|---|
| `engine/glass.py` | 3-term Sellmeier; 10 Schott/Heraeus glasses, each checked against published nd and Vd |
| `engine/raytracer.py` | ray–conic/asphere intersection by Newton iteration on the sag, exact vector Snell, TIR and vignetting reported |
| `engine/paraxial.py` | ABCD matrices, EFL/BFD/FFD, f/#, curvature sum |
| `engine/metrics.py` | spot statistics, LSA, chromatic shift, wavefront integrated from transverse aberration, FFT PSF with a self-calibrated Airy ruler, MTF |
| `engine/optimizer.py` | Levenberg–Marquardt with a numeric Jacobian, penalties for focal length / edge thickness / buildability |
| `engine/presets.py` | singlet, aspheric singlet, Fraunhofer achromat, Cooke triplet |
| `engine/api.py` | JSON payloads used by both the web app and the film |

Design decisions worth naming:

* **The wavefront comes from ray data.** W(y) = −(1/R)∫A(y)dy, so the PSF and MTF
  are derived from the same traced rays that pass the convergence test. The defocus
  identity above is what proves the integral and the reference radius are right.
* **The PSF calibrates its own ruler.** A flat pupil's first null sits at exactly
  1.22 λ f/#, so the code transforms a perfect aperture, finds that null, and scales
  every later PSF by it. A wrong pixel scale cannot silently survive.
* **The presets are outputs, not inputs.** The achromat radii shipped in
  `presets.py` are where the optimizer converged; the optimizer starts from a
  mis-bent copy so the "Auto-optimize" button demonstrates a real improvement.

## The workbench

```bash
python3 app.py            # http://127.0.0.1:8770
```

FastAPI/uvicorn when present, otherwise the same routes over stdlib
`http.server`, so the bench runs on a bare Python. Live cross-section with real
surface sag and glass shading, three wavelength bundles (F 486.1, d 587.6,
C 656.3 nm), aperture/field sliders, spot diagrams against the Airy circle,
transverse ray fan, telemetry, and an **Auto-optimize (DLS)** button that streams
the run.

The bench draws the glass at a magnified scale and the air path compressed, the
way layout drawings do — a single scale would make a 100 mm focal length render
elements 7 mm thick as an invisible sliver.

## The film

```bash
python3 scripts/film_data.py            # ~10 s: traces everything once
python3 scripts/render_film.py /tmp/film_data.pkl demo_showcase.mp4
```

**`demo_showcase.mp4` — 1920×1080, 60 fps, 55 s, H.264, 2.8 MB, silent.**

| Act | Time | What is on screen |
|---|---|---|
| 1 · the proof | 0–10 s | the convergence table (3.1e-08 mm) and the singlet caustic: marginal rays crossing 5.3 mm short of the paraxial focus |
| 2 · interact | 10–25 s | aperture opening 1→25 mm, then field 0→10°, with three-band spot diagrams and the Airy ring |
| 3 · optimize | 25–45 s | 30 DLS iterations morphing the elements while the merit curve falls and the spot collapses into the Airy disc |
| 4 · verify | 45–55 s | Cooke triplet, MTF to 100 lp/mm against the diffraction limit, scorecard |

It is rendered frame-by-frame from the same `engine` package the app serves, so
a frame cannot drift from the physics, and it needs no browser, GPU or display.

## Reproduce

```bash
bash check.sh        # physics verification + API smoke test + film metadata
```

Requires Python 3.11+ with numpy, matplotlib-free rendering via Pillow, and
ffmpeg. No scipy, no optical package, no browser.

## Honest limits

* **Meridional and rotational pupil sampling.** Enough for spherical aberration,
  axial colour, defocus, coma estimate and the MTF on axis. Not a full 2-D
  diffraction field analysis.
* **Sequential only.** No non-sequential paths, no ghosts, no coatings or
  polarisation; refraction is geometric with real dispersion.
* **Cemented doublets and air-spaced triplets**, not a general N-element
  optimizer: the DLS varies curvatures and air spaces, not glass selection.
* **The triplet's radii** are scaled from the published 1934 Cooke design, and
  are presented as a preset rather than as a reproduction of that paper.

## Licence

MIT. Sellmeier coefficients: Schott/Heraeus data sheets, cross-checked against
published nd/Vd in the verification suite.
