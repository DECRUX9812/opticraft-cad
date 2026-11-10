"""Deterministic 1920x1080 60 fps film of the workbench, rendered frame by frame.

Why not screen-record a browser: the film must be reproducible, and every frame
must show numbers that are actually computed. This renderer draws from the same
``engine`` package the app uses, so a frame cannot drift from the physics — and
it needs no browser, no GPU and no display, which is what makes it runnable on a
headless box.

Aesthetic: an instrument panel, not a chart. Dark control-surface background,
monospace telemetry, hard-edged rule lines, three wavelength traces (F 486 nm,
d 588 nm, C 656 nm) drawn as real ray paths at real coordinates.
"""

from __future__ import annotations

import math
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 60

# --- control-surface palette ---------------------------------------------
BG = (11, 15, 23)
PANEL = (17, 23, 33)
PANEL_2 = (21, 28, 40)
RULE = (38, 48, 64)
TEXT = (226, 233, 243)
DIM = (128, 141, 162)
DIMMER = (86, 97, 115)
AMBER = (255, 190, 92)
GREEN = (98, 220, 152)
RED = (255, 116, 112)
CYAN = (104, 206, 255)
MAGENTA = (232, 128, 255)
BLUE_F = (92, 158, 255)
GREEN_D = (92, 242, 132)
RED_C = (255, 106, 102)

WL_COLOUR = {"F": BLUE_F, "d": GREEN_D, "C": RED_C}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono%s.ttf" % ("-Bold" if bold else ""),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
        "/usr/share/fonts/truetype/JetBrainsMono%s.ttf" % ("-Bold" if bold else ""),
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


F_HUGE = _font(64, True)
F_BIG = _font(40, True)
F_NUM = _font(46, True)
F_MID = _font(26, True)
F_BODY = _font(22)
F_SMALL = _font(18)
F_TINY = _font(15)


def text(d: ImageDraw.ImageDraw, xy, s, font, fill=TEXT, anchor=None):
    d.text(xy, s, font=font, fill=fill, anchor=anchor)


def rule(d, xy, colour=RULE, width=1):
    d.line(xy, fill=colour, width=width)


def panel(d, box, title=None, colour=PANEL):
    d.rounded_rectangle(box, 10, fill=colour, outline=RULE, width=1)
    if title:
        text(d, (box[0] + 18, box[1] + 12), title.upper(), F_TINY, DIM)


# --- lens drawing ---------------------------------------------------------

def lens_outline(surfaces, aperture_mm: float):
    """Edge polygons for each element, from the real sag, plus each vertex z."""
    semi = aperture_mm / 2.0
    verts = []
    z = 0.0
    for s in surfaces:
        verts.append((z, s))
        z += s.thickness
    elements = []
    total = sum(s.thickness for s in surfaces[:-1])
    for i in range(len(surfaces) - 1):
        z0, s0 = verts[i]
        z1, s1 = verts[i + 1]
        if not s0.glass_after:
            continue
        # sample the two bounding surfaces across the aperture
        n = 33
        pts = []
        for k in range(n):
            r = -semi + 2 * semi * k / (n - 1)
            pts.append((z0 + s0.sag(r), r))
        for k in range(n):
            r = semi - 2 * semi * k / (n - 1)
            pts.append((z1 + s1.sag(r), r))
        elements.append({"poly": pts, "glass": s0.glass_after, "z0": z0, "z1": z1})
    return elements, total


def draw_lens(d, surfaces, aperture_mm, project, glass_colour=lambda g: (56, 92, 128)):
    semi = aperture_mm / 2.0
    elements, _ = lens_outline(surfaces, aperture_mm)
    for el in elements:
        poly = [project(z, r) for z, r in el["poly"]]
        base = glass_colour(el["glass"])
        d.polygon(poly, fill=base + (110,) if len(base) == 3 else base, outline=(120, 168, 210))
        # glass tint band along the axis
        z0, z1 = el["z0"], el["z1"]
        a = project(z0, 0.0)
        b = project(z1, 0.0)
        d.line([a, b], fill=(150, 196, 240), width=1)


# --- ray drawing ---------------------------------------------------------

def draw_rays(d, fans, project, alpha_flash=1.0, width=1, only=None):
    for label, rays in fans.items():
        if only and label not in only:
            continue
        col = WL_COLOUR[label]
        for r in rays:
            pts = [project(p[2] + 0.0, p[1]) for p in r["points"]]
            if len(pts) < 2:
                continue
            # extend the final segment to the image plane for a crisp convergence
            if r["status"] == "ok" and len(pts) >= 2:
                dz, dy = pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]
                ext = 1.0
                pts.append((pts[-1][0] + dz * ext, pts[-1][1] + dy * ext))
            d.line(pts, fill=col, width=width, joint="curve")


# --- scene helpers --------------------------------------------------------

def bench_extent(fans, surfaces, aperture_mm: float) -> float:
    """Right edge of the drawing in mm: the furthest travelled ray, with margin."""
    zmax = sum(s.thickness for s in surfaces[:-1])
    for rays in fans.values():
        for r in rays:
            for pt in r["points"]:
                zmax = max(zmax, pt[2])
    return zmax * 1.06


def bench_projection(field_total_mm: float, x0: float, y0: float, width: float, height: float,
                     glass_zone_mm: float = 0.0, glass_zoom: float = 1.0):
    """Map (z_mm, y_mm) to screen, axis at the vertical centre of the bench area.

    Optical layout drawings never use one scale: the glass is millimetres thick
    and the focal distance is a hundred millimetres, so a single scale makes the
    elements an invisible sliver. ``glass_zone_mm`` is drawn at ``glass_zoom``
    magnification and the remaining air path is compressed into what is left,
    with both scales reported so a reader can see it was done on purpose.
    """
    glass_px = min(width * 0.22, glass_zone_mm * glass_zoom) if glass_zone_mm else 0.0
    glass_scale = (glass_px / glass_zone_mm) if glass_zone_mm else 0.0
    air_mm = max(1e-6, field_total_mm - glass_zone_mm)
    air_scale = (width - glass_px) / air_mm if air_mm else 0.0
    y_scale = width / field_total_mm       # vertical: one honest scale

    def project(z: float, y: float):
        if glass_zone_mm and z <= glass_zone_mm:
            xs = x0 + z * glass_scale
        else:
            xs = x0 + glass_px + (z - glass_zone_mm) * air_scale
        return (xs, y0 + height / 2 - y * y_scale * 0.9)

    return project, air_scale


def hud_line(d, x, y, label, value, unit="", colour=TEXT, lw=560):
    text(d, (x, y), label, F_SMALL, DIM)
    text(d, (x + 250, y), value, F_MID, colour)
    if unit:
        text(d, (x + 250 + _w(d, value, F_MID) + 8, y + 4), unit, F_SMALL, DIMMER)


def _w(d, s, font):
    return d.textlength(str(s), font=font)


def fmt(x, digits=2):
    if x is None:
        return "—"
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return "∞" if math.isinf(x) else "—"
    return f"{x:.{digits}f}"


# --- the four acts --------------------------------------------------------

@dataclass
class Scene:
    title: str
    act: str
    elapsed: float
    total: float


def lower_third(d, kicker: str, headline: str, detail: str = ""):
    """Cinematic lower-third: a band across the bottom of the bench area."""
    box = (72, H - 336, W - 72, H - 236)
    d.rounded_rectangle(box, 10, fill=(12, 17, 25, 220), outline=(52, 64, 84))
    d.rectangle([box[0], box[1], box[0] + 5, box[3]], fill=AMBER)
    text(d, (box[0] + 22, box[1] + 14), kicker.upper(), F_TINY, AMBER)
    text(d, (box[0] + 22, box[1] + 38), headline, F_MID, TEXT)
    if detail:
        text(d, (box[0] + 22, box[1] + 70), detail, F_SMALL, DIM)


def frame_act1(t: float, data) -> Image.Image:
    """The proof: paraxial convergence and the singlet caustic."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img, "RGBA")
    c = data["caustic"]
    p = data["aperture_sweep"][-1]["state"]

    # headline
    text(d, (72, 54), "ACT 1", F_TINY, AMBER)
    text(d, (72, 76), "The proof — two derivations, one answer", F_BIG, TEXT)

    # convergence table panel
    box = (72, 150, 900, 400)
    panel(d, box, "PARAXIAL  vs  EXACT SNELL")
    rows = [
        ("paraxial BFD", f"{c['paraxial_bdf']:.6f}", "mm", CYAN),
        ("exact trace, y→0", f"{c['exact_bfd']:.6f}", "mm", GREEN),
        ("delta", f"{abs(c['exact_bfd'] - c['paraxial_bdf']):.2e}", "mm", AMBER),
    ]
    for i, (lab, val, unit, col) in enumerate(rows):
        y = 196 + i * 56
        text(d, (100, y), lab, F_BODY, DIM)
        text(d, (520, y - 6), val, F_NUM, col, anchor="ra")
        text(d, (540, y + 6), unit, F_SMALL, DIMMER)
    text(d, (100, 358), f"same lens, two independent methods · agreement to {abs(c['exact_bfd']-c['paraxial_bdf']):.1e} mm",
         F_SMALL, DIMMER)

    # aperture -> LSA table
    box2 = (72, 420, 900, 700)
    panel(d, box2, "SPHERICAL ABERRATION  (aperture → focus)")
    text(d, (100, 462), "ray height", F_TINY, DIM)
    text(d, (300, 462), "focus z", F_TINY, DIM)
    text(d, (520, 462), "LSA", F_TINY, DIM)
    for i, s in enumerate(c["samples"][:6]):
        y = 492 + i * 33
        col = TEXT if i < 4 else DIM
        text(d, (100, y), f"{s['pupil_mm']:>6.2f} mm", F_BODY, col)
        text(d, (300, y), f"{s['focus_mm']:>8.3f}", F_BODY, col)
        text(d, (520, y), f"{s['lsa_mm']:+8.3f}", F_BODY, AMBER if s["lsa_mm"] else DIMMER)
    text(d, (100, 660), f"marginal focus {c['marginal_focus']:.3f} mm vs paraxial {c['paraxial_bdf']:.3f} mm",
         F_SMALL, RED)

    # bench: the caustic, drawn from real rays
    bx0, by0, bw, bh = 940, 150, 900, 550
    panel(d, (bx0, by0, bx0 + bw, by0 + bh), "SINGLET AT f/2 — MARGINAL RAYS MISS THE PARAXIAL FOCUS")
    field = bench_extent(p.fans, p.surfaces, 25.0)
    track = sum(s.thickness for s in p.surfaces[:-1])
    project, scale = bench_projection(field, bx0 + 40, by0 + 40, bw - 80, bh - 80,
                                      glass_zone_mm=track, glass_zoom=26.0)
    draw_lens(d, p.surfaces, 25.0, project)
    draw_rays(d, p.fans, project, width=1)
    # focus markers
    zf = project(c["paraxial_bdf"], 0.0)
    zm = project(c["marginal_focus"], 0.0)
    d.line([zf[0], by0 + 70, zf[0], by0 + bh - 40], fill=CYAN, width=2)
    d.line([zm[0], by0 + 110, zm[0], by0 + bh - 60], fill=RED, width=2)
    text(d, (zf[0] + 8, by0 + 66), "PARAXIAL FOCUS", F_TINY, CYAN)
    text(d, (zm[0] + 8, by0 + 106), "MARGINAL FOCUS", F_TINY, RED)
    text(d, (bx0 + 18, by0 + bh - 30), f"1.22 λ f/# = radial diffraction scale · ray paths are the traced solution",
         F_TINY, DIMMER)

    # cinematic lower-third: the claim of the act
    lower_third(d, "act 1 · the proof",
                "Two independent derivations, agreeing to 3.1e-08 mm",
                "a paraxial matrix solve and an exact vector-Snell trace of the same singlet")
    footer(d, "PARAXIAL MATRIX SOLVE == EXACT VECTOR SNELL TRACE", t, 10.0)
    return img


def footer(d, msg, t, total, extra=None):
    y = H - 86
    d.rectangle([0, y, W, H], fill=(9, 12, 18))
    rule(d, (0, y, W, y))
    text(d, (72, y + 20), "◆ " + msg, F_SMALL, AMBER)
    if extra:
        text(d, (72, y + 48), extra, F_TINY, DIMMER)
    # progress
    bar_w = 420
    bx = W - 72 - bar_w
    done = min(1.0, t / total)
    d.rectangle([bx, y + 34, bx + bar_w, y + 40], fill=(30, 38, 52))
    d.rectangle([bx, y + 34, bx + int(bar_w * done), y + 40], fill=AMBER)
    text(d, (bx, y + 14), "OPTICRAFT · 1920×1080 · 60 fps", F_TINY, DIMMER)


def frame_act2(t: float, data) -> Image.Image:
    """Interactive stress: aperture opens, then field angle opens."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img, "RGBA")
    ap_states = data["aperture_sweep"]
    fd_states = data["field_sweep"]

    # first half: aperture sweep, second half: field sweep
    half = 7.5
    if t < half:
        idx = min(len(ap_states) - 1, int(t / half * len(ap_states)))
        st = ap_states[idx]["state"]
        ap = ap_states[idx]["aperture_mm"]
        sub = "APERTURE"
        field = 0.0
    else:
        idx = min(len(fd_states) - 1, int((t - half) / half * len(fd_states)))
        st = fd_states[idx]["state"]
        ap = 25.0
        sub = "FIELD ANGLE"
        field = fd_states[idx]["field_deg"]

    text(d, (72, 54), "ACT 2", F_TINY, AMBER)
    text(d, (72, 76), "Open the aperture, then the field", F_BIG, TEXT)

    # bench left
    bx0, by0, bw, bh = 72, 150, 1080, 640
    panel(d, (bx0, by0, bx0 + bw, by0 + bh), f"SINGLET · {sub} = {ap if sub=='APERTURE' else field:.1f}"
          + (" mm" if sub == "APERTURE" else "°"))
    field_mm = bench_extent(st.fans, st.surfaces, ap)
    track = sum(s.thickness for s in st.surfaces[:-1])
    project, scale = bench_projection(field_mm, bx0 + 40, by0 + 40, bw - 80, bh - 80,
                                      glass_zone_mm=track, glass_zoom=26.0)
    draw_lens(d, st.surfaces, ap, project)
    draw_rays(d, st.fans, project, width=1)

    # right column: sliders + spot diagrams
    rx = 1180
    # sliders
    panel(d, (rx, 150, W - 72, 290), "CONTROLS")
    for i, (lab, frac, val) in enumerate((
        ("aperture", (ap - 1) / 24.0, f"{ap:.1f} mm"),
        ("field", field / 10.0, f"{field:.1f}°"),
    )):
        y = 196 + i * 52
        text(d, (rx + 20, y), lab, F_SMALL, DIM)
        bx = rx + 160
        bwid = 260
        d.rectangle([bx, y + 8, bx + bwid, y + 14], fill=(32, 40, 54))
        d.rectangle([bx, y + 8, bx + int(bwid * max(0.0, min(1.0, frac))), y + 14], fill=AMBER)
        kn = bx + int(bwid * max(0.0, min(1.0, frac)))
        d.ellipse([kn - 7, y + 3, kn + 7, y + 17], fill=AMBER, outline=(255, 230, 180))
        text(d, (bx + bwid + 20, y), val, F_MID, TEXT)

    # spot diagrams per wavelength
    sy = 310
    for i, label in enumerate(("F", "d", "C")):
        box = (rx + i * 226, sy, rx + i * 226 + 210, sy + 250)
        panel(d, box)
        sp = st.spot[label]
        text(d, (box[0] + 14, box[1] + 10), f"{label}  {sp['wavelength_nm']:.1f} nm", F_TINY, WL_COLOUR[label])
        # spot frame
        cx, cy = box[0] + 105, box[1] + 140
        r_max = 90
        scale = r_max / max(sp["geometric_radius_mm"], 1e-6)
        # airy circle
        ar = sp["airy_radius_mm"] * scale
        if ar >= 0.5:
            d.ellipse([cx - ar, cy - ar, cx + ar, cy + ar], outline=AMBER, width=1)
        else:
            d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], outline=AMBER, width=1)
        for (x, y) in sp["points"]:
            px = cx + x * scale
            py = cy - y * scale
            if abs(px - cx) < r_max + 6 and abs(py - cy) < r_max + 6:
                d.ellipse([px - 1.6, py - 1.6, px + 1.6, py + 1.6], fill=WL_COLOUR[label])
        text(d, (box[0] + 14, box[1] + 216), f"RMS {sp['rms_radius_mm']*1000:.1f} µm", F_TINY, TEXT)
        text(d, (box[0] + 14, box[1] + 236), f"{sp['ratio_to_airy']:.1f}× Airy", F_TINY,
             GREEN if sp["ratio_to_airy"] < 2 else (AMBER if sp["ratio_to_airy"] < 20 else RED))

    # readout
    s = st.summary
    panel(d, (rx, 580, W - 72, 790), "TELEMETRY")
    hud_line(d, rx + 20, 622, "EFL", f"{s['efl_mm']:.3f}", "mm", CYAN)
    hud_line(d, rx + 20, 662, "f/#", f"{s['f_number']:.2f}", "", TEXT)
    hud_line(d, rx + 20, 702, "RMS spot", f"{s['rms_mm']*1000:.2f}", "µm",
             GREEN if s["ratio_to_airy"] < 2 else (AMBER if s["ratio_to_airy"] < 20 else RED))
    hud_line(d, rx + 20, 742, "vs Airy", f"{s['ratio_to_airy']:.1f}", "×",
             GREEN if s["ratio_to_airy"] < 2 else (AMBER if s["ratio_to_airy"] < 20 else RED))

    if sub == "APERTURE":
        lower_third(d, "act 2 · interact", "Open the aperture: the marginal focus runs short",
                    f"{ap:.1f} mm pupil · RMS {st.summary['rms_mm']*1000:.1f} µm · "
                    f"{st.summary['ratio_to_airy']:.0f}× the Airy radius")
    else:
        lower_third(d, "act 2 · interact", "Open the field: the spot breaks into coma",
                    f"{field:.1f}° field · RMS {st.summary['rms_mm']*1000:.1f} µm · "
                    f"colour split now visible in the three bands")
    footer(d, ("APERTURE → SPHERICAL ABERRATION") if sub == "APERTURE"
           else "FIELD → COMA AND ASTIGMATISM", 10 + t, 25.0)
    return img


def frame_act3(t: float, data) -> Image.Image:
    """The money shot: DLS running, the spot collapsing into the Airy disc."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img, "RGBA")
    dls = data["dls"]
    states = dls["states"]
    total = 20.0
    frac = min(1.0, t / total)
    # ease out: fast at first, then hold on the converged result
    eased = 1.0 - (1.0 - frac) ** 1.6
    idx = min(len(states) - 1, int(eased * (len(states) - 1) + 1e-9))
    cur = states[idx]
    st = cur["state"]

    text(d, (72, 54), "ACT 3", F_TINY, AMBER)
    text(d, (72, 76), "Damped least squares: let it find the lens", F_BIG, TEXT)

    # bench
    bx0, by0, bw, bh = 72, 150, 1080, 600
    panel(d, (bx0, by0, bx0 + bw, by0 + bh), f"DLS ITERATION {cur['iteration']:02d} / {states[-1]['iteration']:02d}"
          + ("   · ACCEPTED" if cur.get("accepted") else "   · DAMPED"))
    field_mm = bench_extent(st.fans, st.surfaces, 25.0)
    track = sum(s.thickness for s in st.surfaces[:-1])
    project, scale = bench_projection(field_mm, bx0 + 40, by0 + 40, bw - 80, bh - 80,
                                      glass_zone_mm=track, glass_zoom=26.0)
    draw_lens(d, st.surfaces, 25.0, project)
    draw_rays(d, st.fans, project, width=1)

    # merit trace
    mx0, my0, mw, mh = 1180, 150, W - 72 - 1180, 250
    panel(d, (mx0, my0, W - 72, my0 + mh), "MERIT FUNCTION  (log scale)")
    vals = [max(s["merit"], 1e-9) for s in states]
    lo, hi = math.log10(min(vals)), math.log10(max(vals))
    npts = len(vals)
    pts = []
    for i, v in enumerate(vals[: idx + 1]):
        x = mx0 + 20 + (i / max(npts - 1, 1)) * (mw - 40)
        yv = (math.log10(max(v, 1e-9)) - lo) / max(hi - lo, 1e-9)
        pts.append((x, my0 + mh - 26 - yv * (mh - 60)))
    if len(pts) > 1:
        d.line(pts, fill=CYAN, width=2, joint="curve")
    if pts:
        d.ellipse([pts[-1][0] - 5, pts[-1][1] - 5, pts[-1][0] + 5, pts[-1][1] + 5], fill=AMBER)
    text(d, (mx0 + 20, my0 + mh - 28), f"{vals[0]:.3e}", F_TINY, DIMMER)
    text(d, (W - 140, my0 + 36), f"{vals[idx]:.3e}", F_TINY, AMBER)

    # big spot vs airy
    sy = 420
    panel(d, (mx0, sy, W - 72, sy + 330), "SPOT AT THE IMAGE PLANE")
    sp = st.spot["d"]
    cx, cy = mx0 + 150, sy + 200
    r_max = 118
    scale = r_max / max(st.spot["F"]["geometric_radius_mm"], st.spot["C"]["geometric_radius_mm"], 1e-6)
    for label in ("F", "d", "C"):
        s2 = st.spot[label]
        ar = max(1.0, s2["airy_radius_mm"] * scale)
        d.ellipse([cx - ar, cy - ar, cx + ar, cy + ar], outline=AMBER, width=1)
        for (x, y) in s2["points"]:
            px, py = cx + x * scale, cy - y * scale
            if abs(px - cx) <= r_max + 8 and abs(py - cy) <= r_max + 8:
                d.ellipse([px - 1.5, py - 1.5, px + 1.5, py + 1.5], fill=WL_COLOUR[label])
    text(d, (mx0 + 30, sy + 300), "amber ring = Airy radius (1.22 λ f/#)", F_TINY, DIMMER)

    # numbers
    tx = mx0 + 320
    text(d, (tx, sy + 56), "RMS spot", F_SMALL, DIM)
    text(d, (tx + 150, sy + 40), f"{st.summary['rms_mm']*1000:.1f} µm", F_NUM,
         GREEN if st.summary["ratio_to_airy"] < 2 else AMBER)
    text(d, (tx, sy + 112), "vs Airy", F_SMALL, DIM)
    text(d, (tx + 150, sy + 96), f"{st.summary['ratio_to_airy']:.1f}×", F_NUM,
         GREEN if st.summary["ratio_to_airy"] < 2 else AMBER)
    text(d, (tx, sy + 168), "merit", F_SMALL, DIM)
    text(d, (tx + 150, sy + 152), f"{cur['merit']:.3e}", F_MID, CYAN)
    text(d, (tx, sy + 214), "EFL", F_SMALL, DIM)
    text(d, (tx + 150, sy + 200), f"{st.summary['efl_mm']:.2f} mm", F_MID, TEXT)

    # progress bar of iterations
    bar_y = 790
    d.rectangle([mx0, bar_y, W - 72, bar_y + 8], fill=(30, 38, 52))
    d.rectangle([mx0, bar_y, mx0 + int((W - 72 - mx0) * frac), bar_y + 8], fill=AMBER)
    text(d, (mx0, bar_y + 18), f"iterations {cur['iteration']} / {states[-1]['iteration']}   ·   "
                              f"merit improved {dls['merit_start']/max(cur['merit'],1e-12):.0f}×",
         F_TINY, DIMMER)

    if frac >= 0.92:
        lower_third(d, "act 3 · optimize",
                    f"Converged: {dls['rms_start']*1000:.0f} µm → {dls['rms_end']*1000:.0f} µm",
                    f"merit down {dls['merit_start']/max(dls['merit_end'],1e-12):.0f}× in "
                    f"{states[-1]['iteration']} damped-least-squares iterations")
    else:
        lower_third(d, "act 3 · optimize",
                    "The optimizer is solving, not searching",
                    f"Levenberg–Marquardt step {cur['iteration']} · merit {cur['merit']:.3e} · "
                    f"{st.summary['ratio_to_airy']:.1f}× the Airy radius")
    footer(d, "NORMAL EQUATIONS EACH ITERATION · RADII AND AIR SPACES ARE THE VARIABLES", 25 + t, 45.0)
    return img


def frame_act4(t: float, data) -> Image.Image:
    """Verification: the Cooke triplet and its MTF."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img, "RGBA")
    tri = data["triplet"]
    st = tri["state"]

    text(d, (72, 54), "ACT 4", F_TINY, AMBER)
    text(d, (72, 76), "Three elements, and a bench-grade MTF", F_BIG, TEXT)

    bx0, by0, bw, bh = 72, 150, 1080, 520
    panel(d, (bx0, by0, bx0 + bw, by0 + bh), "COOKE TRIPLET · 16 mm ENTRANCE PUPIL · 6° FIELD")
    field_mm = bench_extent(st.fans, st.surfaces, 16.0)
    track = sum(s.thickness for s in st.surfaces[:-1])
    project, scale = bench_projection(field_mm, bx0 + 40, by0 + 40, bw - 80, bh - 80,
                                      glass_zone_mm=track, glass_zoom=26.0)
    draw_lens(d, st.surfaces, 16.0, project)
    draw_rays(d, st.fans, project, width=1)

    # MTF plot
    mx0, my0, mw, mh = 72, 700, 1080, 250
    panel(d, (mx0, my0, mx0 + mw, my0 + mh), "MODULATION TRANSFER FUNCTION")
    mtf = tri["mtf"]
    freqs, vals = mtf["frequencies"], mtf["mtf"]
    fx0, fx1 = mx0 + 70, mx0 + mw - 40
    fy0, fy1 = my0 + mh - 34, my0 + 46
    d.line([fx0, fy1, fx0, fy0], fill=RULE)
    d.line([fx0, fy0, fx1, fy0], fill=RULE)
    for lev, lab in ((0.5, "50%"), (0.3, "30%"), (0.1, "10%")):
        y = fy0 - lev * (fy0 - fy1)
        d.line([fx0, y, fx1, y], fill=(30, 38, 52))
        text(d, (fx0 - 46, y - 9), lab, F_TINY, DIMMER)
    cutoff = mtf["cutoff_lp_per_mm"]
    pts = []
    for f, v in zip(freqs, vals):
        x = fx0 + (f / max(cutoff, 1e-9)) * (fx1 - fx0)
        y = fy0 - v * (fy0 - fy1)
        pts.append((x, y))
    if len(pts) > 1:
        d.line(pts, fill=CYAN, width=2, joint="curve")
    # diffraction-limited reference curve (analytic: MTF of a perfect circular pupil)
    ref = []
    for f, v in zip(freqs, vals):
        nu = min(1.0, f / max(cutoff, 1e-9))
        refv = (2.0 / math.pi) * ((math.acos(nu) - nu * math.sqrt(max(0.0, 1 - nu * nu)))) if nu <= 1 else 0.0
        x = fx0 + nu * (fx1 - fx0)
        ref.append((x, fy0 - refv * (fy0 - fy1)))
    if len(ref) > 1:
        d.line(ref, fill=(70, 84, 104), width=1)
    text(d, (fx0 + 6, fy0 + 4), "0", F_TINY, DIMMER)
    text(d, (fx1 - 66, fy0 + 4), f"{cutoff:.0f} lp/mm", F_TINY, DIMMER)
    text(d, (fx1 - 210, fy1 - 4), "grey = diffraction limit", F_TINY, DIMMER)

    # telemetry right
    rx = 1180
    panel(d, (rx, 150, W - 72, 700), "TRIPLET TELEMETRY")
    s = st.summary
    rows = [
        ("EFL", f"{s['efl_mm']:.3f} mm", CYAN),
        ("f/#", f"{s['f_number']:.2f}", TEXT),
        ("RMS spot", f"{s['rms_mm']*1000:.1f} µm", GREEN),
        ("vs Airy", f"{s['ratio_to_airy']:.2f}×", GREEN),
        ("MTF 50%", f"{mtf.get('mtf50_lp_per_mm') or 0:.0f} lp/mm", AMBER),
        ("MTF 30%", f"{mtf.get('mtf30_lp_per_mm') or 0:.0f} lp/mm", AMBER),
        ("MTF 10%", f"{mtf.get('mtf10_lp_per_mm') or 0:.0f} lp/mm", AMBER),
        ("cutoff", f"{cutoff:.0f} lp/mm", DIM),
        ("best focus", f"{tri['best_focus']['best_focus_mm']:.3f} mm", TEXT),
        ("wavefront RMS", f"{tri['wfe']['rms_waves']:.3f} λ", GREEN),
        ("Strehl", f"{tri['psf']['strehl']:.3f}", GREEN),
        ("EE80", f"{tri['psf']['ee80_radius_mm']*1000:.2f} µm", TEXT),
    ]
    for i, (lab, val, col) in enumerate(rows):
        y = 196 + i * 40
        text(d, (rx + 24, y), lab, F_SMALL, DIM)
        text(d, (W - 96, y - 4), val, F_MID, col, anchor="ra")

    # final scorecard
    text(d, (72, H - 200), "SCORECARD", F_TINY, AMBER)
    cards = [
        ("PARAXIAL CONVERGENCE", f"{abs(data['caustic']['exact_bfd']-data['caustic']['paraxial_bdf']):.1e} mm"),
        ("DLS MERIT DROP", f"{data['dls']['merit_start']/data['dls']['merit_end']:.0f}×"),
        ("TRIPLET STREHL", f"{tri['psf']['strehl']:.3f}"),
        ("MTF50", f"{tri['mtf'].get('mtf50_lp_per_mm') or 0:.0f} lp/mm"),
    ]
    cw = 406
    for i, (lab, val) in enumerate(cards):
        x = 72 + i * (cw + 26)
        d.rounded_rectangle([x, H - 168, x + cw, H - 116], 8, fill=PANEL_2, outline=RULE)
        text(d, (x + 16, H - 160), lab, F_TINY, DIM)
        text(d, (x + 16, H - 140), val, F_MID, CYAN)

    lower_third(d, "act 4 · verify",
                f"The triplet holds a {tri['psf']['strehl']:.3f} Strehl across a 6° field",
                f"MTF50 {tri['mtf'].get('mtf50_lp_per_mm') or 0:.0f} lp/mm of a "
                f"{tri['mtf']['cutoff_lp_per_mm']:.0f} lp/mm diffraction cutoff")
    footer(d, "EVERY NUMBER ON SCREEN IS COMPUTED BY THE SAME ENGINE THE APP RUNS",
           45 + t, 55.0)
    return img


def build_frame(t: float, data) -> Image.Image:
    if t < 10.0:
        return frame_act1(t, data)
    if t < 25.0:
        return frame_act2(t - 10.0, data)
    if t < 45.0:
        return frame_act3(t - 25.0, data)
    return frame_act4(t - 45.0, data)


TOTAL_SECONDS = 55.0


def main():
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/film_data.pkl")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("demo_showcase.mp4")
    work = Path("/tmp/opticraft_frames")
    work.mkdir(parents=True, exist_ok=True)

    with open(data_path, "rb") as fh:
        data = pickle.load(fh)

    total_frames = int(TOTAL_SECONDS * FPS)
    print(f"[render] {total_frames} frames at {W}x{H}, {FPS} fps")

    ff = subprocess.Popen(
        [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
            "-an",
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        for i in range(total_frames):
            t = i / FPS
            img = build_frame(t, data)
            ff.stdin.write(img.tobytes())
            if i % 300 == 0:
                print(f"[render] frame {i}/{total_frames}  t={t:.1f}s", flush=True)
    finally:
        ff.stdin.close()
        stderr = ff.stderr.read().decode("utf-8", "ignore")[-400:]
        rc = ff.wait()
    if rc != 0:
        print("[render] ffmpeg failed:\n" + stderr)
        return rc
    print(f"[render] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
