"""Opticraft workbench — HTTP API over the engine.

Run:  python3 app.py            (then open http://127.0.0.1:8770)

Serves two things:

* ``/``            the interactive bench (static/index.html)
* ``/api/...``     JSON computed by ``engine`` — no numbers are hard-coded in the UI

FastAPI is used when it is installed (it is in the Hermes venv); otherwise the
same routes are served by a stdlib ``http.server`` thread, so the workbench runs
on a bare Python with only numpy and Pillow present. The engine is the single
source of truth either way.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from engine.api import design_payload, optimize_payload, presets_payload, ray_fan_payload  # noqa: E402
from engine.glass import WAVELENGTHS  # noqa: E402

STATIC = HERE / "static"
PORT = 8770
HOST = "127.0.0.1"


def route(path: str, query: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """One dispatcher shared by both servers, so they cannot drift."""
    try:
        if path in ("/api/presets", "/api/presets/"):
            return 200, presets_payload()

        if path in ("/api/design", "/api/design/"):
            preset = query.get("preset")
            surfaces = query.get("surfaces")
            if isinstance(surfaces, str):
                surfaces = json.loads(surfaces)
            payload = design_payload(
                preset=preset,
                surfaces=surfaces,
                aperture_mm=float(query["aperture_mm"]) if query.get("aperture_mm") else None,
                field_deg=float(query.get("field_deg") or 0.0),
            )
            return 200, payload

        if path in ("/api/optimize", "/api/optimize/"):
            preset = query.get("preset")
            surfaces = query.get("surfaces")
            if isinstance(surfaces, str):
                surfaces = json.loads(surfaces)
            payload = optimize_payload(
                preset=preset,
                surfaces=surfaces,
                aperture_mm=float(query["aperture_mm"]) if query.get("aperture_mm") else None,
                efl_target_mm=float(query["efl_target_mm"]) if query.get("efl_target_mm") else None,
                field_deg=float(query.get("field_deg") or 0.0),
                iterations=int(query.get("iterations") or 22),
            )
            return 200, payload

        if path in ("/api/fan", "/api/fan/"):
            preset = query.get("preset") or "singlet"
            from engine.presets import get as get_preset
            p = get_preset(preset)
            fans = {}
            for label, w in (("F", WAVELENGTHS["F"]), ("d", WAVELENGTHS["d"]), ("C", WAVELENGTHS["C"])):
                fans[label] = ray_fan_payload(
                    surfaces=list(p.design.surfaces),
                    aperture_mm=float(query.get("aperture_mm") or p.design.aperture_mm),
                    field_deg=float(query.get("field_deg") or 0.0),
                    wavelength_nm=w, samples=int(query.get("samples") or 41))
            return 200, {"ok": True, "fans": fans}

        if path in ("/api/health", "/api/health/"):
            return 200, {"ok": True, "engine": "opticraft", "wavelengths": WAVELENGTHS}

        return 404, {"ok": False, "error": f"no route {path}"}
    except KeyError as exc:
        return 404, {"ok": False, "error": f"missing parameter: {exc}"}
    except Exception as exc:  # noqa: BLE001 - the API must answer, not crash
        traceback.print_exc()
        return 500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def run_fastapi() -> None:
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Opticraft workbench")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/{name}")
    def api(name: str, request: Request):
        status, payload = route(f"/api/{name}", dict(request.query_params))
        return JSONResponse(payload, status_code=status)

    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    import uvicorn
    print(f"[opticraft] http://{HOST}:{PORT}  (FastAPI)")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


def run_stdlib() -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                body = (STATIC / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path.startswith("/static/"):
                f = STATIC / parsed.path[len("/static/"):]
                if f.is_file():
                    body = f.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            status, payload = route(parsed.path, query)
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    print(f"[opticraft] http://{HOST}:{PORT}  (stdlib server)")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


def main() -> None:
    try:
        run_fastapi()
    except ImportError:
        run_stdlib()


if __name__ == "__main__":
    main()
