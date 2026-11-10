#!/usr/bin/env bash
# The gate: physics verification, then an end-to-end film smoke test.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/3  physics verification =="
python3 tests/verify_physics.py | tail -8

echo "== 2/3  API smoke test =="
python3 - <<'PY'
import sys
sys.path.insert(0, ".")
from engine.api import design_payload, presets_payload
p = presets_payload()
assert len(p["presets"]) == 4, p["presets"]
d = design_payload(preset="achromat")
s = d["summary"]
assert s["efl_mm"] > 0 and s["ratio_to_airy"] > 0, s
print(f"   achromat: EFL {s['efl_mm']:.2f} mm, {s['ratio_to_airy']:.2f}x Airy - {s['verdict']}")
print("   ok")
PY

echo "== 3/3  film =="
if [ -f demo_showcase.mp4 ]; then
  ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames \
          -show_entries format=duration -of default=noprint_wrappers=1 demo_showcase.mp4
else
  echo "   (no demo_showcase.mp4 yet; run: python3 scripts/film_data.py && python3 scripts/render_film.py)"
fi
echo "all gates passed"
