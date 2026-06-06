# extract_mdp_log.py
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "pipeline_results.json"

with RESULTS_PATH.open() as f:
    data = json.load(f)

frames = data["frames"]

# Chercher exemples NS et EW avec densités différentes
examples = []
for frame in frames:
    d = frame["decision"]
    zones = frame["zones"]
    n = zones.get("N", {}).get("count", 0)
    s = zones.get("S", {}).get("count", 0)
    e = zones.get("E", {}).get("count", 0)
    w = zones.get("W", {}).get("count", 0)
    p = frame.get("pedestrians", 0)
    em = frame.get("emergency", False)

    examples.append({
        "frame": frame["frame"],
        "N": n, "S": s, "E": e, "W": w,
        "pedestrians": p,
        "emergency": em,
        "phase": d["phase"],
        "duration": d["duration"],
        "reason": d["reason"],
        "score_NS": round(d["score_NS"], 1),
        "score_EW": round(d["score_EW"], 1),
    })

# Afficher 10 frames espacées
step = max(1, len(examples) // 10)
for i in range(0, len(examples), step):
    e = examples[i]
    print(
        f"F{e['frame']:04d} | "
        f"N={e['N']} S={e['S']} E={e['E']} W={e['W']} "
        f"P={e['pedestrians']} EM={e['emergency']} | "
        f"Phase={e['phase']} {e['duration']}s | "
        f"NS={e['score_NS']} EW={e['score_EW']}"
    )
