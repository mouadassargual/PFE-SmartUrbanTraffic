"""
Compare waiting times between fixed and MDP SUMO simulations.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


BASE = Path(__file__).resolve().parent


def parse_tripinfo(filename: str) -> dict:
    path = BASE / filename
    if not path.exists():
        raise SystemExit(f"Missing result file: {path}")

    root = ET.parse(path).getroot()
    waiting_times = []
    durations = []

    for trip in root.findall("tripinfo"):
        waiting_times.append(float(trip.get("waitingTime", 0)))
        durations.append(float(trip.get("duration", 0)))

    return {
        "avg_waiting": sum(waiting_times) / len(waiting_times) if waiting_times else 0,
        "max_waiting": max(waiting_times) if waiting_times else 0,
        "avg_duration": sum(durations) / len(durations) if durations else 0,
        "total_vehicles": len(waiting_times),
    }


def main() -> int:
    fixed = parse_tripinfo("tripinfo_fixed.xml")
    mdp = parse_tripinfo("tripinfo_mdp.xml")

    print("=" * 55)
    print("RESULTATS COMPARATIFS - Fixed vs MDP")
    print("=" * 55)
    print(f"{'Metrique':<30} {'Feux fixes':>10} {'MDP':>10}")
    print("-" * 55)
    print(f"{'Vehicules simules':<30} {fixed['total_vehicles']:>10} {mdp['total_vehicles']:>10}")
    print(f"{'Temps attente moyen (s)':<30} {fixed['avg_waiting']:>10.1f} {mdp['avg_waiting']:>10.1f}")
    print(f"{'Temps attente max (s)':<30} {fixed['max_waiting']:>10.1f} {mdp['max_waiting']:>10.1f}")
    print(f"{'Duree trajet moyenne (s)':<30} {fixed['avg_duration']:>10.1f} {mdp['avg_duration']:>10.1f}")

    if fixed["avg_waiting"] > 0:
        reduction = (fixed["avg_waiting"] - mdp["avg_waiting"]) / fixed["avg_waiting"] * 100
        print("-" * 55)
        print(f"{'Reduction temps attente':<30} {reduction:>10.1f}%")
    print("=" * 55)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
