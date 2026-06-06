"""
Simulation B - MDP adaptive control via TraCI.

Run from this folder on Windows after generating agadir_mdp.sumocfg.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
SUMO_CFG = BASE / "agadir_mdp.sumocfg"
EDGE_MAP_FILE = BASE / "edge_map_windows.json"


def add_paths() -> None:
    sys.path.insert(0, str(ROOT))
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        tools = Path(sumo_home) / "tools"
        if tools.exists():
            sys.path.insert(0, str(tools))


add_paths()

try:
    import traci
except ModuleNotFoundError as exc:
    raise SystemExit("TraCI not found. Set PYTHONPATH=%SUMO_HOME%\\tools;%PYTHONPATH%") from exc

from pipeline.decision import MDPDecision


def load_edge_map() -> dict:
    return json.loads(EDGE_MAP_FILE.read_text())


def edge_to_direction(edge_map: dict) -> dict[str, str]:
    return {
        values["in"]: direction
        for direction, values in edge_map["directions"].items()
    }


def get_zone_counts(edge_map: dict) -> dict[str, int]:
    counts = {"N": 0, "S": 0, "E": 0, "W": 0}
    for direction, values in edge_map["directions"].items():
        counts[direction] = int(traci.edge.getLastStepVehicleNumber(values["in"]))
    return counts


def counts_to_state(counts: dict[str, int]) -> dict:
    return {
        "N": {"count": counts["N"], "score": float(counts["N"])},
        "S": {"count": counts["S"], "score": float(counts["S"])},
        "E": {"count": counts["E"], "score": float(counts["E"])},
        "W": {"count": counts["W"], "score": float(counts["W"])},
        "pedestrians": 0,
        "emergency": False,
        "emergency_zone": None,
    }


def green_dirs_for_phase(phase: str) -> list[str]:
    if phase == "NS":
        return ["N", "S"]
    if phase == "EW":
        return ["E", "W"]
    if phase == "EMERGENCY":
        return ["N", "S", "E", "W"]
    return []


def tls_state(tls_id: str, edge_map: dict, green_dirs: list[str]) -> str:
    edge_dirs = edge_to_direction(edge_map)
    allowed = set(green_dirs)
    chars = []
    for link_group in traci.trafficlight.getControlledLinks(tls_id):
        if not link_group:
            chars.append("r")
            continue
        incoming_lane = link_group[0][0]
        edge_id = incoming_lane.rsplit("_", 1)[0]
        direction = edge_dirs.get(edge_id)
        chars.append("G" if direction in allowed else "r")
    return "".join(chars)


def apply_phase(tls_id: str, edge_map: dict, phase: str) -> None:
    traci.trafficlight.setRedYellowGreenState(
        tls_id,
        tls_state(tls_id, edge_map, green_dirs_for_phase(phase)),
    )


def write_decisions(rows: list[dict]) -> None:
    if not rows:
        return
    out_path = BASE / "mdp_decisions.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_mdp_simulation() -> None:
    os.chdir(BASE)
    edge_map = load_edge_map()
    tls_id = edge_map["tls_id"]

    sumo_cmd = [
        "sumo",
        "-c",
        str(SUMO_CFG),
        "--tripinfo-output",
        "tripinfo_mdp.xml",
        "--summary-output",
        "summary_mdp.xml",
        "--no-step-log",
        "true",
    ]
    traci.start(sumo_cmd)

    mdp = MDPDecision()
    step = 0
    phase_timer = 0
    phase_duration = 30
    current_phase = "NS"
    rows = []

    apply_phase(tls_id, edge_map, current_phase)
    print("MDP Simulation started...")

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1
        phase_timer += 1

        if phase_timer >= phase_duration:
            counts = get_zone_counts(edge_map)
            state = counts_to_state(counts)
            decision = mdp.decide(state)

            current_phase = decision["phase"]
            phase_duration = int(decision["duration"])
            if current_phase == "ALL_RED":
                phase_duration = min(phase_duration, 5)
            phase_timer = 0

            apply_phase(tls_id, edge_map, current_phase)
            rows.append(
                {
                    "step": step,
                    "phase": current_phase,
                    "duration": phase_duration,
                    "score_NS": round(decision["score_NS"], 3),
                    "score_EW": round(decision["score_EW"], 3),
                    "N": counts["N"],
                    "S": counts["S"],
                    "E": counts["E"],
                    "W": counts["W"],
                }
            )

            print(
                f"Step {step:04d} | Phase={current_phase:<7} "
                f"Duration={phase_duration:>2}s | "
                f"N={counts['N']} S={counts['S']} E={counts['E']} W={counts['W']}"
            )

    traci.close()
    write_decisions(rows)
    print("MDP Simulation completed.")


if __name__ == "__main__":
    run_mdp_simulation()
