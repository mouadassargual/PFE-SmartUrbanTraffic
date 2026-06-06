#!/usr/bin/env python3
"""
Run fixed-time vs MDP traffic-light experiments in SUMO.

The MDP controller reuses pipeline.decision.MDPDecision and sends green phases
to SUMO through TraCI. Metrics are read from SUMO tripinfo output.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "sumo" / "simple_intersection"
DIRECTIONS = ("N", "S", "E", "W")
DEFAULT_EDGE_MAP = {
    "tls_id": None,
    "directions": {
        "N": {"in": "N_in"},
        "S": {"in": "S_in"},
        "E": {"in": "E_in"},
        "W": {"in": "W_in"},
    },
}


def add_sumo_tools_to_path() -> None:
    candidates = []
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidates.append(Path(sumo_home) / "tools")
    candidates.extend(
        [
            Path("/opt/homebrew/share/sumo/tools"),
            Path("/usr/local/share/sumo/tools"),
            Path("/usr/share/sumo/tools"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            return


add_sumo_tools_to_path()
sys.path.insert(0, str(ROOT))

try:
    import traci
except ModuleNotFoundError as exc:
    raise SystemExit("TraCI not found. Set SUMO_HOME or install SUMO with Python tools.") from exc

from pipeline.decision import MDPDecision


def sumo_binary(gui: bool) -> str:
    binary = shutil.which("sumo-gui" if gui else "sumo")
    if not binary:
        raise SystemExit("SUMO binary not found. Install SUMO first.")
    return binary


def load_edge_map(path: Path | None) -> dict:
    if path is None:
        return DEFAULT_EDGE_MAP
    if not path.exists():
        raise SystemExit(f"Missing edge map: {path}")
    data = json.loads(path.read_text())
    if "directions" not in data:
        data = {"tls_id": data.get("tls_id"), "directions": data}
    for direction in DIRECTIONS:
        if direction not in data["directions"] or "in" not in data["directions"][direction]:
            raise SystemExit(f"Edge map missing input edge for direction {direction}: {path}")
    return data


def controlled_direction(link_group, edge_map: dict) -> str | None:
    if not link_group:
        return None
    incoming_lane = link_group[0][0]
    edge_id = incoming_lane.rsplit("_", 1)[0]
    for direction, values in edge_map["directions"].items():
        if edge_id == values["in"]:
            return direction
    return None


def tls_state(tls_id: str, green_dirs: list[str], edge_map: dict) -> str:
    allowed = set(green_dirs)
    chars = []
    for link_group in traci.trafficlight.getControlledLinks(tls_id):
        direction = controlled_direction(link_group, edge_map)
        chars.append("G" if direction in allowed else "r")
    return "".join(chars)


def apply_phase(tls_id: str, green_dirs: list[str], edge_map: dict) -> None:
    traci.trafficlight.setRedYellowGreenState(tls_id, tls_state(tls_id, green_dirs, edge_map))


def current_sumo_state(edge_map: dict) -> dict:
    state = {}
    for direction, values in edge_map["directions"].items():
        edge_id = values["in"]
        count = int(traci.edge.getLastStepVehicleNumber(edge_id))
        halted = int(traci.edge.getLastStepHaltingNumber(edge_id))
        score = float(count + halted)
        state[direction] = {"count": count, "score": score}
    state["pedestrians"] = 0
    state["emergency"] = False
    state["emergency_zone"] = None
    return state


def decision_green_dirs(decision: dict) -> list[str]:
    phase = decision.get("phase")
    if phase == "NS":
        return ["N", "S"]
    if phase == "EW":
        return ["E", "W"]
    if phase == "EMERGENCY":
        return ["N", "S", "E", "W"]
    return []


def mdp_hold_duration(decision: dict, args: argparse.Namespace) -> int:
    hold = max(1, int(decision["duration"]))
    if decision.get("phase") == "ALL_RED":
        hold = min(hold, max(1, int(args.idle_check_interval)))
    if args.mdp_max_duration is not None:
        hold = min(hold, max(1, int(args.mdp_max_duration)))
    return hold


def parse_tripinfo(path: Path) -> dict:
    if not path.exists():
        return {"vehicles": 0, "avg_waiting_time": None, "avg_time_loss": None, "avg_duration": None}
    root = ET.parse(path).getroot()
    trips = root.findall("tripinfo")
    if not trips:
        return {"vehicles": 0, "avg_waiting_time": None, "avg_time_loss": None, "avg_duration": None}

    waiting = [float(trip.get("waitingTime", 0.0)) for trip in trips]
    time_loss = [float(trip.get("timeLoss", 0.0)) for trip in trips]
    duration = [float(trip.get("duration", 0.0)) for trip in trips]
    return {
        "vehicles": len(trips),
        "avg_waiting_time": round(sum(waiting) / len(waiting), 3),
        "avg_time_loss": round(sum(time_loss) / len(time_loss), 3),
        "avg_duration": round(sum(duration) / len(duration), 3),
    }


def run_mode(args: argparse.Namespace, mode: str) -> dict:
    cfg = args.cfg or (args.scenario_dir / "simple.sumocfg")
    if not cfg.exists():
        raise SystemExit(f"Missing SUMO config: {cfg}")
    edge_map = load_edge_map(args.edge_map)

    results_dir = args.results_dir or (cfg.parent / "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    tripinfo = results_dir / f"{mode}{suffix}_tripinfo.xml"
    summary = results_dir / f"{mode}{suffix}_summary.xml"
    decisions_csv = results_dir / f"{mode}{suffix}_decisions.csv"

    cmd = [
        sumo_binary(args.gui),
        "-c",
        str(cfg),
        "--tripinfo-output",
        str(tripinfo),
        "--summary-output",
        str(summary),
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
    ]

    traci.start(cmd, port=args.traci_port)
    tls_ids = traci.trafficlight.getIDList()
    if not tls_ids:
        traci.close()
        raise SystemExit("No traffic light found in SUMO network.")
    tls_id = args.tls_id or edge_map.get("tls_id") or tls_ids[0]
    if tls_id not in tls_ids:
        traci.close()
        raise SystemExit(f"Traffic light '{tls_id}' not found. Available TLS: {list(tls_ids)}")
    mdp = MDPDecision() if mode == "mdp" else None
    decisions = []

    try:
        step = 0
        next_mdp_decision_step = 0
        current_green = ["N", "S"]
        apply_phase(tls_id, current_green, edge_map)

        while step < args.max_steps and traci.simulation.getMinExpectedNumber() > 0:
            if mode == "fixed":
                current_green = ["N", "S"] if (step // args.fixed_cycle) % 2 == 0 else ["E", "W"]
                apply_phase(tls_id, current_green, edge_map)
            elif mode == "mdp" and step >= next_mdp_decision_step:
                state = current_sumo_state(edge_map)
                decision = mdp.decide(state)
                current_green = decision_green_dirs(decision)
                apply_phase(tls_id, current_green, edge_map)
                hold = mdp_hold_duration(decision, args)
                next_mdp_decision_step = step + hold
                decisions.append(
                    {
                        "step": step,
                        "phase": decision["phase"],
                        "duration": decision["duration"],
                        "hold": hold,
                        "score_NS": round(decision["score_NS"], 3),
                        "score_EW": round(decision["score_EW"], 3),
                        "N": state["N"]["count"],
                        "S": state["S"]["count"],
                        "E": state["E"]["count"],
                        "W": state["W"]["count"],
                    }
                )
            traci.simulationStep()
            step += 1
    finally:
        traci.close()

    if decisions:
        with decisions_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(decisions[0].keys()))
            writer.writeheader()
            writer.writerows(decisions)

    metrics = parse_tripinfo(tripinfo)
    metrics.update({"mode": mode, "steps": step, "tripinfo": str(tripinfo), "summary": str(summary)})
    if decisions:
        metrics["decisions_csv"] = str(decisions_csv)
    return metrics


def comparison(fixed: dict, mdp: dict) -> dict:
    fixed_wait = fixed.get("avg_waiting_time")
    mdp_wait = mdp.get("avg_waiting_time")
    gain = None
    if fixed_wait is not None and fixed_wait > 0 and mdp_wait is not None:
        gain = round((fixed_wait - mdp_wait) / fixed_wait * 100.0, 2)
    return {
        "fixed_avg_waiting_time": fixed_wait,
        "mdp_avg_waiting_time": mdp_wait,
        "waiting_time_reduction_pct": gain,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SUMO fixed vs MDP experiment")
    parser.add_argument("--scenario-dir", type=Path, default=SCENARIO_DIR)
    parser.add_argument("--cfg", type=Path, default=None, help="SUMO .sumocfg; default simple scenario")
    parser.add_argument("--edge-map", type=Path, default=None, help="JSON mapping N/S/E/W to SUMO input edges")
    parser.add_argument("--tls-id", type=str, default=None, help="Traffic-light id to control")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["fixed", "mdp", "both"], default="both")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--fixed-cycle", type=int, default=30)
    parser.add_argument("--decision-interval", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--traci-port", type=int, default=8813)
    parser.add_argument("--idle-check-interval", type=int, default=5)
    parser.add_argument("--mdp-max-duration", type=int, default=None)
    parser.add_argument("--tag", type=str, default="", help="Optional suffix for result JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    modes = ["fixed", "mdp"] if args.mode == "both" else [args.mode]
    results = {}
    for mode in modes:
        print(f"Running SUMO mode: {mode}")
        results[mode] = run_mode(args, mode)

    payload = {
        "settings": {
            "fixed_cycle": args.fixed_cycle,
            "idle_check_interval": args.idle_check_interval,
            "mdp_max_duration": args.mdp_max_duration,
            "max_steps": args.max_steps,
        },
        "results": results,
    }
    if "fixed" in results and "mdp" in results:
        payload["comparison"] = comparison(results["fixed"], results["mdp"])

    suffix = f"_{args.tag}" if args.tag else ""
    cfg = args.cfg or (args.scenario_dir / "simple.sumocfg")
    summary_dir = args.results_dir or (cfg.parent / "results")
    summary_dir.mkdir(parents=True, exist_ok=True)
    out_path = summary_dir / f"sumo_comparison{suffix}.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print("\nSUMO comparison")
    for mode, metrics in results.items():
        print(
            f"- {mode:5s}: vehicles={metrics['vehicles']} "
            f"avg_wait={metrics['avg_waiting_time']}s "
            f"avg_loss={metrics['avg_time_loss']}s"
        )
    if "comparison" in payload:
        gain = payload["comparison"]["waiting_time_reduction_pct"]
        print(f"- reduction attente MDP vs fixe: {gain}%")
    print(f"Results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
