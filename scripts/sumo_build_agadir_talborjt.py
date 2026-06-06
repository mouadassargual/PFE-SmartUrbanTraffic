#!/usr/bin/env python3
"""
Build the Agadir Talborjt SUMO scenario from the OSM extract.

The OSM file contains real geometry. We force the main multi-approach node to a
SUMO traffic light, then generate straight-through demand from the measured
pipeline densities.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OSM_FILE = ROOT / "sumo" / "agadir_talborjt.osm"
SCENARIO_DIR = ROOT / "sumo" / "agadir_talborjt"
DEMAND_JSON = ROOT / "sumo" / "simple_intersection" / "demand" / "pipeline_demand.json"
TYPEMAP = Path("/opt/homebrew/share/sumo/data/typemap/osmNetconvert.typ.xml")

TLS_ID = "cluster_13875345940_13875345941_13880325192_5153644277_#2more"
DIRECTIONS = ("N", "S", "E", "W")
EDGE_MAP = {
    "tls_id": TLS_ID,
    "note": "Approach labels for the Talborjt OSM scenario. They are simulation labels mapped to the four main incoming roads around the controlled junction.",
    "directions": {
        "N": {"in": "-591072157#1", "out": "666212794#1", "route": "N_to_S"},
        "S": {"in": "-28805336#0", "out": "591072157#1", "route": "S_to_N"},
        "E": {"in": "180596058#2", "out": "-440316493#2", "route": "E_to_W"},
        "W": {"in": "440316493#2", "out": "-180596058#2", "route": "W_to_E"},
    },
}


def indent_xml(element: ET.Element) -> None:
    ET.indent(element, space="  ")


def run_netconvert(osm: Path, net: Path) -> None:
    netconvert = shutil.which("netconvert")
    if not netconvert:
        raise SystemExit("netconvert not found. Install SUMO first.")
    if not osm.exists():
        raise SystemExit(f"Missing OSM file: {osm}")
    if not TYPEMAP.exists():
        raise SystemExit(f"Missing SUMO OSM typemap: {TYPEMAP}")

    cmd = [
        netconvert,
        "--osm-files",
        str(osm),
        "--type-files",
        str(TYPEMAP),
        "--output-file",
        str(net),
        "--geometry.remove",
        "--roundabouts.guess",
        "--ramps.guess",
        "--junctions.join",
        "--tls.set",
        TLS_ID,
        "--tls.default-type",
        "static",
    ]
    subprocess.run(cmd, check=True)


def add_vehicle(root: ET.Element, veh_id: str, route_id: str, depart: float) -> None:
    ET.SubElement(
        root,
        "vehicle",
        id=veh_id,
        type="car",
        route=route_id,
        depart=f"{depart:.2f}",
        departLane="best",
        departSpeed="max",
    )


def write_routes(demand_json: Path, routes_path: Path, scale: float, count_field: str) -> int:
    if not demand_json.exists():
        raise SystemExit(f"Missing demand file: {demand_json}")
    demand = json.loads(demand_json.read_text())
    windows = demand.get("windows", [])

    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5", length="4.5", maxSpeed="13.9")
    ET.SubElement(root, "vType", id="motorcycle", accel="3.0", decel="5.0", sigma="0.5", length="2.0", maxSpeed="16.7")
    ET.SubElement(root, "vType", id="bus", accel="1.5", decel="3.5", sigma="0.5", length="12.0", maxSpeed="11.1")

    for direction, values in EDGE_MAP["directions"].items():
        ET.SubElement(root, "route", id=values["route"], edges=f"{values['in']} {values['out']}")

    vehicles = []
    for window in windows:
        start = float(window.get("start_second", 0.0))
        end = float(window.get("end_second", start + 30.0))
        span = max(1.0, end - start)
        for direction in DIRECTIONS:
            values = window["directions"][direction]
            raw_count = float(values[count_field])
            count = max(0, int(round(raw_count * scale)))
            route_id = EDGE_MAP["directions"][direction]["route"]
            for index in range(count):
                depart = start + (index + 0.5) * span / max(count, 1)
                veh_id = f"{direction}_{window['window']}_{index}"
                vehicles.append((depart, veh_id, route_id))

    vehicles.sort(key=lambda item: item[0])
    for depart, veh_id, route_id in vehicles:
        add_vehicle(root, veh_id, route_id, depart)

    indent_xml(root)
    ET.ElementTree(root).write(routes_path, encoding="utf-8", xml_declaration=True)
    return len(vehicles)


def write_cfg(path: Path, net: Path, routes: Path, end: int) -> None:
    root = ET.Element("configuration")
    input_el = ET.SubElement(root, "input")
    ET.SubElement(input_el, "net-file", value=net.name)
    ET.SubElement(input_el, "route-files", value=routes.name)
    time_el = ET.SubElement(root, "time")
    ET.SubElement(time_el, "begin", value="0")
    ET.SubElement(time_el, "end", value=str(end))
    ET.SubElement(time_el, "step-length", value="1")
    processing_el = ET.SubElement(root, "processing")
    ET.SubElement(processing_el, "ignore-route-errors", value="true")
    indent_xml(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Agadir Talborjt SUMO scenario")
    parser.add_argument("--osm", type=Path, default=OSM_FILE)
    parser.add_argument("--scenario-dir", type=Path, default=SCENARIO_DIR)
    parser.add_argument("--demand", type=Path, default=DEMAND_JSON)
    parser.add_argument("--scale", type=float, default=0.6)
    parser.add_argument("--count-field", choices=["mean_count", "max_count"], default="max_count")
    parser.add_argument("--end", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.scenario_dir.mkdir(parents=True, exist_ok=True)

    net = args.scenario_dir / "agadir_controlled.net.xml"
    routes = args.scenario_dir / "agadir.rou.xml"
    cfg = args.scenario_dir / "agadir.sumocfg"
    edge_map = args.scenario_dir / "edge_map.json"

    run_netconvert(args.osm, net)
    vehicle_count = write_routes(args.demand, routes, args.scale, args.count_field)
    write_cfg(cfg, net, routes, args.end)
    edge_map.write_text(json.dumps(EDGE_MAP, indent=2))

    print(f"Network  : {net}")
    print(f"Routes   : {routes}")
    print(f"Config   : {cfg}")
    print(f"Edge map : {edge_map}")
    print(f"Vehicles : {vehicle_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
