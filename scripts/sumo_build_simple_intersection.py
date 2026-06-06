#!/usr/bin/env python3
"""
Build a minimal four-way SUMO scenario from prepared demand.

This is the fast testbed for the MDP/SUMO link. It is intentionally simple:
one traffic-light junction, four approaches, straight movements only.
The same Python controller can later be run on an OSM-imported Tablorjt net.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "sumo" / "simple_intersection"
DEMAND_JSON = SCENARIO_DIR / "demand" / "pipeline_demand.json"
DIRECTIONS = ("N", "S", "E", "W")


def indent_xml(element: ET.Element) -> None:
    ET.indent(element, space="  ")


def write_nodes(path: Path) -> None:
    root = ET.Element("nodes")
    nodes = [
        ("center", "traffic_light", "0", "0"),
        ("N", "priority", "0", "250"),
        ("S", "priority", "0", "-250"),
        ("E", "priority", "250", "0"),
        ("W", "priority", "-250", "0"),
    ]
    for node_id, node_type, x, y in nodes:
        ET.SubElement(root, "node", id=node_id, type=node_type, x=x, y=y)
    indent_xml(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_edges(path: Path) -> None:
    root = ET.Element("edges")
    edges = [
        ("N_in", "N", "center"),
        ("N_out", "center", "N"),
        ("S_in", "S", "center"),
        ("S_out", "center", "S"),
        ("E_in", "E", "center"),
        ("E_out", "center", "E"),
        ("W_in", "W", "center"),
        ("W_out", "center", "W"),
    ]
    for edge_id, from_node, to_node in edges:
        ET.SubElement(
            root,
            "edge",
            id=edge_id,
            from_=from_node,
            to=to_node,
            numLanes="1",
            speed="13.9",
            priority="1",
        )
    # xml.etree writes from_ literally. Fix it after serialization.
    indent_xml(root)
    text = ET.tostring(root, encoding="unicode")
    text = text.replace("from_=", "from=")
    path.write_text('<?xml version="1.0" encoding="utf-8"?>\n' + text)


def write_connections(path: Path) -> None:
    root = ET.Element("connections")
    connections = [
        ("N_in", "S_out"),
        ("S_in", "N_out"),
        ("E_in", "W_out"),
        ("W_in", "E_out"),
    ]
    for from_edge, to_edge in connections:
        ET.SubElement(root, "connection", from_=from_edge, to=to_edge)
    indent_xml(root)
    text = ET.tostring(root, encoding="unicode")
    text = text.replace("from_=", "from=")
    path.write_text('<?xml version="1.0" encoding="utf-8"?>\n' + text)


def run_netconvert(nodes: Path, edges: Path, connections: Path, net: Path) -> None:
    netconvert = shutil.which("netconvert")
    if not netconvert:
        raise SystemExit("netconvert not found. Install SUMO first.")
    cmd = [
        netconvert,
        "--node-files",
        str(nodes),
        "--edge-files",
        str(edges),
        "--connection-files",
        str(connections),
        "--output-file",
        str(net),
        "--tls.guess",
        "true",
        "--no-turnarounds",
        "true",
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


def write_routes(demand_json: Path, route_path: Path, scale: float, count_field: str) -> int:
    demand = json.loads(demand_json.read_text())
    windows = demand.get("windows", [])
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5", length="5.0", maxSpeed="13.9")
    ET.SubElement(root, "route", id="N_straight", edges="N_in S_out")
    ET.SubElement(root, "route", id="S_straight", edges="S_in N_out")
    ET.SubElement(root, "route", id="E_straight", edges="E_in W_out")
    ET.SubElement(root, "route", id="W_straight", edges="W_in E_out")

    vehicles = []
    for window in windows:
        start = float(window.get("start_second", 0.0))
        end = float(window.get("end_second", start + 1.0))
        span = max(1.0, end - start)
        for direction in DIRECTIONS:
            values = window["directions"][direction]
            raw_count = float(values[count_field])
            count = max(0, int(round(raw_count * scale)))
            for index in range(count):
                depart = start + (index + 0.5) * span / max(count, 1)
                veh_id = f"{direction}_{window['window']}_{index}"
                vehicles.append((depart, veh_id, f"{direction}_straight"))

    vehicles.sort(key=lambda item: item[0])
    for depart, veh_id, route_id in vehicles:
        add_vehicle(root, veh_id, route_id, depart)

    indent_xml(root)
    ET.ElementTree(root).write(route_path, encoding="utf-8", xml_declaration=True)
    return len(vehicles)


def write_cfg(path: Path, net: Path, routes: Path, end: int) -> None:
    root = ET.Element("configuration")
    input_el = ET.SubElement(root, "input")
    ET.SubElement(input_el, "net-file", value=net.name)
    ET.SubElement(input_el, "route-files", value=routes.name)
    time_el = ET.SubElement(root, "time")
    ET.SubElement(time_el, "begin", value="0")
    ET.SubElement(time_el, "end", value=str(end))
    indent_xml(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build simple SUMO intersection scenario")
    parser.add_argument("--scenario-dir", type=Path, default=SCENARIO_DIR)
    parser.add_argument("--demand", type=Path, default=DEMAND_JSON)
    parser.add_argument("--scale", type=float, default=0.6, help="Scale pipeline densities into SUMO arrivals")
    parser.add_argument("--count-field", choices=["mean_count", "max_count"], default="max_count")
    parser.add_argument("--end", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.demand.exists():
        raise SystemExit(f"Missing demand file: {args.demand}")
    args.scenario_dir.mkdir(parents=True, exist_ok=True)

    nodes = args.scenario_dir / "simple.nod.xml"
    edges = args.scenario_dir / "simple.edg.xml"
    connections = args.scenario_dir / "simple.con.xml"
    net = args.scenario_dir / "simple.net.xml"
    routes = args.scenario_dir / "routes.rou.xml"
    cfg = args.scenario_dir / "simple.sumocfg"

    write_nodes(nodes)
    write_edges(edges)
    write_connections(connections)
    run_netconvert(nodes, edges, connections, net)
    vehicle_count = write_routes(args.demand, routes, args.scale, args.count_field)
    write_cfg(cfg, net, routes, args.end)

    print(f"Network  : {net}")
    print(f"Routes   : {routes}")
    print(f"Config   : {cfg}")
    print(f"Vehicles : {vehicle_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
