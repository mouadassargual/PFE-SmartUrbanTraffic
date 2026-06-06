#!/usr/bin/env python3
"""
Prepare the Windows SUMO Talborjt experiment.

Run this on the Windows machine where SUMO is installed. It converts the OSM
extract, creates route flows, creates fixed TLS phases, and writes both SUMO
configs used by the report experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OSM_FILE = ROOT / "sumo" / "agadir_talborjt.osm"
OUT_DIR = ROOT / "sumo" / "windows_talborjt"
EDGE_MAP_FILE = OUT_DIR / "edge_map_windows.json"

TLS_ID = "cluster_13875345940_13875345941_13880325192_5153644277_#2more"
FLOW_RATES = {"N": 960, "S": 120, "E": 360, "W": 720}


def add_sumo_tools_to_path() -> None:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        return
    tools = Path(sumo_home) / "tools"
    if tools.exists():
        sys.path.insert(0, str(tools))


def indent_xml(element: ET.Element) -> None:
    ET.indent(element, space="  ")


def require_sumo_home() -> Path:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise SystemExit("SUMO_HOME is not set. Example: set SUMO_HOME=C:\\Program Files (x86)\\Eclipse\\Sumo")
    path = Path(sumo_home)
    if not path.exists():
        raise SystemExit(f"SUMO_HOME does not exist: {path}")
    return path


def load_edge_map(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing edge map: {path}")
    data = json.loads(path.read_text())
    if data.get("tls_id", "REPLACE_TLS_ID") == "REPLACE_TLS_ID":
        raise SystemExit(f"Please edit tls_id in {path}")
    for direction, values in data["directions"].items():
        if values.get("in", "").startswith("EDGE_") or values.get("out", "").startswith("EDGE_"):
            raise SystemExit(f"Please replace placeholder edge IDs for direction {direction} in {path}")
    return data


def run_netconvert(osm: Path, net: Path, edge_map: dict) -> None:
    sumo_home = require_sumo_home()
    netconvert = shutil.which("netconvert")
    if not netconvert:
        netconvert = str(sumo_home / "bin" / "netconvert.exe")
    typemap = sumo_home / "data" / "typemap" / "osmNetconvert.typ.xml"
    if not Path(netconvert).exists() and shutil.which("netconvert") is None:
        raise SystemExit("netconvert was not found. Add %SUMO_HOME%\\bin to PATH.")
    if not typemap.exists():
        raise SystemExit(f"Missing SUMO typemap: {typemap}")
    if not osm.exists():
        raise SystemExit(f"Missing OSM file: {osm}")

    cmd = [
        netconvert,
        "--osm-files",
        str(osm),
        "--type-files",
        str(typemap),
        "--output-file",
        str(net),
        "--geometry.remove",
        "--roundabouts.guess",
        "--ramps.guess",
        "--junctions.join",
        "--tls.set",
        edge_map["tls_id"],
        "--tls.default-type",
        "static",
    ]
    subprocess.run(cmd, check=True)


def write_routes(path: Path, edge_map: dict) -> None:
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5", length="4.5", maxSpeed="13.9")
    ET.SubElement(root, "vType", id="motorcycle", accel="3.0", decel="5.0", sigma="0.5", length="2.0", maxSpeed="16.7")
    ET.SubElement(root, "vType", id="bus", accel="1.5", decel="3.5", sigma="0.5", length="12.0", maxSpeed="11.1")

    for direction, values in edge_map["directions"].items():
        route_id = values.get("route", f"{direction}_route")
        ET.SubElement(root, "route", id=route_id, edges=f"{values['in']} {values['out']}")
        ET.SubElement(
            root,
            "flow",
            id=f"flow_{direction}",
            type="car",
            route=route_id,
            begin="0",
            end="3600",
            vehsPerHour=str(FLOW_RATES[direction]),
        )

    indent_xml(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def yellow_from_green(state: str) -> str:
    return "".join("y" if char in {"G", "g"} else "r" for char in state)


def state_for_dirs(tl, edge_map: dict, green_dirs: set[str]) -> str:
    connections = tl.getConnections()
    max_index = max(conn[2] for conn in connections)
    state = ["r"] * (max_index + 1)
    edge_to_direction = {
        values["in"]: direction
        for direction, values in edge_map["directions"].items()
    }
    for from_lane, _to_lane, link_index in connections:
        edge_id = from_lane.getEdge().getID()
        direction = edge_to_direction.get(edge_id)
        if direction in green_dirs:
            state[link_index] = "G"
    return "".join(state)


def write_fixed_tls(path: Path, net_path: Path, edge_map: dict) -> None:
    add_sumo_tools_to_path()
    try:
        import sumolib
    except ModuleNotFoundError as exc:
        raise SystemExit("sumolib not found. Set PYTHONPATH=%SUMO_HOME%\\tools;%PYTHONPATH%") from exc

    net = sumolib.net.readNet(str(net_path))
    traffic_lights = {tl.getID(): tl for tl in net.getTrafficLights()}
    tls_id = edge_map["tls_id"]
    if tls_id not in traffic_lights:
        raise SystemExit(f"TLS '{tls_id}' not found in {net_path}. Open sumo-gui and update edge_map_windows.json.")

    tl = traffic_lights[tls_id]
    ns_state = state_for_dirs(tl, edge_map, {"N", "S"})
    ew_state = state_for_dirs(tl, edge_map, {"E", "W"})

    root = ET.Element("additional")
    logic = ET.SubElement(root, "tlLogic", id=tls_id, type="static", programID="fixed", offset="0")
    ET.SubElement(logic, "phase", duration="30", state=ns_state)
    ET.SubElement(logic, "phase", duration="5", state=yellow_from_green(ns_state))
    ET.SubElement(logic, "phase", duration="30", state=ew_state)
    ET.SubElement(logic, "phase", duration="5", state=yellow_from_green(ew_state))
    indent_xml(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_cfg(path: Path, net: Path, routes: Path, fixed_tls: Path | None, fixed: bool) -> None:
    root = ET.Element("configuration")
    input_el = ET.SubElement(root, "input")
    ET.SubElement(input_el, "net-file", value=net.name)
    ET.SubElement(input_el, "route-files", value=routes.name)
    if fixed_tls is not None:
        ET.SubElement(input_el, "additional-files", value=fixed_tls.name)
    time_el = ET.SubElement(root, "time")
    ET.SubElement(time_el, "begin", value="0")
    ET.SubElement(time_el, "end", value="3600")
    ET.SubElement(time_el, "step-length", value="1")
    processing_el = ET.SubElement(root, "processing")
    ET.SubElement(processing_el, "ignore-route-errors", value="true")
    if fixed:
        output_el = ET.SubElement(root, "output")
        ET.SubElement(output_el, "tripinfo-output", value="tripinfo_fixed.xml")
        ET.SubElement(output_el, "summary-output", value="summary_fixed.xml")
    indent_xml(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Windows SUMO files for Talborjt")
    parser.add_argument("--osm", type=Path, default=OSM_FILE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--edge-map", type=Path, default=EDGE_MAP_FILE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    edge_map = load_edge_map(args.edge_map)

    net = args.out_dir / "agadir.net.xml"
    routes = args.out_dir / "agadir.rou.xml"
    fixed_tls = args.out_dir / "fixed_tls.add.xml"

    run_netconvert(args.osm, net, edge_map)
    write_routes(routes, edge_map)
    write_fixed_tls(fixed_tls, net, edge_map)
    write_cfg(args.out_dir / "agadir_fixed.sumocfg", net, routes, fixed_tls, fixed=True)
    write_cfg(args.out_dir / "agadir_mdp.sumocfg", net, routes, None, fixed=False)

    print("Windows SUMO Talborjt files prepared:")
    print(f"  Network : {net}")
    print(f"  Routes  : {routes}")
    print(f"  Fixed   : {args.out_dir / 'agadir_fixed.sumocfg'}")
    print(f"  MDP     : {args.out_dir / 'agadir_mdp.sumocfg'}")
    print(f"  TLS     : {edge_map['tls_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
