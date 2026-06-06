# SUMO experiment for H5

This branch contains the SUMO testbed used to compare:

1. fixed traffic lights: 30 s North/South, then 30 s East/West;
2. the project MDP controller from `pipeline/decision.py`.

The first scenario is a minimal four-way intersection used only as a software
testbed. The real Talborjt validation must be executed on the Windows PC with
SUMO installed.

## Immediate reproducible test

From the repository root:

```bash
python3 scripts/sumo_prepare_demand.py
python3 scripts/sumo_build_simple_intersection.py
python3 scripts/sumo_run_experiment.py --mode both --tag native
```

For a sensitivity test that caps MDP phases to the same 30 s maximum used by
the fixed baseline:

```bash
python3 scripts/sumo_run_experiment.py --mode both --mdp-max-duration 30 --tag mdp30
```

Outputs are written to:

```text
sumo/simple_intersection/demand/pipeline_demand.json
sumo/simple_intersection/routes.rou.xml
sumo/simple_intersection/results/sumo_comparison_native.json
sumo/simple_intersection/results/sumo_comparison_mdp30.json
```

This simple scenario is not the final report evidence.

## Using a real Agadir intersection

Use the Windows workflow in:

```text
sumo/windows_talborjt/README_WINDOWS.md
```

Main Windows commands:

```bat
set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
set PATH=%SUMO_HOME%\bin;%PATH%
set PYTHONPATH=%SUMO_HOME%\tools;%PYTHONPATH%
python scripts\sumo_prepare_windows_talborjt.py
cd sumo\windows_talborjt
sumo -c agadir_fixed.sumocfg
python sumo_mdp_control.py
python analyze_results.py
```

The controlled traffic-light node prepared for Talborjt is:

```text
cluster_13875345940_13875345941_13880325192_5153644277_#2more
```

The approach labels `N/S/E/W` are simulation labels mapped in:

```text
sumo/windows_talborjt/edge_map_windows.json
```

The final report should use the Windows-produced `tripinfo_fixed.xml`,
`tripinfo_mdp.xml`, terminal output, and screenshots. Do not use local draft
results as final evidence.

## Academic interpretation

The simple scenario validates the software link:

```text
pipeline_results.json -> SUMO demand -> fixed lights vs MDP via TraCI
```

The final report should only claim "real geometry of Talborjt" after the OSM
network has been imported and simulated on the Windows PC.

The simple generated intersection is a calibration test, not final evidence for
Tablorjt. If native MDP durations perform worse than fixed cycles, keep that
result: it shows that phase durations must be calibrated before the final
OSM-based experiment.
