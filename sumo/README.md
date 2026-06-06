# SUMO experiment for H5

This branch contains the SUMO testbed used to compare:

1. fixed traffic lights: 30 s North/South, then 30 s East/West;
2. the project MDP controller from `pipeline/decision.py`.

The first scenario is a minimal four-way intersection so the full chain can be
tested immediately. The same controller can then be reused on a real OpenStreetMap
intersection from Agadir, for example Tablorjt, after importing and cleaning the
network in SUMO/NetEdit.

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

If TraCI is not found, run:

```bash
export SUMO_HOME=/opt/homebrew/share/sumo
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```

## Using a real Agadir intersection

The OSM extract `sumo/agadir_talborjt.osm` can be converted and tested with:

```bash
python3 scripts/sumo_build_agadir_talborjt.py
python3 scripts/sumo_run_experiment.py \
  --cfg sumo/agadir_talborjt/agadir.sumocfg \
  --edge-map sumo/agadir_talborjt/edge_map.json \
  --results-dir sumo/agadir_talborjt/results \
  --mode both \
  --tag native
python3 scripts/sumo_run_experiment.py \
  --cfg sumo/agadir_talborjt/agadir.sumocfg \
  --edge-map sumo/agadir_talborjt/edge_map.json \
  --results-dir sumo/agadir_talborjt/results \
  --mode both \
  --mdp-max-duration 30 \
  --tag mdp30
```

The controlled traffic-light node is:

```text
cluster_13875345940_13875345941_13880325192_5153644277_#2more
```

The approach labels `N/S/E/W` are simulation labels mapped in
`sumo/agadir_talborjt/edge_map.json`.

Current Talborjt result with 127 vehicles generated from `pipeline_results.json`:

```text
Fixed 30s cycles : 7.551 s average waiting time
MDP native       : 4.929 s average waiting time
Reduction        : 34.72 %
```

The `--mdp-max-duration 30` sensitivity run gives the same value on this
scenario because the decisions already hold 15 s or 30 s in the observed
traffic states.

For a fresh OSM extraction, the recommended workflow remains:

1. Launch the SUMO OpenStreetMap wizard:

   ```bash
   python3 /opt/homebrew/share/sumo/tools/osmWebWizard.py
   ```

2. Select a small bounding box around the target intersection, for example
   Tablorjt in Agadir.
3. Export the SUMO network.
4. Open it in NetEdit and verify lanes, allowed movements and traffic-light
   phases.
5. Reuse `scripts/sumo_run_experiment.py` with the cleaned `.sumocfg`, or adapt
   the `IN_EDGES` mapping in the script to the real OSM edge IDs.

## Academic interpretation

The simple scenario validates the software link:

```text
pipeline_results.json -> SUMO demand -> fixed lights vs MDP via TraCI
```

The final report should only claim "real geometry of Tablorjt" after the OSM
network has actually been imported, cleaned and used for the measurements.

The simple generated intersection is a calibration test, not final evidence for
Tablorjt. If native MDP durations perform worse than fixed cycles, keep that
result: it shows that phase durations must be calibrated before the final
OSM-based experiment.
