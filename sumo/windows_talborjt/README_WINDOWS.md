# Talborjt SUMO experiment on Windows

This folder is the Windows entry point for the final SUMO validation.
The final screenshots and `tripinfo_*.xml` files must be produced on the
Windows PC where SUMO is installed.

## 1. Open a Windows terminal

From the repository root:

```bat
set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
set PATH=%SUMO_HOME%\bin;%PATH%
set PYTHONPATH=%SUMO_HOME%\tools;%PYTHONPATH%
python -c "import traci, sumolib; print('SUMO Python OK')"
```

If SUMO is installed somewhere else, adjust `SUMO_HOME`.

## 2. Prepare SUMO files

```bat
python scripts\sumo_prepare_windows_talborjt.py
```

This creates:

```text
sumo\windows_talborjt\agadir.net.xml
sumo\windows_talborjt\agadir.rou.xml
sumo\windows_talborjt\fixed_tls.add.xml
sumo\windows_talborjt\agadir_fixed.sumocfg
sumo\windows_talborjt\agadir_mdp.sumocfg
```

The preparation script forces the central Talborjt junction as a traffic light
because the raw OSM extract may not contain a complete vehicle TLS program.

## 3. Visual check in SUMO GUI

```bat
sumo-gui -c sumo\windows_talborjt\agadir_fixed.sumocfg
```

Check that the controlled intersection is the Talborjt junction and that the
approaches in `edge_map_windows.json` match the visible N/S/E/W roads. If the
edge IDs differ on your SUMO version, edit `edge_map_windows.json`, then rerun:

```bat
python scripts\sumo_prepare_windows_talborjt.py
```

## 4. Run Simulation A: fixed lights

From the folder `sumo\windows_talborjt`:

```bat
cd sumo\windows_talborjt
sumo -c agadir_fixed.sumocfg
```

This produces:

```text
tripinfo_fixed.xml
summary_fixed.xml
```

## 5. Run Simulation B: MDP adaptive control

Still from `sumo\windows_talborjt`:

```bat
python sumo_mdp_control.py
```

This produces:

```text
tripinfo_mdp.xml
summary_mdp.xml
mdp_decisions.csv
```

## 6. Analyze results

```bat
python analyze_results.py
```

Take screenshots of:

1. `sumo-gui` with the Talborjt intersection and vehicles;
2. terminal output of `python sumo_mdp_control.py`;
3. terminal output of `python analyze_results.py`.

Then transfer back to the Mac:

```text
tripinfo_fixed.xml
tripinfo_mdp.xml
summary_fixed.xml
summary_mdp.xml
mdp_decisions.csv
screenshots
```
