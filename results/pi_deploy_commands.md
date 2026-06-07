# Smart Traffic Agadir - Deploiement Raspberry Pi

Ce paquet sert a deployer la derniere version du pipeline quand le Raspberry Pi sera de nouveau accessible.

## Paquet dashboard multi-scenes

La version la plus recente pour le dashboard avec selection `ne8th`, `116th`
et `150th` est :

```bash
results/pi_deploy_dashboard_scenes.tar.gz
```

Elle contient le code, le dashboard, les zones JSON, les modeles ONNX 960/800
et les trois clips courts utilises par les scenes du dashboard.

Copie vers le Pi :

```bash
scp results/pi_deploy_dashboard_scenes.tar.gz mouad@mouadpi.local:/home/mouad/thesis/
```

Installation sur le Pi :

```bash
ssh mouad@mouadpi.local
cd /home/mouad/thesis
tar -xzf pi_deploy_dashboard_scenes.tar.gz
source pfe_env/bin/activate
python3 -m py_compile pipeline/config.py pipeline/dashboard.py pipeline/main.py
```

Lancement recommande :

```bash
python3 -m pipeline.main \
  --model models/downloads/YOLO26n_step3_800_from960_best.onnx \
  --imgsz 800 \
  --dashboard \
  --person-roi \
  --vehicle-roi \
  --vehicle-roi-every 3 \
  --conf 0.18
```

Le dashboard est ensuite disponible sur :

```text
http://mouadpi.local:5000
```

## Fichiers finaux

- `pipeline/main.py`
- `pipeline/detector.py`
- `pipeline/dashboard.py`
- `pipeline/tracker.py`
- `pipeline/anonymizer.py`
- `pipeline/decision.py`
- `pipeline/config.py`
- `models/downloads/YOLO26n_step3_960_best.onnx`
- `models/downloads/YOLO26n_step3_960_best.pt`
- `models/downloads/YOLO26n_step3_960_results.csv`

## Copier l'archive vers le Pi

Remplacer `IP_DU_PI` par la nouvelle IP du Raspberry Pi.

```bash
scp results/pi_deploy_smart_traffic_final.tar.gz mouad@IP_DU_PI:/home/mouad/thesis/
```

## Installer sur le Pi

```bash
ssh mouad@IP_DU_PI
cd /home/mouad/thesis
tar -xzf pi_deploy_smart_traffic_final.tar.gz
```

## Test dashboard modele final 960

```bash
python3 -m pipeline.main \
  --video data/videos/ne8th/Bellevue_Bellevue_NE8th__2017-09-11_14-08-31_3min.mp4 \
  --model models/downloads/YOLO26n_step3_960_best.onnx \
  --imgsz 960 \
  --dashboard \
  --person-roi \
  --vehicle-roi \
  --vehicle-roi-every 3 \
  --conf 0.18 \
  --car-conf 0.20 \
  --truck-conf 0.08 \
  --bus-conf 0.12 \
  --motorcycle-conf 0.12 \
  --emergency-conf 0.10 \
  --person-conf 0.10
```

Depuis le navigateur:

```text
http://IP_DU_PI:5000
```

## Si le FPS est trop bas

Essayer d'abord:

```bash
--vehicle-roi-every 5
```

Puis, si necessaire, revenir au modele 800 comme modele temps reel et garder 960 pour le benchmark accuracy.
