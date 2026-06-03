# Smart Traffic Agadir - Deploiement Raspberry Pi

Ce paquet sert a deployer la derniere version du pipeline quand le Raspberry Pi sera de nouveau accessible.

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
