# DataOps / MLOps / DevOps local

Ce projet utilise une couche locale légère pour tracer le dataset final, les paramètres, les métriques, les artefacts du modèle et le versioning du code.

## Installation

```bash
.venv/bin/python -m pip install -r requirements-mlops.txt
```

## GitHub

Le code source et le rapport sont versionnés dans GitHub avec deux branches :

```text
main    -> version stable pour soutenance / livrable
develop -> intégration des corrections, figures et évolutions
```

Les données volumineuses, vidéos, poids `.pt`, exports `.onnx`, cache DVC et runs MLflow ne sont pas poussés dans GitHub. Ils restent documentés par `dvc.yaml`, `dvc.lock`, `params.yaml` et `metrics/`.

Voir :

```text
docs/GITHUB_WORKFLOW.md
docs/SCREENSHOTS_COMMANDS.md
```

## DVC

Le dépôt DVC est initialisé en mode local sans Git (`no_scm = true`). Le wrapper `scripts/run_dvc.sh` force le cache DVC dans le projet pour éviter les problèmes de permissions macOS.

```bash
scripts/run_dvc.sh dag
scripts/run_dvc.sh status
scripts/run_dvc.sh metrics show
```

Artefact dataset suivi par DVC :

```text
data/dataset/dataset_step3_tiny_person_crops.dvc
```

Dataset final suivi :

```text
data/dataset/dataset_step3_tiny_person_crops
```

## MLflow

Le tracking MLflow utilise une base SQLite locale :

```text
mlflow.db
```

Les artefacts sont stockés dans :

```text
mlruns/
```

Mettre à jour le run final :

```bash
.venv/bin/python scripts/log_mlflow_final.py
```

Ouvrir l'interface MLflow :

```bash
.venv/bin/mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5056
```

Run final actuel :

```text
YOLO26n-Step3-960-final-52b92829
```

## Artefacts modèle

Le fichier PyTorch final est synchronisé localement et suivi par DVC :

```text
models/downloads/YOLO26n_step3_960_best.pt
```

L'artefact utilisé pour le déploiement Raspberry Pi est l'export ONNX suivi par DVC :

```text
models/downloads/YOLO26n_step3_960_best.onnx
```

## Redéploiement Raspberry Pi 5 après formatage

Après formatage du Raspberry Pi, la clé SSH change. Sur le Mac, supprimer l'ancienne entrée :

```bash
ssh-keygen -R mouadpi.local
```

Puis tester la connexion. La première connexion doit accepter la nouvelle empreinte SSH :

```bash
ssh mouad@mouadpi.local
```

Si l'accès par clé SSH n'est pas encore configuré, copier la clé publique du Mac vers le Pi :

```bash
ssh-copy-id mouad@mouadpi.local
```

Préparer une archive minimale pour le benchmark Pi 5 depuis le Mac :

```bash
tar -czf /tmp/thesis_pi5_benchmark.tar.gz \
  pipeline \
  requirements.txt \
  models/downloads/YOLO26n_step3_960_best.onnx \
  data/zones \
  data/videos/ne8th/Bellevue_Bellevue_NE8th__2017-09-11_14-08-31_3min.mp4
```

Copier et extraire sur le Pi :

```bash
ssh mouad@mouadpi.local "mkdir -p ~/thesis"
scp /tmp/thesis_pi5_benchmark.tar.gz mouad@mouadpi.local:~/
ssh mouad@mouadpi.local "tar -xzf ~/thesis_pi5_benchmark.tar.gz -C ~/thesis"
```

Installer l'environnement Python sur le Pi :

```bash
ssh mouad@mouadpi.local
cd ~/thesis
sudo apt update
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0
python3 -m venv pfe_env
source pfe_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Lancer le benchmark final avec dashboard :

```bash
cd ~/thesis
source pfe_env/bin/activate
python3 -m pipeline.main \
  --video data/videos/ne8th/Bellevue_Bellevue_NE8th__2017-09-11_14-08-31_3min.mp4 \
  --model models/downloads/YOLO26n_step3_960_best.onnx \
  --imgsz 960 \
  --conf 0.35 \
  --dashboard \
  --person-roi \
  --vehicle-roi
```

Le dashboard doit ensuite être ouvert depuis le navigateur :

```text
http://mouadpi.local:5000
```

Résultat mesuré sur Raspberry Pi 5 pour le modèle final Step 3 960 avec ROI personnes et ROI véhicules :

```text
300 frames, FPS moyen = 2.16, latence moyenne ~= 463 ms/frame
2031 détections anonymisées, 585 tracks cumulés
```

Deux variantes plus légères ont ensuite été mesurées sur la même vidéo :

| Configuration | FPS moyen | Latence approx. | Anonymisés | Tracks |
| --- | ---: | ---: | ---: | ---: |
| ROI personnes + ROI véhicules | 2.16 | 463 ms/frame | 2031 | 585 |
| ROI personnes toutes les 10 frames, sans ROI véhicules | 2.48 | 403 ms/frame | 1438 | 311 |
| Sans ROI additionnelles | 2.51 | 398 ms/frame | 875 | 190 |

Cette configuration est retenue comme mode d'analyse haute précision. Le faible écart entre 2.16 et 2.51 FPS montre que le coût principal vient de l'inférence ONNX en `960x960`, plus que des ROI. Le fichier ONNX final est exporté avec une entrée statique `960x960` : il ne faut donc pas le lancer avec `--imgsz 800`. Un vrai test `800x800` nécessite un export ONNX 800 ou un export ONNX dynamique.

Commande de comparaison plus légère :

```bash
python3 -m pipeline.main \
  --video data/videos/ne8th/Bellevue_Bellevue_NE8th__2017-09-11_14-08-31_3min.mp4 \
  --model models/downloads/YOLO26n_step3_960_best.onnx \
  --imgsz 960 \
  --conf 0.35 \
  --person-roi \
  --roi-every 10 \
  --max-frames 300
```

Commande la plus légère avec le même ONNX statique :

```bash
python3 -m pipeline.main \
  --video data/videos/ne8th/Bellevue_Bellevue_NE8th__2017-09-11_14-08-31_3min.mp4 \
  --model models/downloads/YOLO26n_step3_960_best.onnx \
  --imgsz 960 \
  --conf 0.35 \
  --max-frames 300
```

### Export ONNX 800 expérimental

Pour tester un vrai `800x800`, un second artefact ONNX est exporté depuis les poids finaux Step 3 960 sans écraser le modèle officiel :

```text
models/downloads/YOLO26n_step3_800_from960_best.onnx
```

Reproduction locale :

```bash
scripts/run_dvc.sh repro export_yolo26n_step3_800_from960
```

Vérification attendue :

```text
Entrée ONNX : [1, 3, 800, 800]
Sortie ONNX : [1, 300, 6]
```

Copie vers le Raspberry Pi :

```bash
scp models/downloads/YOLO26n_step3_800_from960_best.onnx \
  mouad@mouadpi.local:/home/mouad/thesis/models/downloads/
```

Commande benchmark Pi 5 utilisée sans ROI additionnelles :

```bash
cd ~/thesis
source pfe_env/bin/activate
python3 -m pipeline.main \
  --video data/videos/ne8th/Bellevue_Bellevue_NE8th__2017-09-11_14-08-31_3min.mp4 \
  --model models/downloads/YOLO26n_step3_800_from960_best.onnx \
  --imgsz 800 \
  --conf 0.35 \
  --max-frames 300
```

Commande benchmark Pi 5 utilisée avec anonymisation renforcée :

```bash
python3 -m pipeline.main \
  --video data/videos/ne8th/Bellevue_Bellevue_NE8th__2017-09-11_14-08-31_3min.mp4 \
  --model models/downloads/YOLO26n_step3_800_from960_best.onnx \
  --imgsz 800 \
  --conf 0.35 \
  --person-roi \
  --roi-every 10 \
  --max-frames 300
```

Résultats mesurés sur Raspberry Pi 5 avec cet ONNX 800 :

| Configuration ONNX 800 | FPS moyen | Latence approx. | Anonymisés | Tracks |
| --- | ---: | ---: | ---: | ---: |
| Sans ROI additionnelles | 3.94 | 254 ms/frame | 1022 | 199 |
| ROI personnes toutes les 10 frames | 3.45 | 290 ms/frame | 1440 | 290 |

L'export 800 améliore nettement le débit par rapport au 960 statique : `3.94 FPS` contre `2.51 FPS` sans ROI, et `3.45 FPS` contre `2.48 FPS` avec ROI personnes toutes les 10 frames. Il reste cependant inférieur au seuil de 5 FPS fixé pour le temps réel fluide. Il constitue donc un meilleur candidat de déploiement que le 960 pour le Pi, mais pas encore une validation complète du temps réel.

### Stride vidéo et INT8 dynamique

Le pipeline accepte maintenant `--vid-stride N`. Avec `--vid-stride 2`, une frame source sur deux est inférée. Le rapport produit deux mesures :

- `FPS moyen` : débit réel des inférences IA.
- `FPS effectif` : couverture vidéo source, soit environ `FPS moyen x vid_stride`.

Avant de tester le stride sur le Pi, copier le nouveau `main.py` :

```bash
scp pipeline/main.py mouad@mouadpi.local:/home/mouad/thesis/pipeline/main.py
```

Benchmark FP32 800 avec stride 2 :

```bash
cd ~/thesis
source pfe_env/bin/activate
python3 -m pipeline.main \
  --video data/videos/ne8th/Bellevue_Bellevue_NE8th__2017-09-11_14-08-31_3min.mp4 \
  --model models/downloads/YOLO26n_step3_800_from960_best.onnx \
  --imgsz 800 \
  --conf 0.35 \
  --vid-stride 2 \
  --max-frames 300
```

Benchmark FP32 800 avec ROI personnes et stride 2 :

```bash
python3 -m pipeline.main \
  --video data/videos/ne8th/Bellevue_Bellevue_NE8th__2017-09-11_14-08-31_3min.mp4 \
  --model models/downloads/YOLO26n_step3_800_from960_best.onnx \
  --imgsz 800 \
  --conf 0.35 \
  --person-roi \
  --roi-every 10 \
  --vid-stride 2 \
  --max-frames 300
```

Un modèle INT8 dynamique expérimental est également généré :

```text
models/downloads/YOLO26n_step3_800_from960_int8.onnx
```

Il est plus léger (`2.8 MB` contre `9.4 MB`) et conserve les détections sur un test local rapide. Le benchmark Pi 5 reste indispensable, car l'accélération INT8 dépend fortement du runtime ARM et du provider ONNX Runtime disponible.

Copie vers le Raspberry Pi :

```bash
scp models/downloads/YOLO26n_step3_800_from960_int8.onnx \
  mouad@mouadpi.local:/home/mouad/thesis/models/downloads/
```

Benchmark INT8 dynamique :

```bash
python3 -m pipeline.main \
  --video data/videos/ne8th/Bellevue_Bellevue_NE8th__2017-09-11_14-08-31_3min.mp4 \
  --model models/downloads/YOLO26n_step3_800_from960_int8.onnx \
  --imgsz 800 \
  --conf 0.35 \
  --max-frames 300
```

Benchmark INT8 dynamique avec stride 2 :

```bash
python3 -m pipeline.main \
  --video data/videos/ne8th/Bellevue_Bellevue_NE8th__2017-09-11_14-08-31_3min.mp4 \
  --model models/downloads/YOLO26n_step3_800_from960_int8.onnx \
  --imgsz 800 \
  --conf 0.35 \
  --vid-stride 2 \
  --max-frames 300
```

Résultats mesurés sur Raspberry Pi 5 :

| Configuration | Frames source | Frames IA | FPS moyen IA | FPS effectif source | Latence approx. | Anonymisés | Tracks | Décision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ONNX 800 FP32, sans ROI, stride 1 | 300 | 300 | 3.94 | 3.94 | 254 ms/frame | 1022 | 199 | Retenu si chaque frame doit être inférée |
| ONNX 800 FP32, ROI personnes toutes les 10 frames, stride 1 | 300 | 300 | 3.45 | 3.45 | 290 ms/frame | 1440 | 290 | Retenu pour anonymisation renforcée |
| ONNX 800 FP32, sans ROI, stride 2 | 300 | 150 | 4.28 | 7.84 | 234 ms/inférence | 541 | 132 | Meilleur compromis FPS/couverture |
| ONNX 800 INT8 dynamique, sans ROI, stride 1 | 300 | 300 | 2.06 | 2.06 | 485 ms/frame | 1140 | 210 | Non retenu |
| ONNX 800 INT8 dynamique, sans ROI, stride 2 | 300 | 150 | 1.97 | 3.76 | 508 ms/inférence | 617 | 138 | Non retenu |

Interprétation :

- Le meilleur débit brut sans sauter de frame reste l'ONNX 800 FP32 sans ROI : `3.94 FPS`.
- Le meilleur compromis pour dépasser le seuil de `5 FPS` en couverture vidéo est l'ONNX 800 FP32 avec `--vid-stride 2` : `7.84 frames source/s`.
- L'INT8 dynamique n'accélère pas ce pipeline sur Raspberry Pi 5 avec ONNX Runtime CPU. Malgré une taille de fichier plus faible, le runtime exécute ce graphe plus lentement que le FP32.
- Pour le rapport, il faut distinguer honnêtement `FPS moyen IA` et `FPS effectif source`. Le stride améliore la couverture vidéo, mais seulement une frame source sur deux est réellement inférée.
