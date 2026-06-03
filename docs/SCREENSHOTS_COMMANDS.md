# Captures à ajouter au rapport

Ce fichier liste les captures réelles à prendre, les commandes à exécuter et les
noms de fichiers recommandés. Les captures doivent être placées dans
`rapport/figures/`.

## 1. DVC terminal

Objectif : montrer la traçabilité DataOps du dataset et des métriques.

Fichier recommandé :

```text
rapport/figures/dvc_terminal.png
```

Commandes à afficher dans un terminal :

```bash
cd "/Users/mouadassargual/Desktop/Thesis Mouad"
scripts/run_dvc.sh status
scripts/run_dvc.sh dag
scripts/run_dvc.sh metrics show
```

Capture macOS :

```bash
screencapture -i rapport/figures/dvc_terminal.png
```

À montrer dans l'image :

- `dvc.yaml`
- `dvc.lock`
- `metrics/final_yolo26n_step3_960.json`
- si possible le DAG DVC ou les métriques finales.

## 2. MLflow UI

Objectif : montrer le tracking MLOps des métriques et artefacts.

Fichier recommandé :

```text
rapport/figures/mlflow_experiment_details.png
```

Démarrer MLflow :

```bash
cd "/Users/mouadassargual/Desktop/Thesis Mouad"
.venv/bin/mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5056
```

Ouvrir :

```text
http://127.0.0.1:5056
```

Capture macOS :

```bash
screencapture -i rapport/figures/mlflow_experiment_details.png
```

À montrer dans l'image :

- expérience `smart-traffic-agadir-yolo26n`
- run final `YOLO26n-Step3-960-final-52b92829`
- métriques `mAP`, `FPS`, `latency`
- artefacts `.pt`, `.onnx`, `.csv`, `.json`.

## 3. GitHub repository

Objectif : documenter DevOps et project management.

Fichier recommandé :

```text
rapport/figures/github_branches.png
```

Après push GitHub, ouvrir le dépôt dans le navigateur et capturer :

- page principale du repo
- branches `main` et `develop`
- éventuellement Pull Request `develop -> main`

Capture macOS :

```bash
screencapture -i rapport/figures/github_branches.png
```

## 4. Raspberry Pi 5 benchmark terminal

Objectif : prouver le benchmark réel sur Pi 5.

Fichier recommandé :

```text
rapport/figures/pi5_benchmark_terminal.png
```

Commandes à lancer sur le Raspberry Pi 5 :

```bash
ssh mouad@mouadpi.local
cd ~/thesis
source pfe_env/bin/activate
python3 -m pipeline.main \
  --video data/videos/ne8th/Bellevue_Bellevue_NE8th__2017-09-11_14-08-31_3min.mp4 \
  --model models/downloads/YOLO26n_step3_800_from960_best.onnx \
  --imgsz 800 \
  --person-roi \
  --max-frames 300
```

Test stride 2 recommandé :

```bash
python3 -m pipeline.main \
  --video data/videos/ne8th/Bellevue_Bellevue_NE8th__2017-09-11_14-08-31_3min.mp4 \
  --model models/downloads/YOLO26n_step3_800_from960_best.onnx \
  --imgsz 800 \
  --person-roi \
  --vid-stride 2 \
  --max-frames 300
```

À montrer dans l'image :

- modèle ONNX utilisé
- `FPS moyen`
- nombre de frames
- anonymisation
- décisions MDP.

Capture depuis le Mac :

```bash
screencapture -i rapport/figures/pi5_benchmark_terminal.png
```

## 5. Setup matériel

Objectif : montrer le contexte réel du déploiement embarqué.

Fichier recommandé :

```text
rapport/figures/setup_raspberry_pi5.jpg
```

À photographier :

- Raspberry Pi 5 allumé
- alimentation
- connexion réseau
- écran/terminal ou dashboard si possible.

## 6. Dashboard final

Objectif : preuve fonctionnelle du pipeline complet.

Fichiers déjà présents :

```text
rapport/figures/dashboard_complet.png
rapport/figures/image_analysee.png
rapport/figures/anonymisation.png
rapport/figures/decision_mdp.png
```

À refaire seulement si tu veux une version plus récente avec le modèle 800.

Commande dashboard :

```bash
python3 -m pipeline.main \
  --video data/videos/ne8th/Bellevue_Bellevue_NE8th__2017-09-11_14-08-31_3min.mp4 \
  --model models/downloads/YOLO26n_step3_800_from960_best.onnx \
  --imgsz 800 \
  --dashboard \
  --person-roi
```

Ouvrir :

```text
http://127.0.0.1:5000
```

Capture :

```bash
screencapture -i rapport/figures/dashboard_complet.png
```
