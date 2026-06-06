# Captures finales à ajouter au rapport

Ce fichier liste les captures réelles à prendre, les commandes à exécuter et les
noms de fichiers recommandés. Les captures doivent être placées dans
`rapport/figures/`.

Les captures DVC et GitHub ne sont pas retenues dans la liste finale. La priorité
est donnée aux preuves directement liées au modèle, au pipeline et au
déploiement Raspberry Pi 5.

## 1. MLflow UI

Objectif : montrer le tracking MLOps des métriques et artefacts.

Fichier recommandé :

```text
rapport/figures/mlflow_ui.png
```

Démarrer MLflow :

```bash
cd "/Users/mouadassargual/Desktop/Thesis Mouad"
.venv/bin/python scripts/log_mlflow_final.py
.venv/bin/mlflow ui \
  --backend-store-uri "sqlite:///$(pwd)/mlflow.db" \
  --default-artifact-root "file://$(pwd)/mlruns" \
  --host 127.0.0.1 \
  --port 5057
```

Ouvrir :

```text
http://127.0.0.1:5057
```

Lien direct vers l'expérience :

```text
http://127.0.0.1:5057/#/experiments/1
```

Important avec MLflow 3.x :

- si le bouton `GenAI` est sélectionné, la page affiche les traces LLM ;
- pour ce projet, cliquer sur `Model training` en haut à gauche ;
- capturer la table des runs, pas la page `Traces`.

Capture macOS :

```bash
screencapture -i rapport/figures/mlflow_ui.png
```

À montrer dans l'image :

- expérience `smart-traffic-agadir-yolo26n`
- run final `YOLO26n-Step3-960-final-6582f915`
- métriques `mAP`, `FPS`, `latency`
- métriques de validation/test et métriques Raspberry Pi 5 ;
- paramètres dataset / entraînement / inférence.

Capture optionnelle complémentaire :

```text
rapport/figures/mlflow_artifacts.png
```

Elle peut montrer les artefacts `.pt`, `.onnx`, `.csv`, `.json` et
`params.yaml`.

Si MLflow apparaît vide :

- vérifier que `Model training` est sélectionné, pas `GenAI` ;
- vérifier que la commande est lancée depuis la racine du projet ;
- vérifier que l'option `--backend-store-uri "sqlite:///$(pwd)/mlflow.db"` est bien utilisée ;
- ouvrir directement `http://127.0.0.1:5057/#/experiments/1` ;
- relancer `scripts/log_mlflow_final.py` avant d'ouvrir l'UI ;
- si un ancien serveur MLflow tourne déjà sur `5056`, garder `5057` ou arrêter l'ancien serveur.

## 2. Raspberry Pi 5 benchmark terminal

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

## 3. Setup matériel

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

## 4. Dashboard final

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
