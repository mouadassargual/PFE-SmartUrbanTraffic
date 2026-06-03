# Smart Traffic Agadir

Pipeline de gestion intelligente du trafic urbain pour un PFE M2 IA embarquée.
Le projet combine détection YOLO26n, anonymisation Privacy-by-Design, tracking,
décision MDP et dashboard Flask, avec benchmark sur Raspberry Pi 5.

## Structure

- `pipeline/` : pipeline Python, détection, anonymisation, tracking, MDP, dashboard.
- `scripts/` : scripts DataOps/MLOps, export ONNX, quantification, logs MLflow.
- `notebooks/` : notebooks Colab pour fine-tuning YOLO.
- `rapport/` : rapport LaTeX et figures.
- `metrics/` : métriques finales JSON.
- `dvc.yaml`, `dvc.lock`, `params.yaml` : traçabilité DataOps/MLOps.
- `MLOPS.md` : guide local DVC/MLflow.

## Artefacts lourds

Les datasets, vidéos, poids `.pt`, exports `.onnx`, caches DVC et runs MLflow
ne sont pas poussés dans GitHub. Ils restent suivis localement via DVC,
documentés dans `params.yaml`, `dvc.lock` et `metrics/`.

## Branches Git

- `main` : version stable utilisée pour le rapport et la soutenance.
- `develop` : branche d'intégration pour les corrections et évolutions.

Workflow recommandé :

```bash
git checkout develop
git add .
git commit -m "Describe change"
git push
```

Puis ouvrir une Pull Request `develop -> main` avant une version stable.

## Exécution rapide

```bash
python3 -m pipeline.main \
  --video data/videos/ne8th/Bellevue_Bellevue_NE8th__2017-09-11_14-08-31_3min.mp4 \
  --model models/downloads/YOLO26n_step3_800_from960_best.onnx \
  --imgsz 800 \
  --person-roi
```

Dashboard :

```bash
python3 -m pipeline.main \
  --video data/videos/ne8th/Bellevue_Bellevue_NE8th__2017-09-11_14-08-31_3min.mp4 \
  --model models/downloads/YOLO26n_step3_800_from960_best.onnx \
  --imgsz 800 \
  --dashboard \
  --person-roi
```

## Reproductibilité

Voir :

- `MLOPS.md`
- `docs/SCREENSHOTS_COMMANDS.md`
- `docs/GITHUB_WORKFLOW.md`
