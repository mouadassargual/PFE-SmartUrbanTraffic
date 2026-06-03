# Workflow GitHub

Le dépôt GitHub sert au versioning du code, du rapport LaTeX et des fichiers de
configuration. Les datasets, vidéos, poids de modèles, exports ONNX, caches DVC
et runs MLflow restent hors Git.

## Branches

- `main` : branche stable, utilisée pour la soutenance et les versions validées.
- `develop` : branche d'intégration pour les corrections, figures, scripts et
  évolutions du pipeline.

Workflow :

```text
develop -> Pull Request -> main
```

## Initialisation locale

```bash
cd "/Users/mouadassargual/Desktop/Thesis Mouad"
git init -b main
git add .
git commit -m "Initial smart traffic thesis pipeline"
git checkout -b develop
```

## Création du dépôt GitHub

Option avec GitHub CLI :

```bash
gh auth login
gh repo create smart-traffic-agadir --private --source=. --remote=origin --push
git checkout main
git push -u origin main
git checkout develop
git push -u origin develop
```

Option sans GitHub CLI :

1. Créer un dépôt vide sur GitHub.
2. Copier l'URL SSH ou HTTPS.
3. Lancer :

```bash
git remote add origin git@github.com:<USER>/smart-traffic-agadir.git
git checkout main
git push -u origin main
git checkout develop
git push -u origin develop
```

## Règles de gestion

- Les commits de correction se font sur `develop`.
- `main` reçoit uniquement des versions stables via Pull Request.
- Les fichiers lourds restent exclus par `.gitignore`.
- Les métriques finales sont conservées dans `metrics/`.
- Les artefacts de modèles sont documentés dans `params.yaml` et `dvc.lock`.
- Les preuves de benchmark et captures sont stockées dans `rapport/figures/`.

## Vérifications avant push

```bash
git status --short
git branch
find . -maxdepth 3 -type f -size +50M
```

Si un fichier lourd apparaît dans `git status`, l'ajouter à `.gitignore` avant le
commit.
