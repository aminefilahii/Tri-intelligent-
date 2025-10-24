# Tri-intelligent — Projet complet (prêt à lancer)

## 1) Préparer le dataset (split 80/10/10)
```bash
python split_dataset.py
```

## 2) Entraîner le modèle (ResNet18 fine-tuning)
```bash
pip install -r requirements.txt
python train.py
```

## 3) Évaluer le modèle
```bash
python eval.py
```

## 4) Lancer l'API et la page webcam
```bash
uvicorn fastapi.main:app --reload --port 8000
```
Ouvre ensuite `fastapi/templates/webcam.html` dans ton navigateur et clique "Capturer & Envoyer".

### Arborescence
```
Tri_intelligent_ready/
├─ data/ (créé par split_dataset.py)
├─ checkpoints/ (créé par train.py)
├─ split_dataset.py
├─ datasets.py
├─ train.py
├─ eval.py
├─ validate_dataset.py
└─ fastapi/
   ├─ main.py
   ├─ templates/
   │  ├─ index.html
   │  └─ webcam.html
   └─ static/css/main.css
```