import os, shutil, random, json, hashlib
from pathlib import Path
from collections import defaultdict
from PIL import Image

random.seed(42)

# ✅ Chemin dataset KaggleHub fourni par l'utilisateur
RAW = Path(r"C:\Users\Amine\.cache\kagglehub\datasets\farzadnekouei\trash-type-image-dataset\versions\1")
OUT = Path("data")

SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}
IMG_EXT = {".jpg",".jpeg",".png",".bmp",".webp"}

# Remap noms d'origine -> 5 classes finales
CANON = {
    "paper":"papier_emballage", "cardboard":"papier_emballage", "paper_cardboard":"papier_emballage",
    "glass":"verre", "bottle_glass":"verre",
    "plastic":"plastique", "bottle_plastic":"plastique",
    "metal":"metal", "can":"metal", "aluminum":"metal",
    "trash":"non_recyclable", "other":"non_recyclable", "organic":"non_recyclable",
}

def is_image_ok(p: Path) -> bool:
    try:
        with Image.open(p) as im: im.verify()
        return True
    except Exception:
        return False

def hash_file(p: Path)->str:
    h = hashlib.md5()
    with open(p,'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

def gather_images(root: Path):
    files_by_class = defaultdict(list)
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT and is_image_ok(p):
            src = p.parent.name.lower().strip()
            tgt = CANON.get(src)
            if tgt:
                files_by_class[tgt].append(p)
    return files_by_class

def main():
    assert RAW.exists(), f"RAW introuvable: {RAW}"
    files_by_class = gather_images(RAW)
    OUT.mkdir(exist_ok=True)
    for sp in SPLIT: (OUT/sp).mkdir(exist_ok=True)

    seen = set()
    stats = {}
    for cls, files in files_by_class.items():
        uniq = []
        for f in files:
            try:
                h = hash_file(f)
            except Exception:
                continue
            if h not in seen:
                seen.add(h); uniq.append(f)

        random.shuffle(uniq)
        n = len(uniq); n_tr = int(n*SPLIT["train"]); n_va = int(n*SPLIT["val"])
        buckets = {"train": uniq[:n_tr], "val": uniq[n_tr:n_tr+n_va], "test": uniq[n_tr+n_va:]}
        for sp, lst in buckets.items():
            d = OUT/sp/cls; d.mkdir(parents=True, exist_ok=True)
            for src in lst: shutil.copy2(src, d/src.name)
        stats[cls] = {k: len(v) for k,v in buckets.items()}

    with open("data_stats.json","w",encoding="utf-8") as f: json.dump(stats,f,indent=2,ensure_ascii=False)
    classes = sorted(stats.keys())
    with open("class_mapping.json","w",encoding="utf-8") as f: json.dump({i:c for i,c in enumerate(classes)}, f, indent=2, ensure_ascii=False)
    print("OK → data/{train,val,test} + data_stats.json + class_mapping.json ✅")

if __name__ == "__main__":
    main()