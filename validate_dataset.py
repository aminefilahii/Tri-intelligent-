from pathlib import Path
from PIL import Image

def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

def main(root="data"):
    root = Path(root)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    bad = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            if not is_valid_image(p):
                bad.append(p)

    if bad:
        print("Images corrompues / non supportées :")
        for b in bad:
            print(" -", b)
    else:
        print("Dataset OK ✅")

if __name__ == "__main__":
    main()