from __future__ import annotations
from pathlib import Path
import argparse, random, hashlib, math, shutil
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
from fontTools.ttLib import TTFont
from tqdm import tqdm

PERSIAN_CHARS = list("ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")
LABELS = [
    'Alef','Be','Pe','Te','Se','Jim','Che','He','Khe','Dal','Zal','Re','Ze','Zhe',
    'Sin','Shin','Sad','Zad','Ta','Za','Ayin','Ghayin','Fe','Ghaf','Kaf','Gaf',
    'Lam','Mim','Noon','Vav','Heh','Ye'
]
assert len(PERSIAN_CHARS) == len(LABELS) == 32

def list_font_files(fonts_dir: Path) -> List[Path]:
    exts = (".ttf",".otf",".ttc",".TTF",".OTF",".TTC")
    return sorted([p for p in fonts_dir.glob("*") if p.suffix in exts])

def auto_locate_fonts(script_dir: Path) -> Path | None:
    for p in [script_dir/"Fonts", script_dir/"Datasets"/"Fonts", script_dir.parent/"Fonts"]:
        if p.exists() and any(p.iterdir()):
            return p
    return None

def font_supports_char(font_path: Path, ch: str) -> bool:
    try:
        tt = TTFont(str(font_path), lazy=True)
        for t in tt["cmap"].tables:
            if ord(ch) in t.cmap:
                return True
    except Exception:
        pass
    return False

def render_centered(char: str, font_path: Path, font_size: int, canvas: Tuple[int,int],
                    bg: int=0, fg: int=255) -> Image.Image:
    img = Image.new('L', canvas, color=bg)
    font = ImageFont.truetype(str(font_path), font_size)
    draw = ImageDraw.Draw(img)
    try:
        bbox = draw.textbbox((0,0), char, font=font)
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        x = (canvas[0]-w)//2 - bbox[0]
        y = (canvas[1]-h)//2 - bbox[1]
    except Exception:
        w, h = draw.textsize(char, font=font)
        x = (canvas[0]-w)//2
        y = (canvas[1]-h)//2
    draw.text((x, y), char, font=font, fill=fg)
    return img

# --- light augmentations (deterministic with seed passed externally) ---
def augment(img: Image.Image, mode: str) -> Image.Image:
    if mode == "rot":
        return img.rotate(np.random.uniform(-10,10), resample=Image.BILINEAR, fillcolor=0)
    if mode == "shift":
        tx = int(np.random.uniform(-3,3))
        ty = int(np.random.uniform(-3,3))
        return ImageOps.expand(ImageOps.crop(img, (max(0,-tx), max(0,-ty), max(0,tx), max(0,ty))), border=(max(0,tx),max(0,ty),max(0,-tx),max(0,-ty)), fill=0)
    if mode == "noise":
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, 8, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")
    if mode == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=0.6))
    if mode == "invert":
        return ImageOps.invert(img)
    return img

def sha1_of_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:10]

def make_mosaic(paths: List[Path], out_path: Path, cols=8, cell=64):
    if not paths: return
    rows = math.ceil(len(paths)/cols)
    mosaic = Image.new('L', (cols*cell, rows*cell), color=0)
    for i,p in enumerate(paths[:rows*cols]):
        try:
            im = Image.open(p).convert('L').resize((cell,cell), Image.NEAREST)
            x = (i%cols)*cell; y = (i//cols)*cell
            mosaic.paste(im, (x,y))
        except: pass
    mosaic.save(out_path)

def split_by_font(rows: List[Dict], ratios=(0.8,0.1,0.1), seed=42):
    random.seed(seed)
    # برای جلوگیری از لیک، فونت‌های هر کلاس را به split ها اختصاص می‌دهیم
    by_label: Dict[str, Dict[str, List[Dict]]] = {}
    for r in rows:
        by_label.setdefault(r["label"], {}).setdefault(r["font_stem"], []).append(r)
    train,val,test = [],[],[]
    for lab, font_map in by_label.items():
        fonts = list(font_map.keys())
        random.shuffle(fonts)
        n = len(fonts)
        n_train = int(n*ratios[0]); n_val = int(n*ratios[1])
        for i,f in enumerate(fonts):
            bucket = train if i < n_train else (val if i < n_train+n_val else test)
            bucket.extend(font_map[f])
    return train,val,test

def stratified_split(rows: List[Dict], by_label="label", ratios=(0.8,0.1,0.1), seed=42):
    random.seed(seed)
    g: Dict[str,List[Dict]] = {}
    for r in rows: g.setdefault(r[by_label], []).append(r)
    train,val,test = [],[],[]
    for lab, items in g.items():
        random.shuffle(items)
        n=len(items); n_tr=int(n*ratios[0]); n_val=int(n*ratios[1])
        train+=items[:n_tr]; val+=items[n_tr:n_tr+n_val]; test+=items[n_tr+n_val:]
    return train,val,test

def main():
    parser = argparse.ArgumentParser(description="Pro generator for Persian alphabet from fonts")
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--fonts-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=script_dir/"DS-3")
    parser.add_argument("--meta-csv", type=Path, default=script_dir/"dataset.csv")
    parser.add_argument("--sizes", type=int, nargs="+", default=[64], help="e.g. --sizes 64 128")
    parser.add_argument("--font-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", type=float, nargs=3, default=(0.8,0.1,0.1))
    parser.add_argument("--split-by-font", action="store_true", help="avoid font leakage between splits")
    parser.add_argument("--augment", type=int, default=0, help="augmentations per (char,font,size). 0 = off")
    parser.add_argument("--preview", action="store_true", help="save mosaic previews per label")
    parser.add_argument("--build-npz", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed); random.seed(args.seed)

    FONTS_DIR = args.fonts_dir or auto_locate_fonts(script_dir)
    if not FONTS_DIR or not FONTS_DIR.exists():
        raise RuntimeError("Fonts dir not found. Use --fonts-dir or create ./Fonts")
    font_files = list_font_files(FONTS_DIR)
    if not font_files: raise RuntimeError(f"No fonts in {FONTS_DIR}")

    out_root = args.output_dir; meta_csv = args.meta_csv
    if out_root.exists(): shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    preview_dir = script_dir/"preview"; preview_dir.mkdir(exist_ok=True)

    rows: List[Dict] = []
    print(f"[INFO] Fonts: {len(font_files)} | Sizes: {args.sizes} | Output: {out_root}")

    for idx,(ch,label) in enumerate(zip(PERSIAN_CHARS, LABELS)):
        label_dir = out_root/label; label_dir.mkdir(parents=True, exist_ok=True)
        char_code = f"U+{ord(ch):04X}"
        for fpath in tqdm(font_files, desc=f"{idx:02d}-{label}", leave=False):
            if not font_supports_char(fpath, ch): continue
            font_stem = fpath.stem
            for size in args.sizes:
                base_img = render_centered(ch, fpath, args.font_size, (size,size), bg=0, fg=255)
                # اصل
                payload = np.array(base_img, dtype=np.uint8).tobytes()
                h = sha1_of_bytes(payload)
                base_name = f"{label}__{char_code}__{font_stem}__sz{size}__base-{h}.png"
                base_path = label_dir / base_name
                base_img.save(base_path)
                rows.append({
                    "label": label, "label_id": idx, "char": ch, "char_code": char_code,
                    "font": fpath.name, "font_stem": font_stem, "size": size,
                    "augment_id": "base", "image_path": str(base_path.relative_to(script_dir))
                })
                # افزوده‌ها
                modes = ["rot","shift","noise","blur","invert"]
                for k in range(args.augment):
                    mode = modes[k % len(modes)]
                    aug = augment(base_img, mode)
                    payload = np.array(aug, dtype=np.uint8).tobytes()
                    h = sha1_of_bytes(payload)
                    name = f"{label}__{char_code}__{font_stem}__sz{size}__aug{mode}-{k}-{h}.png"
                    pth = label_dir / name
                    aug.save(pth)
                    rows.append({
                        "label": label, "label_id": idx, "char": ch, "char_code": char_code,
                        "font": fpath.name, "font_stem": font_stem, "size": size,
                        "augment_id": f"aug-{mode}-{k}", "image_path": str(pth.relative_to(script_dir))
                    })

        # پیش‌نمایش موزاییک (اختیاری)
        if args.preview:
            sample_paths = sorted(label_dir.glob("*.png"))[:64]
            make_mosaic(sample_paths, preview_dir/f"{label}_preview.png", cols=8, cell=64)

    # ذخیره متادیتا
    df = pd.DataFrame(rows)
    meta_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(meta_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Metadata -> {meta_csv} (rows={len(df)})")

    # Split
    ratios = tuple(args.splits)
    if args.split_by_font:
        train,val,test = split_by_font(rows, ratios=ratios, seed=args.seed)
    else:
        train,val,test = stratified_split(rows, ratios=ratios, seed=args.seed)
    for name, part in [("train",train),("val",val),("test",test)]:
        pd.DataFrame(part).to_csv(script_dir/f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] Splits -> {script_dir/'train.csv'}, {script_dir/'val.csv'}, {script_dir/'test.csv'}")

    # NPZ (اختیاری)
    if args.build_npz:
        X,y=[],[]
        for r in tqdm(rows, desc="Building NPZ"):
            p = script_dir/ r["image_path"]
            arr = np.array(Image.open(p).convert("L"), dtype=np.uint8)
            X.append(arr); y.append(r["label_id"])
        X = np.stack(X,0); y=np.array(y,dtype=np.int64)
        np.savez_compressed(script_dir/"persian_alphabet.npz", images=X, labels=y)
        print(f"[OK] NPZ -> {script_dir/'persian_alphabet.npz'}")

if __name__ == "__main__":
    main()
