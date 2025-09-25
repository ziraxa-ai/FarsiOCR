# Dataset/Create Dataset/validate_dataset.py
from pathlib import Path
import pandas as pd
from PIL import Image
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
META = SCRIPT_DIR/"dataset.csv"

def main():
    assert META.exists(), f"Not found: {META}"
    df = pd.read_csv(META)
    print("[INFO] rows:", len(df))
    # مسیرها
    missing = []
    for p in df["image_path"]:
        if not (SCRIPT_DIR/p).exists():
            missing.append(p)
    print("[CHECK] missing files:", len(missing))
    # سایزها
    bad_size=[]
    for p in df["image_path"].head(2000):  # برای سرعت، بخشی را تست می‌کنیم (یا همه، اگر خواستی)
        im = Image.open(SCRIPT_DIR/p)
        if im.mode!="L":
            bad_size.append((p,"mode",im.mode))
        if im.size[0]!=im.size[1]:
            bad_size.append((p,"not_square",im.size))
    print("[CHECK] bad images:", len(bad_size))
    # توزیع کلاس‌ها
    by_label = Counter(df["label"])
    print("[DIST] labels:", by_label)
    by_font = Counter(df["font"])
    print("[DIST] top fonts:", by_font.most_common(5))

if __name__=="__main__":
    main()
