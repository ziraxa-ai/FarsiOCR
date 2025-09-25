# Dataset/Create Dataset/contours_batch.py
from pathlib import Path
import cv2 as cv
import numpy as np

INPUT_DIR   = Path(__file__).resolve().parent / "Real Data"
OUTPUT_DIR  = Path(__file__).resolve().parent / "Real Processed"
TARGET_SIZE = 128
MIN_AREA    = 20
USE_ADAPTIVE = False

def binarize(gray):
    if USE_ADAPTIVE:
        return cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,31,5)
    _,th = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    return th

def pad_square(img, val=0):
    h,w = img.shape[:2]
    if h==w: return img
    if h>w:
        pad=(h-w)//2
        return cv.copyMakeBorder(img,0,0,pad,h-w-pad,cv.BORDER_CONSTANT,value=val)
    pad=(w-h)//2
    return cv.copyMakeBorder(img,pad,w-h-pad,0,0,cv.BORDER_CONSTANT,value=val)

def process_one(img_path: Path, out_dir: Path):
    bgr = cv.imread(str(img_path))
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    th = binarize(gray)
    contours,_ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # sort by row then col
    def key(c):
        x,y,w,h = cv.boundingRect(c)
        return (y//5, x)
    contours = sorted([c for c in contours if cv.contourArea(c)>=MIN_AREA], key=key)
    boxes=[]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem
    count=0
    for c in contours:
        x,y,w,h = cv.boundingRect(c)
        crop = gray[y:y+h, x:x+w]
        crop = pad_square(crop,0)
        crop = cv.resize(crop,(TARGET_SIZE,TARGET_SIZE), interpolation=cv.INTER_AREA)
        count+=1
        cv.imwrite(str(out_dir/f"{stem}_{count:03d}.png"), crop)
        boxes.append((x,y,w,h))
    # preview
    preview = bgr.copy()
    for (x,y,w,h) in boxes:
        cv.rectangle(preview,(x,y),(x+w,y+h),(0,0,255),2)
    cv.imwrite(str(out_dir/f"{stem}_preview.png"), preview)
    return count,len(contours)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_saved=0; total_files=0
    for img_path in INPUT_DIR.glob("*.*"):
        if img_path.suffix.lower() not in {".png",".jpg",".jpeg",".bmp"}: 
            continue
        total_files+=1
        saved,_=process_one(img_path, OUTPUT_DIR)
        total_saved+=saved
    print("="*50)
    print(f"Processed files: {total_files}")
    print(f"Crops saved: {total_saved}")
    print(f"Output: {OUTPUT_DIR}")

if __name__=="__main__":
    main()
