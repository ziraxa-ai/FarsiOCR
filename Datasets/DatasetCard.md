# FarsiOCR Synthetic & Real Dataset

## Summary
Character-level Persian alphabet dataset generated from multiple fonts + extracted real samples.

## Contents
- 32 Persian letters (with ذ)
- Grayscale images, sizes: 64x64 (and optionally 128x128)
- Synthetic (font-rendered) + Real (contour-extracted)

## Structure
- `DS-3/<Label>/*.png`
- `dataset.csv` with columns: `label,label_id,char,char_code,font,font_stem,size,augment_id,image_path`.
- `train.csv`, `val.csv`, `test.csv` (optionally split-by-font).

## Generation
- `generate_dataset_pro.py` with center-placed glyphs, optional augmentations (rotate/shift/noise/blur/invert).
- `contours_batch.py` for real-data extraction (Otsu/Adaptive thresholding).

## Splits
- Stratified by label or font-aware (`--split-by-font`) to avoid leakage.

## Licensing
- Code: MIT
- Fonts: see `FONTS_LICENSES.md`
