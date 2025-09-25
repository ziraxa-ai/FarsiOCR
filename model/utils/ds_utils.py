import pathlib
from pathlib import Path
import pandas as pd, numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

labels_ = [
    'Alef','Be','Pe','Te','Se','Jim','Che','He','Khe','Dal','Zal','Re','Ze','Zhe',
    'Sin','Shin','Sad','Zad','Ta','Za','Ayin','Ghayin','Fe','Ghaf','Kaf','Gaf',
    'Lam','Mim','Noon','Vav','Heh','Ye'
]

label2idx = {l:i for i,l in enumerate(labels_)}
idx2label = {i:l for l,i in label2idx.items()}

def load_split_csv(csv_path:Path):
    return pd.read_csv(csv_path)

def load_image_and_labels(df: pd.DataFrame, repo_root: Path):
    X, y = [], []
    for _, r in df.iterrows():
        p = repo_root/ "Dataset"/ "Create Dataset"/ Path(r["image_path"])
        arr = np.array(Image.open(p).convert("L"),dtype=np.uint8)
        X.append(arr)
        y.append(int(r.get("label_id", label2idx[r["label"]])))
    return np.stack(X, 0), np.array(y, dtype=np.int64)

def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro')
    f1u = f1_score(y_true, y_pred, average='micro')
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(len(labels_))))
    rep = classification_report(y_true, y_pred, target_names=labels_, digits=4)
    return acc, f1m, f1u, rep
