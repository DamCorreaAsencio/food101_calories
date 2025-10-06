# Entrena MobileNetV2 (capa final) sobre Food-101 en CPU, SIN scikit-learn.
import os, random
import os, argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

def get_root():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getenv("FOOD101_ROOT", r"D:\Datasets"),
                    help="Carpeta PADRE que contiene 'food-101'")
    args, _ = ap.parse_known_args()
    return args.root

ROOT = get_root()

# -------- CONFIG --------
ROOT = r"D:\Datasets"          # carpeta PADRE que contiene "food-101"
SEED = 42
N_PER_CLASS_TRAIN = 12         # súbelo/bájalo según tu PC
N_PER_CLASS_VAL   = 5
BATCH_SIZE = 16                # 16 para CPU; si va bien, sube a 32
EPOCHS = 3                     # empieza corto
LR = 1e-3
NUM_WORKERS = 0                # en Windows deja 0
MODEL_OUT = "food101_torch.pth"
CLASSES_OUT = "food101_classes.npy"
# ------------------------

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Transform acorde a MobileNetV2
try:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    transform = weights.transforms()
except Exception:
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

# Cargar datasets
train_ds = datasets.Food101(ROOT, split='train', transform=transform, download=False)
val_ds   = datasets.Food101(ROOT, split='test',  transform=transform, download=False)
classes = train_ds.classes
num_classes = len(classes)
class_to_idx = {c:i for i,c in enumerate(classes)}
print(f"Clases={num_classes} | train={len(train_ds)} | test={len(val_ds)}")

# Función robusta para obtener el índice de clase de un item i
def label_idx(ds, i):
    # 1) Algunos datasets tienen .targets
    if hasattr(ds, "targets"):
        return int(ds.targets[i])
    # 2) Food101 suele tener ._labels (nombres de clase)
    if hasattr(ds, "_labels"):
        lbl = ds._labels[i]
        if isinstance(lbl, (int, np.integer)):
            return int(lbl)
        return class_to_idx[str(lbl)]
    # 3) Fallback lento pero seguro
    _, y = ds[i]
    return int(y)

# Construir subconjuntos balanceados por clase
per_cls_train, per_cls_val = defaultdict(list), defaultdict(list)
for i in range(len(train_ds)):
    y = label_idx(train_ds, i)
    per_cls_train[y].append(i)
for i in range(len(val_ds)):
    y = label_idx(val_ds, i)
    per_cls_val[y].append(i)

train_idx, val_idx = [], []
for c in range(num_classes):
    random.shuffle(per_cls_train[c]); random.shuffle(per_cls_val[c])
    train_idx += per_cls_train[c][:N_PER_CLASS_TRAIN]
    val_idx   += per_cls_val[c][:N_PER_CLASS_VAL]

print(f"Subtrain={len(train_idx)} | Subval={len(val_idx)}")
train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(Subset(val_ds,   val_idx),   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Modelo: congelar features y entrenar solo la capa final
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=weights if 'weights' in locals() else None)
for p in model.features.parameters(): p.requires_grad = False
in_feats = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_feats, num_classes))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)

def acc_top1(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

for ep in range(1, EPOCHS+1):
    # ---- train ----
    model.train(); tr_loss=tr_acc=n=0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward(); optimizer.step()
        b = yb.size(0); tr_loss += loss.item()*b; tr_acc += acc_top1(logits,yb)*b; n += b
    tr_loss/=n; tr_acc/=n

    # ---- val ----
    model.eval(); va_loss=va_acc=n=0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            b = yb.size(0); va_loss += loss.item()*b; va_acc += acc_top1(logits,yb)*b; n += b
    va_loss/=n; va_acc/=n
    print(f"Epoch {ep:02d} | train {tr_loss:.4f}/{tr_acc:.3f} | val {va_loss:.4f}/{va_acc:.3f}")

# Guardar pesos y clases
torch.save({"state_dict": model.state_dict(), "num_classes": num_classes}, MODEL_OUT)
np.save(CLASSES_OUT, np.array(classes))
print("Guardado:", MODEL_OUT, CLASSES_OUT)