# app_torch.py  —  Predicción en consola (sin Flask)
import io, argparse
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms

MODEL_PATH   = "food101_torch.pth"   # generado por train_food101_light.py
CLASSES_PATH = "food101_classes.npy" # generado por train_food101_light.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Preprocesado idéntico al entrenamiento ---
def get_preprocess():
    try:
        w = models.MobileNet_V2_Weights.IMAGENET1K_V1
        return w.transforms()
    except Exception:
        return transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

# --- Reconstruir modelo y cargar checkpoint ---
def load_model_and_classes():
    classes = np.load(CLASSES_PATH, allow_pickle=True).tolist()
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    num_classes = ckpt.get("num_classes", len(classes))

    m = models.mobilenet_v2(weights=None)
    in_feats = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_feats, num_classes))
    m.load_state_dict(ckpt["state_dict"], strict=True)
    m.eval().to(DEVICE)
    return m, classes

# --- kcal (versión simple) ---
KCAL_DEFAULT = 200  # si no quieres reglas, usa un promedio fijo
def kcal_100g(label, default=KCAL_DEFAULT):
    # Si quieres reglas por alimento, reemplaza por un dict y busca por substring.
    return default

def predict(image_path, grams=150, topk=3):
    preprocess = get_preprocess()
    model, classes = load_model_and_classes()

    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).cpu().numpy()[0]

    idxs = probs.argsort()[::-1][:topk]
    results = []
    for i in idxs:
        label = classes[i]
        p = float(probs[i])
        k100 = kcal_100g(label)
        results.append({
            "label": label,
            "prob": round(p, 4),
            "kcal_100g": k100,
            "grams": int(grams),
            "kcal": round(k100 * (grams/100.0), 1),
        })
    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="ruta a la imagen")
    ap.add_argument("--grams", type=float, default=150, help="porción en gramos")
    args = ap.parse_args()

    out = predict(args.img, grams=args.grams, topk=3)
    print("Top-3:")
    for r in out:
        print(f"- {r['label']}: prob={r['prob']}  kcal/100g={r['kcal_100g']}  "
              f"porción={r['grams']}g  kcal={r['kcal']}")
