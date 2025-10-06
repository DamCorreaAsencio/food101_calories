# Food-101 Calories (MVP)
Clasificador ligero de alimentos con MobileNetV2 entrenado sobre Food-101 y estimación de calorías por porción.
Incluye:

train_food101_light.py → entrena (transfer learning, solo PyTorch, CPU ok).

>.  
>├─ train_food101_light.py        --# Entrenamiento (MobileNetV2 -> 101 clases)  
>├─ app_torch.py                  --# Predicción por consola (top-3 + kcal)  
>├─ food101_torch.pth             --# (incluido) pesos del modelo (~9 MB)  
>├─ food101_classes.npy           --# (incluido) nombres de clases  
>├─ requirements.txt              --# dependencias mínimas  
>└─ .gitignore
## 
## Requisitos 

- Python 3.12 o 3.13

- Windows/PowerShell o cualquier OS con Python

- (Primera ejecución) Internet para descargar los pesos de MobileNetV2  

### Instala dependencias:
>python -m pip install --upgrade pip  
>python -m pip install pillow numpy flask  
>python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  

### Dataset

Descarga y extrae Food-101 y colócalo así:
><RUTA_BASE>\food-101\  
  ├─ images\...(carpetas por clase)  
  └─ meta\train.txt, test.txt, classes.txt, ...  

#### ¡OJO, PARA EXTRAER EL DATASET CORRECTAMENTE!
#### Crear carpeta destino (si no existe). Adaptar tu ruta de extracción.
>New-Item -ItemType Directory -Path "D:\Datasets" -Force | Out-Null    

#### Extraer el .tar.gz completo
>tar -xf "D:\Descargas\food-101.tar.gz" -C "D:\Datasets"    

#### Verificación rápida
>Get-ChildItem "D:\Datasets\food-101\meta"  
>Get-ChildItem "D:\Datasets\food-101\images" | Select-Object -First 5  

En mi equipo uso D:\Datasets\food-101\....  
El script acepta la ruta con --root (no es obligatorio que coincida con la mía); pero por si acaso.  
## 
## ENTRENAMIENTO
### Windows (mi caso)  
>python train_food101_light.py --root "D:\Datasets"  

### Ejemplos en otras máquinas  
>python train_food101_light.py --root "/home/usuario/datasets"     
## Predicción por consola  
>Predicción por consola (Top-3 + kcal)
